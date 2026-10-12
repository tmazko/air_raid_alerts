"""Baseline forecasters: lookup tables fitted on training rows only, evaluated per fold.

Every baseline is refitted for every walk-forward fold (splits.py) and sees ONLY:
  * train rows : labelled rows with t < fold.train_end (after the embargo), and
  * eval rows  : oblast, t, hour-of-day -- NO labels, so eval labels cannot be used by construction.
Predictions are static within a fold (no updating from the eval fold itself).

Baselines (probability that a new alarm starts in (t, t + N h]):
  B0_global       train positive rate, same for all rows
  B1_oblast       train positive rate per oblast
  B2_oblast_hour  train rate per (oblast, UTC hour of day), smoothed toward the B1 rate
  B3_oblast_28d   rate per oblast over the last 28 days before train_end, smoothed toward B1

Smoothing: (positives + K * prior) / (rows + K), K = SMOOTH_K. K and RECENT_DAYS are fixed a priori,
not tuned on any metric.

Metrics: Brier score, Brier skill vs B0, log loss, PR-AUC (average precision), ROC-AUC, within-oblast ROC-AUC.
Reported: pooled test1-3 (primary), every fold separately, pooled test1-4 (supplementary, test4 has
quarantined rows), pooled validation.

Run from the repository root (needs data/processed/hourly_grid.csv and scikit-learn):
    python baselines.py reports/baselines.txt   # report as UTF-8 file
    python baselines.py                         # report to the terminal

Output: data/processed/baseline_metrics.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from splits import GRID_PATH, HORIZONS_H, fold_masks, test_folds, validation_folds

METRICS_PATH = Path("data/processed/baseline_metrics.csv")
SMOOTH_K = 20
RECENT_DAYS = 28
EPS = 1e-6
BASELINES = ("B0_global", "B1_oblast", "B2_oblast_hour", "B3_oblast_28d")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# --------------------------------------------------------------------------- baselines
# Signature: (train, ev, label, train_end) -> probabilities for the rows of ev.
# `train` has labels; `ev` has only oblast, t, hour.

def _oblast_rate(train: pd.DataFrame, label: str) -> pd.Series:
    return train.groupby("oblast")[label].mean()


def b0_global(train, ev, label, train_end):
    return np.full(len(ev), train[label].mean())


def b1_oblast(train, ev, label, train_end):
    return ev["oblast"].map(_oblast_rate(train, label)).fillna(train[label].mean()).to_numpy()


def _smoothed(ev, stats, keys, prior):
    """(pos + K * prior) / (n + K) for each ev row; unseen keys fall back to the prior."""
    m = ev[keys].merge(stats, on=keys, how="left")  # left merge keeps the order of ev
    pos = m["pos"].fillna(0).to_numpy()
    n = m["n"].fillna(0).to_numpy()
    return (pos + SMOOTH_K * prior) / (n + SMOOTH_K)


def b2_oblast_hour(train, ev, label, train_end):
    prior = b1_oblast(train, ev, label, train_end)
    stats = train.groupby(["oblast", "hour"])[label].agg(pos="sum", n="count").reset_index()
    return _smoothed(ev, stats, ["oblast", "hour"], prior)


def b3_oblast_recent(train, ev, label, train_end):
    prior = b1_oblast(train, ev, label, train_end)
    recent = train[train["t"] >= train_end - pd.Timedelta(days=RECENT_DAYS)]
    stats = recent.groupby("oblast")[label].agg(pos="sum", n="count").reset_index()
    return _smoothed(ev, stats, ["oblast"], prior)


MODELS = dict(zip(BASELINES, (b0_global, b1_oblast, b2_oblast_hour, b3_oblast_recent)))


# --------------------------------------------------------------------------- evaluation
def load_grid() -> pd.DataFrame:
    cols = ["oblast", "t", "active_at_t", "quarantined"] + [f"y_{n}h" for n in HORIZONS_H]
    grid = pd.read_csv(GRID_PATH, usecols=cols)
    grid["t"] = pd.to_datetime(grid["t"], utc=True)
    grid["hour"] = grid["t"].dt.hour  # UTC hour of day; known at t, no leakage
    return grid


def get_within_oblast_roc_auc(df: pd.DataFrame) -> float:
    aucs = []
    for _, grp in df.groupby("oblast"):
        if grp["y"].nunique() > 1:  # Need both positive and negative examples
            aucs.append(roc_auc_score(grp["y"], grp["p"]))
    return np.mean(aucs) if aucs else np.nan


def score(sub: pd.DataFrame) -> dict:
    y = sub["y"].to_numpy()
    p = sub["p"].to_numpy()
    return {
        "n": len(y), "pos_rate": y.mean(),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "within_oblast_roc_auc": get_within_oblast_roc_auc(sub),
    }


def collect_predictions(grid: pd.DataFrame, folds) -> pd.DataFrame:
    parts = []
    for fold in folds:
        for n in HORIZONS_H:
            label = f"y_{n}h"
            tr, ev = fold_masks(grid, fold, label)
            train = grid.loc[tr, ["oblast", "t", "hour", label]]
            ev_rows = grid.loc[ev, ["oblast", "t", "hour"]]  # labels deliberately NOT passed
            
            # Leakage guards on what the baselines may see.
            assert train["t"].max() < fold.train_end <= fold.start - pd.Timedelta(hours=max(HORIZONS_H))
            assert ev_rows["t"].min() >= fold.start and ev_rows["t"].max() < fold.end
            y = grid.loc[ev, label].astype(int).to_numpy()
            
            for name, fn in MODELS.items():
                p = fn(train, ev_rows, label, fold.train_end)
                assert len(p) == len(y) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                parts.append(pd.DataFrame({
                    "fold": fold.name, 
                    "label": label, 
                    "baseline": name,
                    "oblast": ev_rows["oblast"].to_numpy(),
                    "y": y, 
                    "p": p
                }))
    return pd.concat(parts, ignore_index=True)


def summarize(preds: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for gname, fold_names in groups.items():
        sub_g = preds[preds["fold"].isin(fold_names)]
        for label, sub_l in sub_g.groupby("label"):
            for base, sub in sub_l.groupby("baseline"):
                rows.append({"group": gname, "label": label, "baseline": base,
                             **score(sub)})
    out = pd.DataFrame(rows)
    ref = out[out["baseline"] == "B0_global"].set_index(["group", "label"])["brier"]
    out["brier_skill_vs_B0"] = 1 - out["brier"] / out.set_index(["group", "label"]).index.map(ref).to_numpy()
    return out


def fold_accounting(grid: pd.DataFrame, folds) -> pd.DataFrame:
    rows = []
    for f in folds:
        w = grid[(grid["t"] >= f.start) & (grid["t"] < f.end)]
        quar = int(w["quarantined"].sum())
        active = int((w["active_at_t"] & ~w["quarantined"]).sum())
        evaluated = len(w) - quar - active
        assert evaluated == int(w["y_1h"].notna().sum())
        rows.append({"fold": f.name, "oblast_hours": len(w), "quarantined(excluded)": quar,
                     "alarm_active_at_t(excluded)": active, "evaluated": evaluated})
    return pd.DataFrame(rows)


def main() -> None:
    grid = load_grid()
    vfolds, tfolds = validation_folds(), test_folds()
    v_names, t_names = [f.name for f in vfolds], [f.name for f in tfolds]
    preds = collect_predictions(grid, vfolds + tfolds)

    groups = {"POOLED test1-3 (PRIMARY)": t_names[:3]}
    groups.update({n: [n] for n in t_names})
    groups["POOLED test1-4 (suppl.; test4 has quarantine)"] = t_names
    groups["POOLED valid1-4"] = v_names
    groups.update({n: [n] for n in v_names})
    res = summarize(preds, groups)

    section("1. Setup")
    print(f"Baselines: {', '.join(BASELINES)}; SMOOTH_K={SMOOTH_K}; RECENT_DAYS={RECENT_DAYS} (fixed, not tuned)")
    print("Training rows per fold: labelled rows with t < train_end; eval rows get no labels as input.")
    print("Hour of day is UTC (local time is UTC+2/+3, so DST blurs it slightly).")

    section("2. Rows evaluated vs excluded per fold (same for all horizons)")
    print(fold_accounting(grid, vfolds + tfolds).to_string(index=False))
    print("\ntest4: rows inside the data-quality quarantine are excluded, so its positive rate "
          "is not comparable with the other folds.")

    cols = ["group", "baseline", "n", "pos_rate", "brier", "brier_skill_vs_B0", "log_loss", "pr_auc", "roc_auc", "within_oblast_roc_auc"]
    for n in HORIZONS_H:
        label = f"y_{n}h"
        section(f"3. Results, label {label}  (PR-AUC of a random ranker = pos_rate)")
        r = res[res["label"] == label][cols]
        order = {g: i for i, g in enumerate(groups)}
        r = r.assign(_g=r["group"].map(order), _b=r["baseline"].map({b: i for i, b in enumerate(BASELINES)}))
        print(r.sort_values(["_g", "_b"]).drop(columns=["_g", "_b"]).round(4).to_string(index=False))

    section("4. Output")
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(METRICS_PATH, index=False)
    print(f"Wrote {len(res)} rows to {METRICS_PATH}\nDone.")


if __name__ == "__main__":
    out_file = None
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(out_path, "w", encoding="utf-8")
        sys.stdout = out_file
    try:
        main()
    finally:
        if out_file is not None:
            sys.stdout = sys.__stdout__
            out_file.close()
            print(f"Report written to {out_path}")