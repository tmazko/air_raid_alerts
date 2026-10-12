"""LightGBM v2: does adding alarm-state features (features.py) beat the baselines?

Compared on the SAME evaluation rows, per walk-forward fold, each model refitted per fold:
  B1_oblast, B3_oblast_28d   baselines from baselines.py (reference)
  LGBM_basic                 features: oblast, hour (as in model_lgbm.py)
  LGBM_state                 features: oblast, hour + FEATURE_COLS from features.py

Leakage:
  * Training rows: labelled rows with t < fold.train_end (6 h embargo), via splits.fold_masks.
  * All features are functions of data known at t (verified by check_features_leakage.py).
  * Hyperparameters are fixed and NOT tuned here, so nothing is fitted to the evaluation folds.
  * USE_TEST = False: only VALIDATION folds are evaluated. Model choices are made on these.
    Flip USE_TEST once, after the configuration is frozen, to get the final test numbers.

Run from the repository root (needs hourly_grid.csv and features_v1.csv in data/processed/):
    python model_lgbm_v2.py reports/model_lgbm_v2.txt   # report as UTF-8 file
    python model_lgbm_v2.py                             # report to the terminal

Output: data/processed/lgbm_v2_metrics.csv
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import collect_predictions, load_grid as load_baseline_grid, score
from features import FEATURE_COLS
from neighbors import NEIGHBOR_COLS
from splits import HORIZONS_H, fold_masks, test_folds, validation_folds

FEATURES_PATH = Path("data/processed/features_v1.csv")
NEIGHBORS_PATH = Path("data/processed/neighbor_features_v1.csv")
METRICS_PATH = Path("data/processed/lgbm_v2_metrics.csv")
PREDS_PATH = Path("data/processed/lgbm_v2_preds.csv")  # fold, label, model, oblast, y, p (for bootstrap)
USE_TEST = True
REFERENCE = ("B1_oblast", "B3_oblast_28d")
MODEL_ORDER = [*REFERENCE, "LGBM_basic", "LGBM_state", "LGBM_nb"]
FEATURE_SETS = {
    "LGBM_basic": ["oblast", "hour"],
    "LGBM_state": ["oblast", "hour"] + FEATURE_COLS,
    "LGBM_nb": ["oblast", "hour"] + FEATURE_COLS + NEIGHBOR_COLS,
}
LGBM_PARAMS = dict(n_estimators=100, learning_rate=0.05, num_leaves=15,
                   random_state=42, n_jobs=-1, verbose=-1)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_grid() -> pd.DataFrame:
    grid = load_baseline_grid()  # oblast, t, active_at_t, quarantined, y_*, hour
    feats = pd.read_csv(FEATURES_PATH)
    feats["t"] = pd.to_datetime(feats["t"], utc=True)
    assert feats[["oblast", "t"]].equals(grid[["oblast", "t"]]), \
        "features_v1.csv is not row-aligned with hourly_grid.csv; rerun features.py"
    
    nbf = pd.read_csv(NEIGHBORS_PATH)
    nbf["t"] = pd.to_datetime(nbf["t"], utc=True)
    assert nbf[["oblast", "t"]].equals(grid[["oblast", "t"]]), \
        "neighbor_features_v1.csv is not row-aligned with hourly_grid.csv; rerun neighbors.py"
    return pd.concat([grid, feats[FEATURE_COLS], nbf[NEIGHBOR_COLS]], axis=1)


def lgbm_predictions(grid: pd.DataFrame, folds) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Fixed category list so that train and eval use identical integer codes.
    oblast_type = pd.CategoricalDtype(sorted(grid["oblast"].unique()))
    parts, importances = [], []
    for fold in folds:
        for n in HORIZONS_H:
            label = f"y_{n}h"
            tr, ev = fold_masks(grid, fold, label)
            assert grid.loc[tr, "t"].max() < fold.train_end <= fold.start - pd.Timedelta(hours=max(HORIZONS_H))
            assert grid.loc[ev, "t"].min() >= fold.start and grid.loc[ev, "t"].max() < fold.end
            y_tr = grid.loc[tr, label].astype(int)
            y_ev = grid.loc[ev, label].astype(int).to_numpy()
            for name, cols in FEATURE_SETS.items():
                X_tr = grid.loc[tr, cols].assign(oblast=lambda d: d["oblast"].astype(oblast_type))
                X_ev = grid.loc[ev, cols].assign(oblast=lambda d: d["oblast"].astype(oblast_type))
                model = lgb.LGBMClassifier(importance_type="gain", **LGBM_PARAMS).fit(X_tr, y_tr)
                p = model.predict_proba(X_ev)[:, 1]
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                parts.append(pd.DataFrame({"fold": fold.name, "label": label, "model": name,
                                           "oblast": grid.loc[ev, "oblast"].to_numpy(), "y": y_ev, "p": p}))
                if name == "LGBM_nb":
                    gain = np.asarray(model.feature_importances_, dtype=float)
                    importances.append(pd.DataFrame({"fold": fold.name, "label": label, "feature": cols,
                                                     "gain_share": gain / gain.sum()}))
    return pd.concat(parts, ignore_index=True), pd.concat(importances, ignore_index=True)


def assert_same_rows(preds: pd.DataFrame) -> None:
    """Every model must have been scored on exactly the same rows in the same order."""
    for (fold, label), g in preds.groupby(["fold", "label"]):
        ref = None
        for model, m in g.groupby("model"):
            cur = (m["oblast"].to_numpy(), m["y"].to_numpy())
            if ref is None:
                ref = cur
            else:
                assert len(cur[1]) == len(ref[1]) and (cur[0] == ref[0]).all() and (cur[1] == ref[1]).all(), \
                    f"{model} was scored on different rows in {fold}/{label}"


def summarize(preds: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for gname, fold_names in groups.items():
        sub_g = preds[preds["fold"].isin(fold_names)]
        for (label, model), sub in sub_g.groupby(["label", "model"]):
            rows.append({"group": gname, "label": label, "model": model, **score(sub)})
    out = pd.DataFrame(rows)
    ref = out[out["model"] == "B3_oblast_28d"].set_index(["group", "label"])["brier"]
    out["brier_vs_B3"] = 1 - out["brier"] / out.set_index(["group", "label"]).index.map(ref).to_numpy()
    return out


def main() -> None:
    grid = load_grid()
    folds = test_folds() if USE_TEST else validation_folds()
    names = [f.name for f in folds]
    pooled = names[:3] if USE_TEST else names
    groups = {f"POOLED {pooled[0]}-{pooled[-1]}": pooled}
    groups.update({n: [n] for n in names})

    section("1. Setup")
    print(f"Evaluating: {'TEST folds (final run)' if USE_TEST else 'VALIDATION folds (model selection)'}")
    print(f"Features LGBM_basic: {FEATURE_SETS['LGBM_basic']}")
    print(f"Features LGBM_state: + {FEATURE_COLS}")
    print(f"Fixed LightGBM params (not tuned): {LGBM_PARAMS}")

    lgbm_preds, importances = lgbm_predictions(grid, folds)
    base = collect_predictions(grid, folds).rename(columns={"baseline": "model"})
    preds = pd.concat([base[base["model"].isin(REFERENCE)], lgbm_preds], ignore_index=True)
    assert_same_rows(preds)
    PREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PREDS_PATH, index=False)
    res = summarize(preds, groups)

    cols = ["model", "n", "pos_rate", "brier", "brier_vs_B3", "log_loss", "pr_auc", "roc_auc",
            "within_oblast_roc_auc"]
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    for n in HORIZONS_H:
        label = f"y_{n}h"
        pooled_name = next(iter(groups))
        section(f"2. {pooled_name}, label {label}  (brier_vs_B3 > 0: better than B3)")
        r = res[(res["label"] == label) & (res["group"] == pooled_name)]
        print(r.assign(_o=r["model"].map(order)).sort_values("_o")[cols].round(4).to_string(index=False))

    for n in HORIZONS_H:
        label = f"y_{n}h"
        section(f"3. Brier per fold, label {label} (consistency across folds)")
        r = res[res["label"] == label].pivot(index="group", columns="model", values="brier")
        print(r.reindex(list(groups))[MODEL_ORDER].round(4).to_string())

    section("4. LGBM_state: mean gain share per feature (across folds)")
    imp = importances.groupby(["feature", "label"])["gain_share"].mean().unstack("label")
    print(imp.sort_values(imp.columns[0], ascending=False).round(3).to_string())

    section("5. Output")
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