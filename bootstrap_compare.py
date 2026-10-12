"""Paired block bootstrap: is the Brier improvement of one model over another larger than noise?

This is evaluation uncertainty only. It refits nothing and uses no information beyond the
predictions already made by model_lgbm_v2.py, so it cannot introduce leakage.

Why blocks: alarms are strongly correlated across oblasts (one attack hits many regions) and across
neighbouring days. Resampling single rows would give far too narrow intervals. Instead, rows are
grouped into blocks of BLOCK_DAYS consecutive UTC days (all oblasts together) and whole blocks are
resampled with replacement. The two models are always scored on the same resampled rows (paired).
Block sizes 1, 3 and 7 days are reported as a sensitivity check; they are fixed a priori.

Pooled folds: every fold in the predictions file except EXCLUDE (test4 has quarantined rows).

Run from the repository root (needs data/processed/lgbm_v2_preds.csv and hourly_grid.csv):
    python bootstrap_compare.py reports/bootstrap_compare.txt   # report as UTF-8 file
    python bootstrap_compare.py                                 # report to the terminal
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import load_grid
from splits import HORIZONS_H, fold_masks, test_folds, validation_folds

PREDS_PATH = Path("data/processed/lgbm_v2_preds.csv")
EXCLUDE = {"test4"}
PAIRS = (("LGBM_nb", "LGBM_state"), ("LGBM_nb", "B3_oblast_28d"), ("LGBM_state", "B3_oblast_28d"))
BLOCK_DAYS = (1, 3, 7)
N_BOOT = 2000
SEED = 0
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def squared_errors(preds: pd.DataFrame, grid: pd.DataFrame, label: str, fold_names: list[str]) -> pd.DataFrame:
    """One row per evaluated (oblast, t): day index and squared error of every model."""
    folds = {f.name: f for f in validation_folds() + test_folds()}
    models = sorted(preds["model"].unique())
    parts = []
    for name in fold_names:
        _, ev = fold_masks(grid, folds[name], label)
        ref = grid.loc[ev, ["oblast", "t"]]
        sub = preds[(preds["fold"] == name) & (preds["label"] == label)]
        d = {"day": ((ref["t"].dt.floor("D") - EPOCH).dt.days).to_numpy()}
        for m in models:
            s = sub[sub["model"] == m]
            assert len(s) == len(ref) and (s["oblast"].to_numpy() == ref["oblast"].to_numpy()).all(), \
                f"predictions of {m} do not match the evaluation rows of {name}/{label}"
            d[f"se_{m}"] = (s["p"].to_numpy() - s["y"].to_numpy()) ** 2
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True)


def paired_bootstrap(se: pd.DataFrame, a: str, b: str, block_days: int, rng) -> dict:
    block = (se["day"] // block_days).to_numpy()
    _, inv = np.unique(block, return_inverse=True)
    n_blocks = inv.max() + 1
    cnt = np.bincount(inv, minlength=n_blocks).astype(float)
    sum_a = np.bincount(inv, weights=se[f"se_{a}"].to_numpy(), minlength=n_blocks)
    sum_b = np.bincount(inv, weights=se[f"se_{b}"].to_numpy(), minlength=n_blocks)
    idx = rng.integers(0, n_blocks, size=(N_BOOT, n_blocks))
    n_boot = cnt[idx].sum(axis=1)
    brier_a = sum_a[idx].sum(axis=1) / n_boot
    brier_b = sum_b[idx].sum(axis=1) / n_boot
    diff = brier_a - brier_b                 # negative: a is better than b
    rel = 1 - brier_a / brier_b              # positive: a is better than b
    return {
        "blocks": n_blocks, "brier_a": sum_a.sum() / cnt.sum(), "brier_b": sum_b.sum() / cnt.sum(),
        "diff": (sum_a.sum() - sum_b.sum()) / cnt.sum(),
        "diff_lo95": np.percentile(diff, 2.5), "diff_hi95": np.percentile(diff, 97.5),
        "rel_impr_%": 100 * (1 - sum_a.sum() / sum_b.sum()),
        "rel_lo95_%": 100 * np.percentile(rel, 2.5), "rel_hi95_%": 100 * np.percentile(rel, 97.5),
        "P(a better)": float((diff < 0).mean()),
    }


def main() -> None:
    preds = pd.read_csv(PREDS_PATH)
    grid = load_grid()
    fold_names = [f for f in dict.fromkeys(preds["fold"]) if f not in EXCLUDE]
    rng = np.random.default_rng(SEED)

    section("1. Setup")
    print(f"Pooled folds: {fold_names} (excluded: {sorted(EXCLUDE)})")
    print(f"Bootstrap samples: {N_BOOT}; block sizes (days): {BLOCK_DAYS}; seed {SEED}")
    print("diff = Brier(a) - Brier(b); negative means a is better. 95% intervals are percentile intervals.")

    rows = []
    for n in HORIZONS_H:
        label = f"y_{n}h"
        se = squared_errors(preds, grid, label, fold_names)
        for a, b in PAIRS:
            for k in BLOCK_DAYS:
                rows.append({"label": label, "a": a, "b": b, "block_days": k,
                             "n": len(se), **paired_bootstrap(se, a, b, k, rng)})
    res = pd.DataFrame(rows)

    for (a, b), g in res.groupby(["a", "b"], sort=False):
        section(f"2. {a} vs {b}")
        print(g.drop(columns=["a", "b"]).round(4).to_string(index=False))
    print("\nDone.")


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