"""Independent leakage check for the walk-forward folds defined in splits.py.

Idea: re-compute every TRAINING label in a "truncated world" that contains only alarm
starts strictly before the fold start (all a forecaster could know at that moment).
If any training label changes, it was built from information inside the evaluation fold.

Negative control: the same check WITHOUT the embargo must find mismatches. If it does not,
the check itself is blind and its "0 mismatches" result proves nothing.

Run from the repository root:
    python check_splits_leakage.py
"""
import numpy as np
import pandas as pd

from splits import GRID_PATH, HORIZONS_H, test_folds, validation_folds

INTERVALS_PATH = "data/processed/oblast_alarm_intervals.csv"
NS_PER_H = 3_600_000_000_000
NO_NEXT = np.iinfo(np.int64).max  # sentinel: no known alarm start after t


def to_ns(s: pd.Series) -> np.ndarray:
    """tz-aware datetime Series -> int64 nanoseconds since epoch (independent of pandas resolution)."""
    return s.dt.tz_convert(None).astype("datetime64[ns]").to_numpy().astype("int64")


def recompute_label(t_ns: np.ndarray, starts_ns: np.ndarray, cutoff_ns: int, horizon_h: int) -> np.ndarray:
    """1 if a known alarm start (strictly before cutoff) lies in (t, t + horizon]."""
    known = starts_ns[starts_ns < cutoff_ns]
    idx = np.searchsorted(known, t_ns, side="right")  # first known start strictly after t
    nxt = np.where(idx < len(known), known[np.minimum(idx, len(known) - 1)], NO_NEXT)
    return (nxt <= t_ns + horizon_h * NS_PER_H).astype(int)


def count_mismatches(grid, starts_by_oblast, train_mask, cutoff, label, horizon_h):
    total = bad = 0
    for oblast, g in grid[train_mask].groupby("oblast"):
        rec = recompute_label(to_ns(g["t"]), starts_by_oblast[oblast], cutoff.value, horizon_h)
        bad += int((rec != g[label].to_numpy()).sum())
        total += len(g)
    return total, bad


def main() -> None:
    grid = pd.read_csv(GRID_PATH, usecols=["oblast", "t"] + [f"y_{n}h" for n in HORIZONS_H])
    grid["t"] = pd.to_datetime(grid["t"], utc=True)
    ev = pd.read_csv(INTERVALS_PATH, usecols=["oblast", "started_at"])
    ev["started_at"] = pd.to_datetime(ev["started_at"], utc=True)
    starts = {ob: np.sort(to_ns(g["started_at"])) for ob, g in ev.groupby("oblast")}

    rows = []
    for fold in validation_folds() + test_folds():
        for n in HORIZONS_H:
            label = f"y_{n}h"
            has_label = grid[label].notna()
            real = has_label & (grid["t"] < fold.train_end)   # what splits.py uses
            naive = has_label & (grid["t"] < fold.start)      # no embargo (negative control)
            n_real, bad_real = count_mismatches(grid, starts, real, fold.start, label, n)
            n_naive, bad_naive = count_mismatches(grid, starts, naive, fold.start, label, n)
            rows.append({"fold": fold.name, "label": label,
                         "train_rows": n_real, "mismatch_with_embargo": bad_real,
                         "train_rows_no_embargo": n_naive, "mismatch_no_embargo": bad_naive})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    assert (out["mismatch_with_embargo"] == 0).all(), "LEAK: a training label depends on the evaluation fold"
    assert (out["mismatch_no_embargo"] > 0).any(), "Check is blind: negative control found nothing"
    print("\nOK: no leakage with embargo; the check detects leakage without it.")


if __name__ == "__main__":
    main()
