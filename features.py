"""Feature engineering, step 1: alarm-state features of the oblast itself, known at time t.

Input : data/processed/oblast_alarm_intervals.csv  (from preprocess.py; UTC)
        data/processed/hourly_grid.csv             (from build_dataset.py)
Output: data/processed/features_v1.csv  (oblast, t + FEATURE_COLS; same rows and order as the grid)

Features of row (oblast, t), all computed from the oblast's own alarm history:
  hours_since_last_end     hours since the latest alarm that FINISHED at or before t
  last_alarm_duration_h    duration of that alarm, in hours
  n_starts_{1,3,6,24,168}h number of alarm starts in (t - w, t]
  alarm_share_{24,168}h    share of the window (t - w, t] spent under alarm

Leakage rules (the contract of this module):
  * Only starts <= t are counted. A start at exactly t is known at t; labels use (t, t + N], so the
    two never overlap.
  * A finish time is read only if finished_at <= t. An alarm still running at t contributes just the
    part before t (clipped at t); its finished_at, which is in the future, is never used.
  * Labels (y_*) are never read here. Do not build features from rolling means of y_*: y_6h at t - 1
    already looks 5 hours past t.

Known data-quality caveat (not leakage): alarms that were "stuck" during the quarantine window
(see build_dataset.py) are closed on 2026-09-02, so rows shortly after the window carry inflated
alarm_share_* and last_alarm_duration_h for those oblasts.

Run from the repository root:
    python features.py reports/features_v1.txt   # report as UTF-8 file
    python features.py                           # report to the terminal
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from splits import GRID_PATH

INTERVALS_PATH = Path("data/processed/oblast_alarm_intervals.csv")
OUT_PATH = Path("data/processed/features_v1.csv")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
H = 3600.0  # seconds per hour
COUNT_WINDOWS_H = (1, 3, 6, 24, 168)
SHARE_WINDOWS_H = (24, 168)
FEATURE_COLS = (
    ["hours_since_last_end", "last_alarm_duration_h"]
    + [f"n_starts_{w}h" for w in COUNT_WINDOWS_H]
    + [f"alarm_share_{w}h" for w in SHARE_WINDOWS_H]
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def to_seconds(s: pd.Series) -> np.ndarray:
    return (s - EPOCH).dt.total_seconds().to_numpy()


def load_intervals() -> pd.DataFrame:
    ev = pd.read_csv(INTERVALS_PATH)
    for c in ["started_at", "finished_at"]:
        ev[c] = pd.to_datetime(ev[c], utc=True)
    return ev


def count_starts(starts: np.ndarray, lo_s: np.ndarray, hi_s: np.ndarray) -> np.ndarray:
    """Number of alarm starts in (lo_s, hi_s]. `starts` must be sorted (seconds)."""
    return np.searchsorted(starts, hi_s, side="right") - np.searchsorted(starts, lo_s, side="right")


def alarm_seconds_until(x: np.ndarray, starts: np.ndarray, ends: np.ndarray, cum: np.ndarray) -> np.ndarray:
    """Total time under alarm up to moment x. Intervals must be sorted and disjoint.

    Finished intervals (end <= x) count in full. The one interval that may be running at x counts
    only for x - start, so its end is never read.
    """
    k = np.searchsorted(ends, x, side="right")  # intervals with end <= x
    done = np.where(k > 0, cum[np.maximum(k - 1, 0)], 0.0)
    nxt = np.minimum(k, len(starts) - 1)
    running = np.where((k < len(starts)) & (starts[nxt] < x), x - starts[nxt], 0.0)
    return done + running


def oblast_features(t_s: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> dict[str, np.ndarray]:
    n = len(t_s)
    if len(starts) == 0:  # no known alarm yet
        feats = {c: np.full(n, np.nan) for c in FEATURE_COLS[:2]}
        feats.update({c: np.zeros(n) for c in FEATURE_COLS[2:]})
        return feats
    assert (starts[1:] >= ends[:-1]).all(), "intervals must be sorted and disjoint"
    dur = ends - starts
    cum = np.cumsum(dur)

    k = np.searchsorted(ends, t_s, side="right")  # alarms finished at or before t
    last = np.maximum(k - 1, 0)
    has_last = k > 0
    feats = {
        "hours_since_last_end": np.where(has_last, (t_s - ends[last]) / H, np.nan),
        "last_alarm_duration_h": np.where(has_last, dur[last] / H, np.nan),
    }
    for w in COUNT_WINDOWS_H:
        feats[f"n_starts_{w}h"] = count_starts(starts, t_s - w * H, t_s).astype(float)
    a_now = alarm_seconds_until(t_s, starts, ends, cum)
    for w in SHARE_WINDOWS_H:
        a_then = alarm_seconds_until(t_s - w * H, starts, ends, cum)
        feats[f"alarm_share_{w}h"] = (a_now - a_then) / (w * H)
    return feats


def build_features(keys: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Features for the rows of `keys` (columns oblast, t; t is tz-aware UTC). Row order is kept.

    `intervals` is everything the forecaster is allowed to know: check_features_leakage.py passes
    a truncated copy to prove that rows at t do not depend on anything after t.
    """
    out = keys[["oblast", "t"]].reset_index(drop=True)
    t_all = to_seconds(out["t"])
    cols = {c: np.full(len(out), np.nan) for c in FEATURE_COLS}
    by_oblast = {ob: g.sort_values("started_at") for ob, g in intervals.groupby("oblast")}
    empty = intervals.iloc[:0]
    for oblast, pos in out.groupby("oblast").indices.items():
        g = by_oblast.get(oblast, empty)
        feats = oblast_features(t_all[pos], to_seconds(g["started_at"]), to_seconds(g["finished_at"]))
        for c in FEATURE_COLS:
            cols[c][pos] = feats[c]
    return pd.concat([out, pd.DataFrame(cols)], axis=1)


def main() -> None:
    ev = load_intervals()
    grid = pd.read_csv(GRID_PATH, usecols=["oblast", "t", "active_at_t", "quarantined"])
    grid["t"] = pd.to_datetime(grid["t"], utc=True)
    feats = build_features(grid[["oblast", "t"]], ev)
    assert len(feats) == len(grid)

    section("1. Overview")
    print(f"Rows: {len(feats):,}; oblasts: {feats['oblast'].nunique()}; features: {len(FEATURE_COLS)}")
    print(f"Grid: {grid['t'].min()} -> {grid['t'].max()}")
    print("\nMissing values per feature (all rows):")
    print(feats[FEATURE_COLS].isna().sum().to_string())
    print("\nOnly hours_since_last_end and last_alarm_duration_h may be NA: an oblast has no finished alarm yet.")

    section("2. Distribution on eligible rows (no alarm at t, not quarantined)")
    elig = ~grid["active_at_t"].to_numpy() & ~grid["quarantined"].to_numpy()
    print(f"Eligible rows: {int(elig.sum()):,}")
    print(feats.loc[elig, FEATURE_COLS].describe().T.round(3).to_string())

    section("3. Output")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(feats):,} rows to {OUT_PATH}\nDone.")


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
