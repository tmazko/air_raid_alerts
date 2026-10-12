"""Data-quality scan: find suspicious windows (possible feed outages or stuck alarms).

Input : data/processed/oblast_alarm_intervals.csv  (from preprocess.py)
        data/processed/hourly_grid.csv             (from build_dataset.py)
Read-only. Prints a report; changes no data.

Two checks:
  1. Runs of consecutive hours with ZERO new alarm starts across all oblasts
     (a pooled gap is unlikely to be real calm; it may mean nothing was recorded).
  2. Very long union events, and clusters of them overlapping in time across oblasts
     (synchronized multi-day "alarms" may be stuck records rather than real alarms).

Run from the repository root:
    python quality_scan.py reports/quality_scan.txt   # report as UTF-8 file
    python quality_scan.py                            # report to the terminal
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INTERVALS_PATH = Path("data/processed/oblast_alarm_intervals.csv")
GRID_PATH = Path("data/processed/hourly_grid.csv")
MIN_ZERO_RUN_H = 12  # report pooled zero-start runs at least this long
LONG_EVENT_H = 48  # report union events at least this long
TOP_N = 25
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 120)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def zero_start_runs(ev: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    hours = pd.DatetimeIndex(sorted(grid["t"].unique()))
    starts = pd.Series(1, index=ev["started_at"].dt.floor("h")).groupby(level=0).sum()
    cnt = starts.reindex(hours, fill_value=0).to_numpy()
    n_active = grid.groupby("t")["active_at_t"].sum().reindex(hours).to_numpy()

    z = (cnt == 0).astype(int)
    d = np.diff(np.concatenate(([0], z, [0])))
    run_start = np.where(d == 1)[0]
    run_end = np.where(d == -1)[0]  # exclusive
    rows = []
    for a, b in zip(run_start, run_end):
        if b - a >= MIN_ZERO_RUN_H:
            rows.append({
                "first_hour": hours[a],
                "last_hour": hours[b - 1],
                "length_h": int(b - a),
                "mean_active_oblasts": round(float(n_active[a:b].mean()), 2),
                "max_active_oblasts": int(n_active[a:b].max()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ev = pd.read_csv(INTERVALS_PATH)
    for c in ["started_at", "finished_at"]:
        ev[c] = pd.to_datetime(ev[c], utc=True)
    grid = pd.read_csv(GRID_PATH, usecols=["t", "active_at_t"])
    grid["t"] = pd.to_datetime(grid["t"], utc=True)

    # ------------------------------------------------------------------ 1
    section(f"1. Pooled zero-start runs (all oblasts together), length >= {MIN_ZERO_RUN_H} h")
    runs = zero_start_runs(ev, grid)
    print(f"Runs found: {len(runs)}")
    if len(runs):
        print("\nRuns by year:")
        print(runs.groupby(runs["first_hour"].dt.year).agg(
            runs=("length_h", "size"), total_hours=("length_h", "sum"), longest_h=("length_h", "max")))
        print(f"\nTop {TOP_N} longest runs (mean_active_oblasts: oblasts in alarm during the run):")
        print(runs.sort_values("length_h", ascending=False).head(TOP_N).to_string(index=False))

    # ------------------------------------------------------------------ 2
    section(f"2. Union events >= {LONG_EVENT_H} h")
    ev["dur_h"] = (ev["finished_at"] - ev["started_at"]).dt.total_seconds() / 3600
    long_ev = ev[ev["dur_h"] >= LONG_EVENT_H].sort_values("started_at").reset_index(drop=True)
    print(f"Events >= {LONG_EVENT_H} h: {len(long_ev)} of {len(ev):,}")
    if len(long_ev):
        print("\nBy oblast:")
        print(long_ev["oblast"].value_counts().to_string())
        print("\nBy year:")
        print(long_ev["started_at"].dt.year.value_counts().sort_index().to_string())

        # Clusters of long events overlapping in time (across oblasts).
        start_s = (long_ev["started_at"] - EPOCH).dt.total_seconds()
        end_s = (long_ev["finished_at"] - EPOCH).dt.total_seconds()
        prev_max_end = end_s.cummax().shift()
        cluster = (prev_max_end.isna() | (start_s > prev_max_end)).cumsum()
        cl = long_ev.groupby(cluster).agg(
            first_start=("started_at", "min"),
            last_end=("finished_at", "max"),
            events=("oblast", "size"),
            oblasts=("oblast", lambda s: ", ".join(sorted(s.unique()))),
        )
        cl["n_oblasts"] = cl["oblasts"].str.count(",") + 1
        print("\nClusters of long events overlapping in time (n_oblasts >= 2):")
        multi = cl[cl["n_oblasts"] >= 2]
        print(multi.to_string(index=False) if len(multi) else "none")
        print(f"\nTop {TOP_N} longest events:")
        top = long_ev.sort_values("dur_h", ascending=False).head(TOP_N).copy()
        top["dur_h"] = top["dur_h"].round(1)
        print(top[["oblast", "started_at", "finished_at", "n_rows", "dur_h"]].to_string(index=False))

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
