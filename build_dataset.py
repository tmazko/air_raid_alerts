"""Preprocessing, step 2: hourly (oblast, t) grid with the alarm state at t and targets.

Input : data/processed/oblast_alarm_intervals.csv  (from preprocess.py; UTC)
Output: data/processed/hourly_grid.csv

Columns of the output:
  oblast       oblast name
  t            grid timestamp, whole hours, UTC; the moment a forecast is made
  active_at_t  True if a union alarm is active at t, i.e. started_at <= t < finished_at.
               Only needs "the alarm has not finished yet at t", which is known at t.
  quarantined  True if t falls into a data-quality quarantine window (see QUARANTINE_WINDOWS).
               Alarm states and labels there are unreliable; such rows are never used.
  y_1h, y_3h, y_6h
               1 if a NEW alarm starts in (t, t + N hours], else 0.
               Empty (NA) when active_at_t or quarantined is True: those rows are not evaluated.
               These are LABELS built from the future. Never use them as features.

Right-censoring: the last grid point is floor(last known alarm start - max horizon), so that
no window (t, t + N] extends beyond the last data point and gets a false 0 label.

Run from the repository root:
    python build_dataset.py reports/build_dataset.txt   # report as UTF-8 file
    python build_dataset.py                             # report to the terminal
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/oblast_alarm_intervals.csv")
OUT_PATH = Path("data/processed/hourly_grid.csv")
HORIZONS_H = (1, 3, 6)
RECENT_WEEKS = 8
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

# Data-quality quarantine (evidence: quality_scan.py). From 2026-08-29 05:00 to 2026-09-02 13:00 UTC
# no new alarm start was recorded in ANY oblast, while 8 oblasts show continuous 105-118 h alarms that
# all close between 2026-09-02 13:56 and 19:01 UTC. Either records are missing or alarm states are
# stuck; in both cases states and labels there are unreliable. This is a manual data-integrity
# decision, not tuned on model metrics. Start = first hour of the zero-start run; end = first full
# hour after the last stuck alarm closed. Rows up to max(HORIZONS_H) hours before the start are
# also flagged, because their label windows reach into the quarantine.
QUARANTINE_WINDOWS = [
    (pd.Timestamp("2026-08-29 05:00", tz="UTC"), pd.Timestamp("2026-09-02 20:00", tz="UTC")),
]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def to_seconds(s: pd.Series) -> np.ndarray:
    return (s - EPOCH).dt.total_seconds().to_numpy()


def load_intervals() -> pd.DataFrame:
    ev = pd.read_csv(IN_PATH)
    for c in ["started_at", "finished_at"]:
        ev[c] = pd.to_datetime(ev[c], utc=True)
    return ev


def build_grid(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    data_end = ev["started_at"].max()
    first_t = ev["started_at"].min().floor("h")
    last_t = (data_end - pd.Timedelta(hours=max(HORIZONS_H))).floor("h")
    hours = pd.date_range(first_t, last_t, freq="h")
    t_s = np.asarray((hours - EPOCH).total_seconds())

    quarantined = np.zeros(len(hours), dtype=bool)
    margin = pd.Timedelta(hours=max(HORIZONS_H))
    for q_start, q_end in QUARANTINE_WINDOWS:
        quarantined |= np.asarray((hours >= q_start - margin) & (hours < q_end))

    frames = []
    for oblast, g in ev.groupby("oblast", sort=True):
        g = g.sort_values("started_at")
        starts = to_seconds(g["started_at"])
        ends = to_seconds(g["finished_at"])

        # Number of union events with start <= t; the event at index idx-1 is the latest one
        # that has started by t, and starts[idx] is the first start strictly after t.
        idx = np.searchsorted(starts, t_s, side="right")
        prev = idx - 1
        active = (prev >= 0) & (t_s < ends[np.clip(prev, 0, None)])
        next_start = np.full(len(t_s), np.inf)
        has_next = idx < len(starts)
        next_start[has_next] = starts[idx[has_next]]

        out = pd.DataFrame(
            {"oblast": oblast, "t": hours, "active_at_t": active, "quarantined": quarantined}
        )
        for n in HORIZONS_H:
            y = (next_start <= t_s + n * 3600).astype(float)
            y[active | quarantined] = np.nan
            out[f"y_{n}h"] = pd.Series(y).astype("Int8")
        frames.append(out)
    return pd.concat(frames, ignore_index=True), data_end, last_t


def main() -> None:
    ev = load_intervals()
    grid, data_end, last_t = build_grid(ev)
    y_cols = [f"y_{n}h" for n in HORIZONS_H]

    # ------------------------------------------------------------------ 1
    section("1. Grid")
    elig = grid[~grid["active_at_t"] & ~grid["quarantined"]]
    print(f"Oblasts: {grid['oblast'].nunique()}")
    print(f"Last known alarm start (data end): {data_end}")
    print(f"Grid: {grid['t'].min()} -> {grid['t'].max()}  (last t = floor(data end - {max(HORIZONS_H)} h))")
    print(f"Grid rows (oblast x hour): {len(grid):,}")
    print(f"  with an active alarm at t (not evaluated): {grid['active_at_t'].sum():,} "
          f"({grid['active_at_t'].mean():.1%})")
    print(f"  quarantined (data-quality window)        : {grid['quarantined'].sum():,}")
    print(f"  eligible rows (no alarm at t, not quarantined): {len(elig):,}")
    for q_start, q_end in QUARANTINE_WINDOWS:
        print(f"  quarantine window: [{q_start}, {q_end}) with {max(HORIZONS_H)} h label margin before it")

    # ------------------------------------------------------------------ 2
    section("2. Eligible rows and positive rate by year")
    e = elig.assign(year=elig["t"].dt.year)
    e[y_cols] = e[y_cols].astype(float)
    by_year = e.groupby("year").agg(
        eligible=("t", "size"), **{f"rate_{c}": (c, "mean") for c in y_cols}
    )
    by_year.insert(0, "grid_rows", grid.groupby(grid["t"].dt.year).size())
    by_year.insert(1, "active_share", grid.groupby(grid["t"].dt.year)["active_at_t"].mean())
    print(by_year.round(4))
    print("\nPositive rate by month (eligible rows):")
    by_month = e.groupby(e["t"].dt.strftime("%Y-%m")).agg(
        eligible=("t", "size"), **{f"rate_{c}": (c, "mean") for c in y_cols}
    )
    print(by_month.round(4))

    # ------------------------------------------------------------------ 3
    section(f"3. Last {RECENT_WEEKS} weeks by oblast")
    w_end = grid["t"].max()
    w_start = w_end - pd.Timedelta(weeks=RECENT_WEEKS)
    recent = grid[grid["t"] > w_start]
    print(f"Window: ({w_start}, {w_end}]")
    r = recent.copy()
    r[y_cols] = r[y_cols].astype(float)
    per_ob = r.groupby("oblast").agg(
        grid_rows=("t", "size"),
        active_share=("active_at_t", "mean"),
        quarantined=("quarantined", "sum"),
        eligible=("y_1h", "count"),
        **{f"pos_{c}": (c, "sum") for c in y_cols},
        **{f"rate_{c}": (c, "mean") for c in y_cols},
    )
    print(per_ob.sort_values("eligible").round(3))
    tot = r[r["y_1h"].notna()]
    print(f"\nPooled eligible rows: {len(tot):,}; positives " +
          ", ".join(f"{c}: {int(tot[c].sum()):,} ({tot[c].mean():.3f})" for c in y_cols))

    # ------------------------------------------------------------------ 4
    section(f"4. Daily view of the last {RECENT_WEEKS} weeks (all oblasts pooled)")
    day = r.groupby(r["t"].dt.strftime("%Y-%m-%d")).agg(
        oblast_hours=("t", "size"),
        active_share=("active_at_t", "mean"),
        quarantined=("quarantined", "sum"),
        eligible=("y_1h", "count"),
        pos_1h=("y_1h", "sum"),
    )
    day["rate_1h"] = day["pos_1h"] / day["eligible"]
    print(day.round(3))

    # ------------------------------------------------------------------ 5
    section("5. Output")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(grid):,} rows to {OUT_PATH}")
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
