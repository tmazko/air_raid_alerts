"""Preprocessing, step 1: clean raw alarms and build per-oblast union alarm intervals.

A union alarm is a maximal period during which at least one alarm row (of the configured
levels) of that oblast is active. Rows are merged only when they overlap or touch; no gap
tolerance is used, because bridging a gap would require knowing a future row at time t.

Run from the repository root:
    python preprocess.py reports/preprocess_step1.txt   # report as UTF-8 file
    python preprocess.py                                # report to the terminal

Output: data/processed/oblast_alarm_intervals.csv  (oblast, started_at, finished_at, n_rows; UTC)
"""
import sys
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/official_data_en.csv")
OUT_PATH = Path("data/processed/oblast_alarm_intervals.csv")
UNION_LEVELS = ("oblast", "raion")  # levels that define "an alarm is active in the oblast"
SENSITIVITY_LEVELS = ("oblast", "raion", "hromada")  # compared in the report only
MIN_ALARMS_PER_OBLAST = 100  # oblasts with fewer rows (in the used levels) are dropped
RECENT_WEEKS = 8
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_raw() -> tuple[pd.DataFrame, int]:
    """Read the CSV, parse UTC timestamps, drop exact duplicate rows."""
    df = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False)
    for c in ["started_at", "finished_at"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if (df["finished_at"] < df["started_at"]).any():
        raise ValueError("Found rows with finished_at < started_at; handle them explicitly.")
    return df, n_raw


def build_union(df: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping or touching intervals within each oblast."""
    parts = []
    for oblast, g in df.groupby("oblast", sort=True):
        g = g.sort_values("started_at")
        start_s = (g["started_at"] - EPOCH).dt.total_seconds()
        fin_s = (g["finished_at"] - EPOCH).dt.total_seconds()
        # Latest finish among all earlier rows of this oblast (excludes the current row).
        prev_max_fin = fin_s.cummax().shift()
        new_event = prev_max_fin.isna() | (start_s > prev_max_fin)
        event_id = new_event.cumsum()
        ev = g.groupby(event_id).agg(
            started_at=("started_at", "min"),
            finished_at=("finished_at", "max"),
            n_rows=("started_at", "size"),
        )
        ev.insert(0, "oblast", oblast)
        parts.append(ev.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def share_in_alarm(ev: pd.DataFrame, w_start: pd.Timestamp, w_end: pd.Timestamp) -> pd.Series:
    """Fraction of the window [w_start, w_end] during which an alarm is active, per oblast."""
    s = ev["started_at"].clip(lower=w_start)
    e = ev["finished_at"].clip(upper=w_end)
    dur = (e - s).dt.total_seconds().clip(lower=0)
    return dur.groupby(ev["oblast"]).sum() / (w_end - w_start).total_seconds()


def main() -> None:
    # ------------------------------------------------------------------ 1
    section("1. Cleaning")
    df, n_raw = load_raw()
    print(f"Raw rows: {n_raw:,}")
    print(f"After dropping exact duplicates: {len(df):,}")
    used = df[df["level"].isin(UNION_LEVELS)]
    print(f"Union levels: {UNION_LEVELS} -> {len(used):,} rows")
    counts = used["oblast"].value_counts()
    dropped = counts[counts < MIN_ALARMS_PER_OBLAST]
    print(f"Oblasts dropped (< {MIN_ALARMS_PER_OBLAST} rows in used levels): "
          f"{ {k: int(v) for k, v in dropped.items()} or 'none'}")
    used = used[~used["oblast"].isin(dropped.index)]
    print(f"Oblasts kept: {used['oblast'].nunique()}")

    # ------------------------------------------------------------------ 2
    section("2. Union alarm intervals")
    ev = build_union(used)
    print(f"Source rows: {len(used):,} -> union events: {len(ev):,}")
    per_ob = ev.groupby("oblast").agg(events=("n_rows", "size"), merged_rows=("n_rows", "sum"))
    per_ob["rows_per_event"] = (per_ob["merged_rows"] / per_ob["events"]).round(2)
    print(per_ob.sort_values("events", ascending=False))

    # ------------------------------------------------------------------ 3
    section("3. Union event duration (hours)")
    dur_h = (ev["finished_at"] - ev["started_at"]).dt.total_seconds() / 3600
    print(dur_h.describe())
    print(dur_h.quantile([0.5, 0.9, 0.99, 0.999]))
    print("\nTop 10 longest union events:")
    print(ev.assign(dur_h=dur_h).sort_values("dur_h", ascending=False).head(10))

    # ------------------------------------------------------------------ 4
    section("4. Union event starts per month")
    print(ev["started_at"].dt.strftime("%Y-%m").value_counts().sort_index().to_string())

    # ------------------------------------------------------------------ 5
    section(f"5. Last {RECENT_WEEKS} weeks: share of time with an active alarm, per oblast")
    w_end = used["started_at"].max()
    w_start = w_end - pd.Timedelta(weeks=RECENT_WEEKS)
    print(f"Window: ({w_start}, {w_end}]")
    base = share_in_alarm(ev, w_start, w_end)
    ev_alt = build_union(df[df["level"].isin(SENSITIVITY_LEVELS) & df["oblast"].isin(ev["oblast"].unique())])
    alt = share_in_alarm(ev_alt, w_start, w_end)
    n_recent = ev[ev["started_at"] > w_start].groupby("oblast").size()
    table = pd.DataFrame({
        "events_in_window": n_recent,
        f"share_alarm_{'+'.join(UNION_LEVELS)}": base.round(3),
        f"share_alarm_{'+'.join(SENSITIVITY_LEVELS)}": alt.round(3),
    }).fillna(0)
    print(table.sort_values("events_in_window", ascending=False))

    # ------------------------------------------------------------------ 6
    section("6. Output")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(ev):,} rows to {OUT_PATH}")
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
