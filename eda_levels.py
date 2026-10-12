"""Follow-up EDA: how alarm levels (oblast / raion / hromada) relate over time.

Run from the repository root:
    python eda_levels.py reports/eda_levels.txt   # write the report as UTF-8 to a file
    python eda_levels.py                          # print to the terminal instead

Read-only with respect to the data. Exact duplicate rows are dropped in memory first.
"""
import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/official_data_en.csv")
RECENT_WEEKS = 8
# Raion-level rows are rare before 2025 (see yearly counts in the first EDA report).
OVERLAP_FROM = pd.Timestamp("2025-01-01", tz="UTC")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load() -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
    for c in ["started_at", "finished_at"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    n_before = len(df)
    df = df.drop_duplicates().sort_values("started_at").reset_index(drop=True)
    return df, n_before


def main() -> None:
    df, n_before = load()

    # ------------------------------------------------------------------ 1
    section("1. After dropping exact duplicate rows")
    print(f"Rows before: {n_before:,}")
    print(f"Rows after : {len(df):,}  (dropped {n_before - len(df):,})")
    print("\nPer level: count, first and last started_at:")
    print(df.groupby("level")["started_at"].agg(["count", "min", "max"]))

    # ------------------------------------------------------------------ 2
    section("2. Alarm starts per month x level (after dedup)")
    df["ym"] = df["started_at"].dt.strftime("%Y-%m")
    print(pd.crosstab(df["ym"], df["level"], margins=True))

    # ------------------------------------------------------------------ 3
    section("3. Oblast level: overlaps within the same oblast (after dedup)")
    ob = df[df["level"] == "oblast"]
    o = ob.sort_values(["oblast", "started_at"]).copy()
    o["start_s"] = (o["started_at"] - EPOCH).dt.total_seconds()
    o["fin_s"] = (o["finished_at"] - EPOCH).dt.total_seconds()
    # Running max of earlier finish times within each oblast (excludes the current row).
    o["prev_max_fin"] = o.groupby("oblast")["fin_s"].transform(lambda s: s.cummax().shift())
    overlap = o["start_s"] < o["prev_max_fin"]
    print(f"Oblast-level rows: {len(o):,}")
    print(f"Rows starting before an earlier alarm of the same oblast has finished: {overlap.sum():,}")
    if overlap.any():
        print(o.loc[overlap, ["oblast", "started_at", "finished_at"]].head(10))

    # ------------------------------------------------------------------ 4
    section(f"4. Coverage in the last {RECENT_WEEKS} weeks: oblast x level (after dedup)")
    end = df["started_at"].max()
    start = end - pd.Timedelta(weeks=RECENT_WEEKS)
    recent = df[df["started_at"] > start]
    print(f"Window: ({start}, {end}]")
    print(pd.crosstab(recent["oblast"], recent["level"], margins=True))

    # ------------------------------------------------------------------ 5
    section("5. Are raion-level alarms nested inside oblast-level alarms?")
    core_end = ob.loc[ob["oblast"] != "Kyiv City", "started_at"].max()
    print(f"Last oblast-level start excluding 'Kyiv City': {core_end}")
    print(f"Overlap window for the check: [{OVERLAP_FROM}, {core_end}]")
    ra = df[(df["level"] == "raion") & (df["started_at"] >= OVERLAP_FROM) & (df["started_at"] <= core_end)]
    r = ra[["oblast", "raion", "started_at"]].sort_values("started_at")
    o2 = (
        ob[["oblast", "started_at", "finished_at"]]
        .rename(columns={"started_at": "ob_start", "finished_at": "ob_finish"})
        .sort_values("ob_start")
    )
    m = pd.merge_asof(r, o2, left_on="started_at", right_on="ob_start", by="oblast", direction="backward")
    inside = m["ob_finish"].notna() & (m["started_at"] <= m["ob_finish"])
    print(f"Raion-level alarms in the window: {len(m):,}")
    if len(m):
        print(f"Raion alarm starts while an oblast-level alarm of the same oblast is active: "
              f"{inside.sum():,} ({inside.mean():.2%})")
        print("\nShare by oblast (top 10 by raion alarm count):")
        by_ob = m.assign(inside=inside).groupby("oblast")["inside"].agg(["size", "mean"])
        print(by_ob.sort_values("size", ascending=False).head(10))

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
