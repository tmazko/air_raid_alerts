"""Exploratory data analysis for official_data_en.csv.

Run from the repository root:  python eda.py
Read-only: prints a report to stdout and changes nothing.
"""
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/official_data_en.csv")
EXPECTED_COLS = ["oblast", "raion", "hromada", "level", "started_at", "finished_at", "source"]
LONG_HOURS = [6, 24, 72]  # thresholds (hours) for "suspiciously long" alerts
RECENT_WEEKS = 8  # candidate test window length

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    # ------------------------------------------------------------------ 1
    section("1. Overview")
    raw = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
    print(f"Rows: {len(raw):,}")
    print(f"Columns: {list(raw.columns)}")
    missing_cols = set(EXPECTED_COLS) - set(raw.columns)
    extra_cols = set(raw.columns) - set(EXPECTED_COLS)
    print(f"Expected columns missing: {missing_cols or 'none'}")
    print(f"Unexpected extra columns: {extra_cols or 'none'}")
    print("\nFirst 5 rows (raw values):")
    print(raw.head())
    print("\nLast 5 rows (raw values):")
    print(raw.tail())

    # ------------------------------------------------------------------ 2
    section("2. Missing values (empty string / 'nan' / 'null' / 'none' in raw data)")
    empty_like = raw.apply(lambda s: s.str.strip().str.lower().isin(["", "nan", "null", "none"]))
    print(pd.DataFrame({"empty_count": empty_like.sum(), "empty_pct": (empty_like.mean() * 100).round(3)}))

    # ------------------------------------------------------------------ 3
    section("3. Datetime parsing (UTC)")
    df = raw.copy()
    for c in ["started_at", "finished_at"]:
        df[c] = pd.to_datetime(df[c].replace("", pd.NA), utc=True, errors="coerce")
        print(f"{c}: unparsed (NaT) = {df[c].isna().sum():,}")
    print("\nSample raw started_at / finished_at values (first 3 rows):")
    print(raw[["started_at", "finished_at"]].head(3))

    # ------------------------------------------------------------------ 4
    section("4. Date range")
    print(f"started_at : {df['started_at'].min()}  ->  {df['started_at'].max()}")
    print(f"finished_at: {df['finished_at'].min()}  ->  {df['finished_at'].max()}")
    print(f"Now (UTC)  : {pd.Timestamp.now(tz='UTC')}")
    print("File sorted by started_at:", df["started_at"].is_monotonic_increasing)

    # ------------------------------------------------------------------ 5
    section("5. level and source: overall and by year")
    df["year"] = df["started_at"].dt.year
    print("level (overall):")
    print(df["level"].value_counts(dropna=False))
    print("\nsource (overall):")
    print(df["source"].value_counts(dropna=False))
    print("\nlevel x source:")
    print(pd.crosstab(df["level"], df["source"], margins=True))
    print("\nBy year, level:")
    print(pd.crosstab(df["year"], df["level"], margins=True))
    print("\nBy year, source:")
    print(pd.crosstab(df["year"], df["source"], margins=True))

    # ------------------------------------------------------------------ 6
    section("6. Oblasts")
    ob = df[df["level"] == "oblast"].copy()
    print(f"Rows with level == 'oblast': {len(ob):,}")
    all_oblasts = sorted(df["oblast"].dropna().unique())
    ob_oblasts = sorted(ob["oblast"].dropna().unique())
    print(f"Unique oblast values (all levels): {len(all_oblasts)}")
    print(f"Unique oblast values (level == 'oblast'): {len(ob_oblasts)}")
    only_other_levels = sorted(set(all_oblasts) - set(ob_oblasts))
    print(f"Present in other levels but absent at oblast level: {only_other_levels or 'none'}")
    print("\nOblast-level alarm counts, first/last alarm start:")
    per_ob = ob.groupby("oblast").agg(
        n=("started_at", "size"),
        first_start=("started_at", "min"),
        last_start=("started_at", "max"),
    )
    print(per_ob.sort_values("n", ascending=False))

    print("\nEmpty raion/hromada share by level (oblast level is expected to be empty):")
    for lvl, g in df.groupby("level"):
        print(
            f"  {lvl}: raion empty in {(g['raion'].str.strip() == '').mean():.1%}, "
            f"hromada empty in {(g['hromada'].str.strip() == '').mean():.1%}"
        )

    # ------------------------------------------------------------------ 7
    section("7. Duplicates")
    n_full = df.duplicated(keep="first").sum()
    print(f"Fully duplicated rows: {n_full:,}")
    key = ["oblast", "raion", "hromada", "level", "started_at"]
    n_key = df.duplicated(subset=key, keep="first").sum()
    print(f"Duplicates by key {key}: {n_key:,}")
    key_ob = ["oblast", "started_at"]
    n_key_ob = ob.duplicated(subset=key_ob, keep="first").sum()
    print(f"Oblast level, duplicates by (oblast, started_at): {n_key_ob:,}")
    if n_key_ob:
        print("Examples:")
        print(ob[ob.duplicated(subset=key_ob, keep=False)].sort_values(key_ob).head(10))

    # ------------------------------------------------------------------ 8
    section("8. Duration: finished_at vs started_at")
    d = df.dropna(subset=["started_at", "finished_at"]).copy()
    d["dur_h"] = (d["finished_at"] - d["started_at"]).dt.total_seconds() / 3600
    print(f"Rows with both timestamps: {len(d):,}")
    print(f"finished_at < started_at : {(d['dur_h'] < 0).sum():,} ({(d['dur_h'] < 0).mean():.4%})")
    print(f"finished_at == started_at: {(d['dur_h'] == 0).sum():,} ({(d['dur_h'] == 0).mean():.4%})")
    for h in LONG_HOURS:
        print(f"duration > {h:>2} h          : {(d['dur_h'] > h).sum():,} ({(d['dur_h'] > h).mean():.4%})")
    print("\nDuration (hours), describe + quantiles, all levels:")
    print(d["dur_h"].describe())
    print(d["dur_h"].quantile([0.5, 0.9, 0.99, 0.999]))
    print("\nDuration (hours) by level:")
    print(d.groupby("level")["dur_h"].describe())
    print("\nTop 10 longest alarms:")
    print(d.sort_values("dur_h", ascending=False)[EXPECTED_COLS + ["dur_h"]].head(10))
    print("\nRows with finished_at < started_at (up to 10):")
    print(d[d["dur_h"] < 0][EXPECTED_COLS + ["dur_h"]].head(10))
    n_open = df["finished_at"].isna().sum()
    print(f"\nRows without finished_at (NaT): {n_open:,}")
    if n_open:
        print(df[df["finished_at"].isna()][EXPECTED_COLS].tail(10))

    # ------------------------------------------------------------------ 9
    section("9. Oblast level: overlapping intervals within the same oblast")
    o = ob.dropna(subset=["started_at", "finished_at"]).sort_values(["oblast", "started_at"]).copy()
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    o["fin_s"] = (o["finished_at"] - epoch).dt.total_seconds()
    o["start_s"] = (o["started_at"] - epoch).dt.total_seconds()
    # Running max of previous finish times within each oblast (excludes the current row).
    o["prev_max_fin"] = o.groupby("oblast")["fin_s"].transform(lambda s: s.cummax().shift())
    o["overlaps_prev"] = o["start_s"] < o["prev_max_fin"]
    print(
        "Rows starting before an earlier alarm of the same oblast has finished: "
        f"{o['overlaps_prev'].sum():,} ({o['overlaps_prev'].mean():.4%})"
    )
    if o["overlaps_prev"].any():
        print(o[o["overlaps_prev"]][EXPECTED_COLS].head(10))
    o["gap_h"] = (o["start_s"] - o.groupby("oblast")["fin_s"].shift()) / 3600
    print("\nGap between previous finish and next start (hours), oblast level:")
    print(o["gap_h"].describe())
    print(
        f"Gap <= 0: {(o['gap_h'] <= 0).sum():,};  "
        f"0 < gap <= 5 min: {((o['gap_h'] > 0) & (o['gap_h'] <= 5 / 60)).sum():,}"
    )

    # ------------------------------------------------------------------ 10
    section("10. Time granularity and hour of day (Europe/Kyiv)")
    s = ob["started_at"].dropna()
    print(f"started_at with non-zero seconds: {(s.dt.second != 0).mean():.2%}")
    print(f"started_at exactly on minute :00: {(s.dt.minute == 0).mean():.2%}")
    try:
        kyiv = s.dt.tz_convert("Europe/Kyiv")
    except Exception:  # older tz databases only know the legacy name
        kyiv = s.dt.tz_convert("Europe/Kiev")
    print("\nOblast-level alarm starts by hour of day (Kyiv time):")
    print(kyiv.dt.hour.value_counts().sort_index().to_string())
    print("\nBy day of week (0 = Monday, Kyiv time):")
    print(kyiv.dt.dayofweek.value_counts().sort_index().to_string())

    # ------------------------------------------------------------------ 11
    section("11. Non-stationarity: oblast-level alarms per month")
    ob["ym"] = ob["started_at"].dt.strftime("%Y-%m")
    print(ob.groupby("ym").size().to_string())

    # ------------------------------------------------------------------ 12
    section(f"12. Recent period (last {RECENT_WEEKS} weeks before max started_at, oblast level)")
    end = ob["started_at"].max()
    start = end - pd.Timedelta(weeks=RECENT_WEEKS)
    recent = ob[ob["started_at"] > start]
    print(f"Window: ({start}, {end}]")
    print(f"Alarms: {len(recent):,}, oblasts with at least one alarm: {recent['oblast'].nunique()}")
    print(recent.groupby("oblast").size().sort_values(ascending=False).to_string())
    silent = sorted(set(ob_oblasts) - set(recent["oblast"].unique()))
    print(f"\nOblasts with no alarms in this window: {silent or 'none'}")

    print("\nDone.")


if __name__ == "__main__":
    main()