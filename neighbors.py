"""Feature engineering, step 2: alarm state of the NEIGHBOURING oblasts at time t.

Input : data/processed/oblast_alarm_intervals.csv, data/processed/hourly_grid.csv
Output: data/processed/neighbor_features_v1.csv  (oblast, t + NEIGHBOR_COLS; same rows/order as the grid)

Features of row (oblast, t), computed from the alarm history of the oblast's neighbours:
  nb_n_active                 number of neighbours with an alarm active at t
  nb_starts_{1,3,6,24,168}h   alarm starts in (t - w, t], summed over neighbours
  nb_hours_since_last_start   hours since the most recent alarm start (<= t) in ANY neighbour
  nb_alarm_share_24h          mean over neighbours of the share of (t - 24 h, t] spent under alarm

Adjacency (EDGES) is static geography, hand-written from land borders between oblasts, so it carries
no information about alarms and cannot leak. It is NOT derived from the data. Only the oblasts that
exist in hourly_grid.csv are listed (Luhanska oblast and Crimea are not in the grid, so oblasts that
border them simply have fewer neighbours here). Kyiv City is an enclave: its only neighbour is
Kyivska oblast.

Leakage rules (same contract as features.py):
  * Only starts <= t are counted.
  * "Active at t" means the latest alarm started at or before t has not finished yet. That is known at
    t; the finish time is used only to compare it with t, never as a value.
  * Finished alarms enter alarm_share only up to t.
  * Labels are never read.

Run from the repository root:
    python neighbors.py reports/neighbors_v1.txt   # report as UTF-8 file
    python neighbors.py                            # report to the terminal
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import COUNT_WINDOWS_H, H, alarm_seconds_until, count_starts, load_intervals, to_seconds
from splits import GRID_PATH

OUT_PATH = Path("data/processed/neighbor_features_v1.csv")
SHARE_WINDOW_H = 24
NEIGHBOR_COLS = (
    ["nb_n_active"]
    + [f"nb_starts_{w}h" for w in COUNT_WINDOWS_H]
    + ["nb_hours_since_last_start", f"nb_alarm_share_{SHARE_WINDOW_H}h"]
)

# Land borders between the oblasts present in the grid (each pair listed once).
EDGES = [
    ("Volynska oblast", "Rivnenska oblast"), ("Volynska oblast", "Lvivska oblast"),
    ("Rivnenska oblast", "Lvivska oblast"), ("Rivnenska oblast", "Ternopilska oblast"),
    ("Rivnenska oblast", "Khmelnytska oblast"), ("Rivnenska oblast", "Zhytomyrska oblast"),
    ("Lvivska oblast", "Ternopilska oblast"), ("Lvivska oblast", "Ivano-Frankivska oblast"),
    ("Lvivska oblast", "Zakarpatska oblast"),
    ("Zakarpatska oblast", "Ivano-Frankivska oblast"),
    ("Ivano-Frankivska oblast", "Chernivetska oblast"), ("Ivano-Frankivska oblast", "Ternopilska oblast"),
    ("Chernivetska oblast", "Ternopilska oblast"), ("Chernivetska oblast", "Khmelnytska oblast"),
    ("Ternopilska oblast", "Khmelnytska oblast"),
    ("Khmelnytska oblast", "Vinnytska oblast"), ("Khmelnytska oblast", "Zhytomyrska oblast"),
    ("Vinnytska oblast", "Zhytomyrska oblast"), ("Vinnytska oblast", "Kyivska oblast"),
    ("Vinnytska oblast", "Cherkaska oblast"), ("Vinnytska oblast", "Kirovohradska oblast"),
    ("Vinnytska oblast", "Odeska oblast"),
    ("Zhytomyrska oblast", "Kyivska oblast"),
    ("Kyivska oblast", "Cherkaska oblast"), ("Kyivska oblast", "Poltavska oblast"),
    ("Kyivska oblast", "Chernihivska oblast"), ("Kyivska oblast", "Kyiv City"),
    ("Chernihivska oblast", "Poltavska oblast"), ("Chernihivska oblast", "Sumska oblast"),
    ("Sumska oblast", "Poltavska oblast"), ("Sumska oblast", "Kharkivska oblast"),
    ("Poltavska oblast", "Kharkivska oblast"), ("Poltavska oblast", "Dnipropetrovska oblast"),
    ("Poltavska oblast", "Kirovohradska oblast"), ("Poltavska oblast", "Cherkaska oblast"),
    ("Kharkivska oblast", "Dnipropetrovska oblast"), ("Kharkivska oblast", "Donetska oblast"),
    ("Cherkaska oblast", "Kirovohradska oblast"),
    ("Kirovohradska oblast", "Dnipropetrovska oblast"), ("Kirovohradska oblast", "Mykolaivska oblast"),
    ("Kirovohradska oblast", "Odeska oblast"),
    ("Dnipropetrovska oblast", "Donetska oblast"), ("Dnipropetrovska oblast", "Zaporizka oblast"),
    ("Dnipropetrovska oblast", "Mykolaivska oblast"), ("Dnipropetrovska oblast", "Khersonska oblast"),
    ("Donetska oblast", "Zaporizka oblast"),
    ("Zaporizka oblast", "Khersonska oblast"),
    ("Khersonska oblast", "Mykolaivska oblast"),
    ("Mykolaivska oblast", "Odeska oblast"),
]
# Short border segments the author is least sure about; double-check them on a map.
UNSURE_EDGES = [
    ("Dnipropetrovska oblast", "Khersonska oblast"),
    ("Dnipropetrovska oblast", "Mykolaivska oblast"),
    ("Kirovohradska oblast", "Odeska oblast"),
]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_adjacency(edges) -> dict[str, list[str]]:
    assert len({frozenset(e) for e in edges}) == len(edges), "duplicate edge"
    adj: dict[str, set[str]] = {}
    for a, b in edges:
        assert a != b, "self loop"
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return {k: sorted(v) for k, v in adj.items()}


ADJACENCY = build_adjacency(EDGES)


def oblast_neighbor_features(t_s: np.ndarray, neighbours: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    """Features for the times t_s of one oblast; `neighbours` = (starts, ends) arrays per neighbour."""
    n = len(t_s)
    n_active = np.zeros(n)
    counts = {w: np.zeros(n) for w in COUNT_WINDOWS_H}
    min_since = np.full(n, np.inf)
    share = np.zeros(n)
    for starts, ends in neighbours:
        if len(starts) == 0:  # neighbour with no known alarm: contributes zeros
            continue
        assert (starts[1:] >= ends[:-1]).all(), "intervals must be sorted and disjoint"
        k = np.searchsorted(starts, t_s, side="right")  # alarms started at or before t
        last = np.maximum(k - 1, 0)
        n_active += (k > 0) & (t_s < ends[last])
        for w in COUNT_WINDOWS_H:
            counts[w] += count_starts(starts, t_s - w * H, t_s)
        since = np.where(k > 0, (t_s - starts[last]) / H, np.inf)
        min_since = np.minimum(min_since, since)
        cum = np.cumsum(ends - starts)
        w_s = SHARE_WINDOW_H * H
        share += (alarm_seconds_until(t_s, starts, ends, cum)
                  - alarm_seconds_until(t_s - w_s, starts, ends, cum)) / w_s
    feats = {"nb_n_active": n_active}
    feats.update({f"nb_starts_{w}h": counts[w] for w in COUNT_WINDOWS_H})
    feats["nb_hours_since_last_start"] = np.where(np.isfinite(min_since), min_since, np.nan)
    feats[f"nb_alarm_share_{SHARE_WINDOW_H}h"] = share / len(neighbours)
    return feats


def build_neighbor_features(keys: pd.DataFrame, intervals: pd.DataFrame, adjacency=None) -> pd.DataFrame:
    """Neighbour features for the rows of `keys` (columns oblast, t; tz-aware UTC). Row order is kept.

    `intervals` is everything the forecaster may know; check_features_leakage.py passes a truncated copy.
    """
    adjacency = ADJACENCY if adjacency is None else adjacency
    out = keys[["oblast", "t"]].reset_index(drop=True)
    t_all = to_seconds(out["t"])
    arrays = {}
    for ob, g in intervals.groupby("oblast"):
        g = g.sort_values("started_at")
        arrays[ob] = (to_seconds(g["started_at"]), to_seconds(g["finished_at"]))
    empty = (np.array([]), np.array([]))
    cols = {c: np.full(len(out), np.nan) for c in NEIGHBOR_COLS}
    for oblast, pos in out.groupby("oblast").indices.items():
        if oblast not in adjacency:
            raise KeyError(f"no adjacency entry for {oblast!r}")
        feats = oblast_neighbor_features(t_all[pos], [arrays.get(nb, empty) for nb in adjacency[oblast]])
        for c in NEIGHBOR_COLS:
            cols[c][pos] = feats[c]
    return pd.concat([out, pd.DataFrame(cols)], axis=1)


def main() -> None:
    ev = load_intervals()
    grid = pd.read_csv(GRID_PATH, usecols=["oblast", "t", "active_at_t", "quarantined"])
    grid["t"] = pd.to_datetime(grid["t"], utc=True)

    section("1. Adjacency checks")
    grid_oblasts = set(grid["oblast"].unique())
    assert set(ADJACENCY) == grid_oblasts, (
        f"adjacency/grid mismatch: only in adjacency {sorted(set(ADJACENCY) - grid_oblasts)}, "
        f"only in grid {sorted(grid_oblasts - set(ADJACENCY))}")
    for a, b in EDGES + UNSURE_EDGES:
        assert b in ADJACENCY[a] and a in ADJACENCY[b], f"edge {a} - {b} is not in the adjacency"
    print(f"Oblasts: {len(ADJACENCY)}; undirected edges: {len(EDGES)}; adjacency is symmetric by construction.")
    print(pd.Series({k: len(v) for k, v in ADJACENCY.items()}, name="n_neighbours").to_string())
    print("\nBorders to double-check on a map (short segments):")
    for a, b in UNSURE_EDGES:
        print(f"  {a} - {b}")

    feats = build_neighbor_features(grid[["oblast", "t"]], ev)
    assert len(feats) == len(grid)

    section("2. Overview")
    print(f"Rows: {len(feats):,}; features: {len(NEIGHBOR_COLS)}")
    print("\nMissing values per feature (all rows):")
    print(feats[NEIGHBOR_COLS].isna().sum().to_string())
    print("\nOnly nb_hours_since_last_start may be NA: no neighbour has started an alarm yet.")

    section("3. Distribution on eligible rows (no alarm at t, not quarantined)")
    elig = ~grid["active_at_t"].to_numpy() & ~grid["quarantined"].to_numpy()
    print(f"Eligible rows: {int(elig.sum()):,}")
    print(feats.loc[elig, NEIGHBOR_COLS].describe().T.round(3).to_string())

    section("4. Output")
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