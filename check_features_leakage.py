"""Independent leakage check for features.py and neighbors.py.

Idea: for a cutoff T, build a "truncated world" that contains only what a forecaster knows at T:
alarms that started after T are dropped, and an alarm still running at T gets no known end.
Features of every row with t <= T must be IDENTICAL in the full and the truncated world.
If a feature changes, it used information from after t. The truncation applies to ALL oblasts, so
neighbour features are tested through the same mechanism.

Negative controls (must find mismatches, otherwise the check is blind and "0" proves nothing):
  * own oblast: a deliberately leaky feature "alarm starts in the next 6 hours"
  * neighbours: the same leaky feature summed over the neighbouring oblasts

Scope: the intervals from preprocess.py are taken as given; this script tests features.py and
neighbors.py only. Checked rows: all oblasts, t in (T - CHECK_HOURS, T], i.e. the rows closest to the
cutoff, where a forward-looking feature would have to differ.

Run from the repository root:
    python check_features_leakage.py
"""
import numpy as np
import pandas as pd

import neighbors as nb
from features import FEATURE_COLS, H, build_features, count_starts, load_intervals, to_seconds
from splits import GRID_PATH, test_folds, validation_folds

N_RANDOM_CUTOFFS = 40
CHECK_HOURS = 24
MIN_HISTORY_DAYS = 30
SEED = 0
UNKNOWN_END = pd.Timedelta(days=3650)  # "still running, end not known yet"


def truncate(ev: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Alarm intervals as known at `cutoff`."""
    known = ev[ev["started_at"] <= cutoff].copy()
    running = known["finished_at"] > cutoff
    known.loc[running, "finished_at"] = cutoff + UNKNOWN_END
    return known


def leaky_next_6h_starts(keys: pd.DataFrame, ev: pd.DataFrame, use_neighbours: bool) -> np.ndarray:
    """Deliberately leaky feature: alarm starts in (t, t + 6 h] in the oblast or its neighbours."""
    starts = {ob: np.sort(to_seconds(g["started_at"])) for ob, g in ev.groupby("oblast")}
    out = np.zeros(len(keys))
    t_all = to_seconds(keys["t"])
    for oblast, pos in keys.groupby("oblast").indices.items():
        sources = nb.ADJACENCY[oblast] if use_neighbours else [oblast]
        for src in sources:
            if src in starts:
                out[pos] += count_starts(starts[src], t_all[pos], t_all[pos] + 6 * H)
    return out


def main() -> None:
    ev = load_intervals()
    grid = pd.read_csv(GRID_PATH, usecols=["oblast", "t"])
    grid["t"] = pd.to_datetime(grid["t"], utc=True)

    hours = pd.date_range(grid["t"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS), grid["t"].max(), freq="h")
    rng = np.random.default_rng(SEED)
    cutoffs = [hours[i] for i in rng.choice(len(hours), N_RANDOM_CUTOFFS, replace=False)]
    cutoffs += [f.start for f in validation_folds() + test_folds()]  # the moments that matter most
    cutoffs = sorted(set(cutoffs))

    groups = {"own": (FEATURE_COLS, build_features), "neighbour": (nb.NEIGHBOR_COLS, nb.build_neighbor_features)}
    checked = 0
    bad = {g: np.zeros(len(cols), dtype=int) for g, (cols, _) in groups.items()}
    bad_control = {"own": 0, "neighbour": 0}
    for T in cutoffs:
        in_window = (grid["t"] > T - pd.Timedelta(hours=CHECK_HOURS)) & (grid["t"] <= T)
        keys = grid.loc[in_window, ["oblast", "t"]].reset_index(drop=True)
        known = truncate(ev, T)
        for g, (cols, build) in groups.items():
            full_f = build(keys, ev)[cols].to_numpy()
            known_f = build(keys, known)[cols].to_numpy()
            same = np.isclose(full_f, known_f, rtol=0, atol=1e-9, equal_nan=True)
            bad[g] += (~same).sum(axis=0)
            bad_control[g] += int((leaky_next_6h_starts(keys, ev, g == "neighbour")
                                   != leaky_next_6h_starts(keys, known, g == "neighbour")).sum())
        checked += len(keys)

    print(f"Cutoffs: {len(cutoffs)}; rows compared per feature: {checked:,}")
    for g, (cols, _) in groups.items():
        print(f"\n[{g} features]")
        print(pd.DataFrame({"feature": cols, "mismatches": bad[g]}).to_string(index=False))
    print("\nNegative controls (leaky 'starts in next 6 h'):")
    for g in groups:
        print(f"  {g}: {bad_control[g]:,} mismatches")
    for g in groups:
        assert (bad[g] == 0).all(), f"LEAK: a {g} feature at t depends on information after t"
        assert bad_control[g] > 0, f"Check is blind for {g} features: the negative control found nothing"
    print("\nOK: no feature depends on the future; the check detects deliberately leaky features.")


if __name__ == "__main__":
    main()