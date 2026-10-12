"""Chronological split definitions: validation and test walk-forward folds with an embargo.

Design
  * Test = the last 8 weeks of the grid, as 4 consecutive 2-week folds (walk-forward):
    every test fold is predicted by a model trained only on rows before that fold.
  * Validation = the 8 weeks right before the test period, same fold layout. It is used for
    model selection (hyperparameters, training-window length); the test folds never are.
  * Embargo: training rows must satisfy t < fold_start - 6 h. A row's label looks up to 6 h ahead,
    so without the gap a training label could contain events from inside the evaluation fold.
  * Training window is expanding by default (TRAIN_START = None = all history).
  * Only rows with a label are used (no alarm at t, not quarantined); see build_dataset.py.

Other scripts import validation_folds(), test_folds() and fold_masks().
Run directly to print a fold summary (needs data/processed/hourly_grid.csv):
    python splits.py reports/splits.txt   # report as UTF-8 file
    python splits.py                      # report to the terminal
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GRID_PATH = Path("data/processed/hourly_grid.csv")
HORIZONS_H = (1, 3, 6)
EMBARGO_H = max(HORIZONS_H)
TEST_START = pd.Timestamp("2026-07-13 00:00", tz="UTC")  # Monday, 8 weeks before the data end
FOLD_WEEKS = 2
N_TEST_FOLDS = 4
N_VALID_FOLDS = 4
TRAIN_START = None  # None = all history; set a timestamp to try a shorter training window

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp  # training rows: t < train_end  (= start - embargo)
    start: pd.Timestamp  # evaluation rows: start <= t < end
    end: pd.Timestamp


def walk_forward_folds(first_start: pd.Timestamp, n_folds: int, prefix: str) -> list[Fold]:
    step = pd.Timedelta(weeks=FOLD_WEEKS)
    embargo = pd.Timedelta(hours=EMBARGO_H)
    folds = []
    for k in range(n_folds):
        start = first_start + k * step
        folds.append(Fold(f"{prefix}{k + 1}", train_end=start - embargo, start=start, end=start + step))
    return folds


def test_folds() -> list[Fold]:
    return walk_forward_folds(TEST_START, N_TEST_FOLDS, "test")


def validation_folds() -> list[Fold]:
    # Shifted back by the embargo so that validation labels never reach into the test period.
    first = TEST_START - pd.Timedelta(hours=EMBARGO_H) - N_VALID_FOLDS * pd.Timedelta(weeks=FOLD_WEEKS)
    return walk_forward_folds(first, N_VALID_FOLDS, "valid")


def fold_masks(grid: pd.DataFrame, fold: Fold, label_col: str) -> tuple[pd.Series, pd.Series]:
    """Boolean masks (train, eval) over grid rows. Rows without a label are never used."""
    has_label = grid[label_col].notna()
    train = has_label & (grid["t"] < fold.train_end)
    if TRAIN_START is not None:
        train &= grid["t"] >= TRAIN_START
    evaluate = has_label & (grid["t"] >= fold.start) & (grid["t"] < fold.end)
    return train, evaluate


def check_folds() -> None:
    """Structural checks on the fold layout; raises AssertionError on a violation."""
    embargo = pd.Timedelta(hours=EMBARGO_H)
    for folds in (validation_folds(), test_folds()):
        for a, b in zip(folds, folds[1:]):
            assert a.end == b.start, "folds must be contiguous"
        for f in folds:
            assert f.train_end + embargo == f.start, "embargo must equal the label horizon"
    assert validation_folds()[-1].end + embargo <= test_folds()[0].start, "validation overlaps test"


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    check_folds()
    grid = pd.read_csv(GRID_PATH, usecols=["oblast", "t", "active_at_t", "quarantined"]
                       + [f"y_{n}h" for n in HORIZONS_H])
    grid["t"] = pd.to_datetime(grid["t"], utc=True)
    embargo = pd.Timedelta(hours=EMBARGO_H)

    section("1. Layout")
    print(f"Grid: {grid['t'].min()} -> {grid['t'].max()}")
    print(f"Embargo: {EMBARGO_H} h; fold length: {FOLD_WEEKS} weeks; training window: "
          f"{'all history' if TRAIN_START is None else 'from ' + str(TRAIN_START)}")
    rows = []
    for kind, folds in (("validation", validation_folds()), ("test", test_folds())):
        for f in folds:
            rows.append({"set": kind, "fold": f.name, "train_end(excl)": f.train_end,
                         "eval_start": f.start, "eval_end(excl)": f.end})
    print(pd.DataFrame(rows).to_string(index=False))

    for label in [f"y_{n}h" for n in HORIZONS_H]:
        section(f"2. Fold sizes and positive rates, label {label}")
        out = []
        for kind, folds in (("validation", validation_folds()), ("test", test_folds())):
            for f in folds:
                tr, ev = fold_masks(grid, f, label)
                # Leakage guards: no training label may look into the evaluation fold.
                assert grid.loc[tr, "t"].max() + embargo <= f.start
                assert not (tr & ev).any()
                in_window = (grid["t"] >= f.start) & (grid["t"] < f.end)
                out.append({
                    "set": kind, "fold": f.name,
                    "train_rows": int(tr.sum()), "train_pos_rate": round(grid.loc[tr, label].mean(), 4),
                    "eval_rows": int(ev.sum()), "eval_pos": int(grid.loc[ev, label].sum()),
                    "eval_pos_rate": round(grid.loc[ev, label].mean(), 4),
                    "eval_last_t": grid.loc[ev, "t"].max(),
                    "quarantined_in_fold": int((in_window & grid["quarantined"]).sum()),
                })
        print(pd.DataFrame(out).to_string(index=False))

    print("\nAll leakage guards passed.\nDone.")


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
