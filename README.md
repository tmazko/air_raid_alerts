# Forecasting air-raid alarms in Ukrainian oblasts

Time-series mini-project (AI Agentic School test task). For every oblast and every hour, the models
estimate the probability that a **new air-raid alarm starts within the next 1, 3 or 6 hours**.
The emphasis is on an honest evaluation: strictly chronological splits, no information from the future
at forecast time, and every claim of improvement backed by a statistical check.

> This is an exercise in forecasting and evaluation, not an operational warning system. Do not use it
> for safety decisions.

## TL;DR

* A LightGBM model with features describing the recent alarm history of the oblast (`LGBM_state`)
  beats the strongest baseline (per-oblast alarm frequency over the last 28 days, `B3`) on a held-out
  test period at all three horizons: Brier score **-1.5% / -1.8% / -1.3%** (1h / 3h / 6h).
* On the test period the improvement is **statistically distinguishable from noise only at the 1-hour
  horizon**. On the validation period, where model choices were made, the gains were larger
  (2.6% / 5.0% / 5.6%) and significant at every horizon; they shrank on the test period.
* Hour of day alone adds nothing (`LGBM_basic` is indistinguishable from the plain per-oblast rate).
  What helps is *recent activity*: the number of alarm starts in the previous 7 days dominates the model.
* Features of neighbouring oblasts (`LGBM_nb`) help a little more, but by a pre-agreed selection rule
  they were not adopted as the primary model. They are reported as a secondary result.

## Task definition

* **Unit:** (oblast, hour `t`), `t` in UTC. A forecast is made at the whole hour `t`.
* **Target:** `y_Nh = 1` if a new alarm starts in the oblast in the interval `(t, t + N hours]`,
  for N in {1, 3, 6}.
* **Evaluated rows:** only moments with **no alarm active at `t`** (a forecast "when it is quiet").
  Rows inside a data-quality quarantine window are excluded (see Data).
* **Output:** a probability. Metrics: Brier score (primary), log loss, PR-AUC, ROC-AUC.
  Brier and log loss are proper scoring rules and reward calibrated probabilities. PR-AUC of a random
  ranker equals the positive rate, which is 13.5% / 26.4% / 35.5% on the test period for the three
  horizons.

## Data

Source: [ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
(file `official_data_en.csv`, official alarms; columns `oblast, raion, hromada, level, started_at,
finished_at, source`; timestamps in UTC). The raw file is **not** committed. Download it and place it at
`data/official_data_en.csv`.

Preprocessing (`preprocess.py`, `build_dataset.py`):

* Exact duplicate rows are dropped. For each oblast the rows of levels `oblast` and `raion` are merged
  into **union alarms** (periods in which at least one such row is active). Overlapping or touching
  rows are merged; no gap tolerance is used, because bridging a gap would require knowing a future row.
  `hromada`-level rows are not used. Oblasts with fewer than 100 alarm rows are dropped; 24 oblasts remain.
* Hourly grid from 2022-03-15 16:00 UTC to 2026-09-06 22:00 UTC: 942,504 (oblast, hour) rows, of which
  795,817 are eligible (alarm not active at `t`, not quarantined). The last grid point is
  `floor(last known alarm start - 6 h)` so that no label window extends beyond the data
  (right-censoring would otherwise create false zeros). The last alarm start in the data is
  2026-09-07 04:38 UTC.
* **Quarantine window** 2026-08-29 05:00 to 2026-09-02 20:00 UTC (plus a 6 h margin before it): no new
  alarm start is recorded in any oblast for 105 hours, while 8 oblasts show continuous 105-118 h alarms
  that all close on 2026-09-02. Either records are missing or alarm states are stuck, so states and
  labels there are unreliable. This was a manual data-integrity decision based on `quality_scan.py`,
  not tuned on model metrics. It affects only the last test fold (see Results).

## Pipeline and repository layout

| Step | Script | Purpose |
|---|---|---|
| 0 | `eda.py`, `eda_levels.py`, `quality_scan.py` | Exploration and data-quality scan (read-only) |
| 1 | `preprocess.py` | Cleaning, union alarm intervals per oblast |
| 2 | `build_dataset.py` | Hourly grid, labels `y_1h/3h/6h`, quarantine flag |
| 3 | `splits.py` | Walk-forward folds with embargo (imported by other scripts) |
| 4 | `check_splits_leakage.py` | Independent leakage check of labels and splits |
| 5 | `baselines.py` | Baselines B0-B3 |
| 6 | `features.py` | Alarm-history features of the oblast itself |
| 7 | `neighbors.py` | Alarm-state features of neighbouring oblasts |
| 8 | `check_features_leakage.py` | Independent leakage check of all features |
| 9 | `model_lgbm_v2.py` | LightGBM models vs baselines on identical rows |
| 10 | `bootstrap_compare.py` | Paired block bootstrap of Brier differences |

`model_lgbm.py` is the first LightGBM experiment (features `oblast`, `hour`); it is superseded by
`model_lgbm_v2.py`, which contains the same model as `LGBM_basic`.

## Evaluation protocol and leakage safeguards

**Splits** (`splits.py`). The test period is the last 8 weeks of the grid, as 4 consecutive 2-week
walk-forward folds (`test1`-`test4`). The 8 weeks before it form the validation period with the same
layout (`valid1`-`valid4`). Every fold is predicted by a model trained only on rows before the fold,
with an **expanding window** and a **6-hour embargo** (`t < fold_start - 6 h`), because a training label
looks up to 6 hours ahead and would otherwise contain events from inside the evaluation fold.
Validation labels never reach into the test period. Models are refitted for every fold. The main test
result pools `test1`-`test3`; `test4` is reported separately because of the quarantine.

| Risk | Safeguard | How it is verified |
|---|---|---|
| Labels contain evaluation-period events | 6 h embargo between training and each fold | Assertions in every script; `check_splits_leakage.py` recomputes training labels in a "truncated world" containing only alarm starts before the fold start. A negative control without the embargo must (and does) find mismatches |
| Features use the future | Every feature of row `(oblast, t)` uses only alarm starts `<= t`, finish times `<= t`, and the part of a running alarm before `t` | `check_features_leakage.py`: for 48 cutoffs (random ones plus every fold start) the features of rows near the cutoff are recomputed in a world where later alarms are dropped and a running alarm has no known end. Result: **0 mismatches** for all 17 features. Two negative controls (a deliberately leaky "alarm starts in the next 6 h" feature, for the oblast and for its neighbours) must find mismatches, and do |
| Baselines peek at evaluation labels | Each baseline is fitted on training rows only and receives no evaluation labels | By construction (labels are not passed) |
| Rolling means of labels | Not used. Features are built from alarm events, never from `y_*` (a lagged `y_6h` already looks 5 h past `t`) | Code review; the check above |
| Fitting to the evaluation data | Baseline constants (`SMOOTH_K = 20`, 28 days) and all LightGBM hyperparameters are fixed a priori and **not tuned** | Visible in code |
| Selecting models on test data | Feature and model choices were made on the validation folds only, with a decision rule written down before seeing the results (below) | See "How the test period was used" |
| Models scored on different rows | Every model is evaluated on exactly the same rows | `assert_same_rows` in `model_lgbm_v2.py` |

## Models and features

* **B0** global positive rate. **B1** positive rate per oblast. **B2** rate per (oblast, UTC hour),
  smoothed towards B1. **B3** per-oblast rate over the 28 days before the training cutoff,
  smoothed towards B1 (K = 20). All are static within a fold.
* **LGBM_basic**: LightGBM on `oblast` (categorical) and `hour`.
* **LGBM_state** *(primary model)*: adds 9 features of the oblast's own alarm history:
  hours since the last finished alarm and its duration; alarm starts in the last 1/3/6/24/168 hours;
  share of time under alarm in the last 24 and 168 hours.
* **LGBM_nb** *(secondary)*: `LGBM_state` plus 8 features of the neighbouring oblasts: number of
  neighbours with an active alarm at `t`; alarm starts in the last 1/3/6/24/168 hours summed over
  neighbours; hours since the latest neighbour alarm start; mean share of the last 24 hours under alarm.
  The adjacency table (`neighbors.py`) is static geography written by hand from land borders between
  the 24 oblasts in the grid; it contains no alarm information and cannot leak. Three short border
  segments are flagged in the code as the least certain.
* LightGBM: 100 trees, learning rate 0.05, 15 leaves, seed 42, no tuning, no early stopping.
  A separate model is trained per fold and per horizon.

## Results

### Selection on validation (valid1-valid4, n = 22,723 per horizon)

Relative Brier improvement over B3, with 95% intervals from the block bootstrap described below
(3-day blocks):

| Model | 1h | 3h | 6h |
|---|---|---|---|
| LGBM_state vs B3 | 2.6% [1.7, 3.4] | 5.0% [3.0, 6.9] | 5.6% [3.0, 8.4] |
| LGBM_nb vs LGBM_state | 1.6% [0.8, 2.5] | 1.4% [0.3, 2.4] | 0.8% [-0.4, 2.0] |

Decision rule, fixed before the results were seen: a richer model replaces a simpler one only if the
95% interval of the Brier difference excludes zero with 3-day blocks and keeps its sign with 1-day and
7-day blocks, and a single feature set is used for all horizons. `LGBM_state` passed the rule against
B3. `LGBM_nb` helped significantly at 1h and 3h but not at 6h, a mixed outcome, so the simpler
`LGBM_state` was kept as the primary model.

### Test (test1-test3 pooled, n = 16,741 per horizon) - evaluated once, after freezing the configuration

Brier score (lower is better):

| Model | 1h | 3h | 6h |
|---|---|---|---|
| B0 global rate | 0.1181 | 0.1958 | 0.2292 |
| B1 per oblast | 0.1065 | 0.1515 | 0.1465 |
| B2 per oblast and hour | 0.1066 | 0.1521 | 0.1478 |
| B3 per oblast, last 28 days | 0.0852 | 0.1086 | 0.1053 |
| LGBM_basic | 0.1064 | 0.1520 | 0.1477 |
| **LGBM_state (primary)** | **0.0839** | **0.1067** | **0.1039** |
| LGBM_nb (secondary) | 0.0832 | 0.1058 | 0.1030 |

Other metrics for the strongest baseline and the two LightGBM models:

| Horizon | Model | Log loss | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| 1h | B3 | 0.2721 | 0.5157 | 0.8761 |
| 1h | LGBM_state | 0.2688 | 0.5504 | 0.8828 |
| 1h | LGBM_nb | 0.2652 | 0.5472 | 0.8853 |
| 3h | B3 | 0.3447 | 0.7529 | 0.9003 |
| 3h | LGBM_state | 0.3389 | 0.7785 | 0.9063 |
| 3h | LGBM_nb | 0.3357 | 0.7779 | 0.9080 |
| 6h | B3 | 0.3493 | 0.8565 | 0.9156 |
| 6h | LGBM_state | 0.3436 | 0.8745 | 0.9218 |
| 6h | LGBM_nb | 0.3405 | 0.8753 | 0.9244 |

Paired block bootstrap on the test period, relative Brier improvement with 95% interval (3-day blocks):

| Comparison | 1h | 3h | 6h |
|---|---|---|---|
| LGBM_state vs B3 | 1.5% [0.3, 2.9] | 1.8% [-0.3, 3.8] | 1.3% [-1.6, 4.7] |
| LGBM_nb vs B3 | 2.3% [0.9, 3.9] | 2.6% [0.4, 4.8] | 2.1% [-1.1, 5.4] |
| LGBM_nb vs LGBM_state | 0.8% [0.0, 1.6] | 0.9% [-0.3, 2.0] | 0.8% [-0.7, 2.1] |

Reading these honestly:

* The primary model improves on the best baseline at every horizon, but the interval excludes zero only
  at 1h. At 3h it excludes zero only with 7-day blocks; at 6h it never does.
* The test-period gain is smaller than the validation gain. With four validation folds and a single
  test period we cannot say whether this is regime change, noise, or mild over-reliance on validation.
* `LGBM_nb` looks better on test than `LGBM_state`, but this was **not** used to change the model
  choice: the rule was fixed before the test run, and switching after seeing test results would turn the
  test period into another validation set. It is reported here for transparency.
* Hour of day does not help (`LGBM_basic` matches B1; `B2` does not beat B1).
* In `LGBM_nb`, `n_starts_168h` carries 55-74% of the total gain, i.e. the model mostly learns "how
  active has this oblast been this week". The number of neighbours currently under alarm matters most
  for the 1-hour horizon (about 12% of gain). Gain importance describes the model, not causality.

### test4 (reported separately, descriptive only)

`test4` (2026-08-24 to 2026-09-07) contains the quarantine window: 2,808 oblast-hours are excluded,
the positive rate for the 1h label is 18.5% versus about 13% in `test1`-`test3`, and alarm-history
features right after the window are inflated for the oblasts with "stuck" alarms. No intervals are
computed for it.

| Horizon | B3 | LGBM_state | LGBM_nb |
|---|---|---|---|
| 1h | 0.1083 | 0.1094 | 0.1034 |
| 3h | 0.1208 | 0.1183 | 0.1112 |
| 6h | 0.1163 | 0.1105 | 0.1044 |

Note that on `test4` the primary model is slightly *worse* than B3 at 1h.

### How the test period was used

The baselines B0-B3 and the first LightGBM (`LGBM_basic`) were evaluated on the test folds early in the
project, before any feature work; their settings (`SMOOTH_K`, 28 days, LightGBM parameters) were fixed
in advance and never changed. All later decisions (which features to build, whether to keep neighbour
features, which model is primary) used the validation folds only. `LGBM_state` and `LGBM_nb` were run
on the test folds exactly once, and nothing was changed afterwards.

## Limitations

* One test period of 8 weeks (6 weeks pooled for the main result). Results may not carry over to a
  different phase of the conflict; the bootstrap measures sampling noise inside the period, not
  regime change.
* No hyperparameter tuning and no probability calibration were performed. LightGBM trained with log
  loss tends to give reasonable probabilities, but calibration was not checked.
* Only the oblast level of the official data is forecast. The alarm state of raions or hromadas is not
  predicted.
* Rows with an alarm already active at `t` are not evaluated, so the models say nothing about when an
  ongoing alarm will end or whether another one will follow.
* Data quality: the quarantine window removes some rows, and "stuck" alarms before and after it, if
  any exist undetected, would add noise. The adjacency table is hand-written and not verified against
  an official source.
* Hour-of-day features use UTC; local time (UTC+2/+3 with daylight saving) is not modelled.

## Reproducing the results

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# Put official_data_en.csv at data/official_data_en.csv, then, from the repository root:
python preprocess.py reports/preprocess_step1.txt
python build_dataset.py reports/build_dataset.txt
python splits.py reports/splits.txt
python check_splits_leakage.py
python baselines.py reports/baselines.txt
python features.py reports/features_v1.txt
python neighbors.py reports/neighbors_v1.txt
python check_features_leakage.py

# Validation run (model selection):
python model_lgbm_v2.py reports/model_lgbm_v2_nb.txt
python bootstrap_compare.py reports/bootstrap_compare_nb.txt

# Final test run (do this once, after the configuration is frozen):
#   set USE_TEST = True in model_lgbm_v2.py, then
python model_lgbm_v2.py reports/model_lgbm_v2_test.txt
python bootstrap_compare.py reports/bootstrap_compare_test.txt
```

Every script takes an optional report path (UTF-8 file); without it the report is printed to the
terminal. Intermediate data are written to `data/processed/` and reports to `reports/`; both folders
are listed in `.gitignore` and are regenerated by the commands above. The leakage checks stop with an
assertion error if a leak or a blind check is detected.

## Possible next steps

* Tune hyperparameters and training-window length on the validation folds, with early stopping on a
  chronological hold-out placed after an embargo.
* Check calibration (reliability tables) and, if needed, calibrate on a chronological tail of the
  training data.
* Calendar features (weekday, month, local hour) and longer-lag history; a discrete-time hazard
  formulation, and probabilities that are consistent across horizons (`P(1h) <= P(3h) <= P(6h)`).
* A longer test history with more folds, and a per-oblast error analysis.