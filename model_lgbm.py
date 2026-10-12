"""LightGBM baseline model for air raid prediction.

Evaluates a basic ML model on walk-forward folds using the same strict 
temporal splits as the baselines.

Features (v1):
  * oblast (categorical)
  * hour (numeric)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from splits import GRID_PATH, HORIZONS_H, fold_masks, test_folds, validation_folds

METRICS_PATH = Path("data/processed/lgbm_metrics.csv")
EPS = 1e-6

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_grid() -> pd.DataFrame:
    cols = ["oblast", "t", "active_at_t", "quarantined"] + [f"y_{n}h" for n in HORIZONS_H]
    grid = pd.read_csv(GRID_PATH, usecols=cols)
    grid["t"] = pd.to_datetime(grid["t"], utc=True)
    grid["hour"] = grid["t"].dt.hour
    grid["oblast"] = grid["oblast"].astype("category")  # Required for LightGBM categorical split
    return grid


def get_within_oblast_roc_auc(df: pd.DataFrame) -> float:
    aucs = []
    for _, grp in df.groupby("oblast", observed=False):
        if grp["y"].nunique() > 1:
            aucs.append(roc_auc_score(grp["y"], grp["p"]))
    return np.mean(aucs) if aucs else np.nan


def score(sub: pd.DataFrame) -> dict:
    y = sub["y"].to_numpy()
    p = sub["p"].to_numpy()
    return {
        "n": len(y),
        "pos_rate": y.mean(),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "within_oblast_roc_auc": get_within_oblast_roc_auc(sub),
    }


def train_and_predict(grid: pd.DataFrame, folds) -> pd.DataFrame:
    features = ["oblast", "hour"]
    parts = []
    
    for fold in folds:
        for n in HORIZONS_H:
            label = f"y_{n}h"
            tr_mask, ev_mask = fold_masks(grid, fold, label)
            
            # 1. Prepare Train
            X_train = grid.loc[tr_mask, features]
            y_train = grid.loc[tr_mask, label].astype(int)
            
            # Leakage guard
            assert grid.loc[tr_mask, "t"].max() < fold.train_end <= fold.start - pd.Timedelta(hours=max(HORIZONS_H))
            
            # 2. Prepare Eval
            X_ev = grid.loc[ev_mask, features]
            y_ev = grid.loc[ev_mask, label].astype(int)
            
            # Leakage guard
            assert grid.loc[ev_mask, "t"].min() >= fold.start and grid.loc[ev_mask, "t"].max() < fold.end
            
            # 3. Train LightGBM
            # Basic params to avoid overfitting on just 2 features
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=15,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train)
            
            # 4. Predict
            p = model.predict_proba(X_ev)[:, 1]
            
            parts.append(pd.DataFrame({
                "fold": fold.name,
                "label": label,
                "model": "LGBM_basic",
                "oblast": grid.loc[ev_mask, "oblast"].to_numpy(),
                "y": y_ev.to_numpy(),
                "p": p
            }))
            
    return pd.concat(parts, ignore_index=True)


def summarize(preds: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for gname, fold_names in groups.items():
        sub_g = preds[preds["fold"].isin(fold_names)]
        for label, sub_l in sub_g.groupby("label"):
            rows.append({"group": gname, "label": label, "model": "LGBM_basic", **score(sub_l)})
    return pd.DataFrame(rows)


def main() -> None:
    grid = load_grid()
    vfolds, tfolds = validation_folds(), test_folds()
    
    section("1. Training LightGBM per fold")
    preds = train_and_predict(grid, vfolds + tfolds)
    
    t_names = [f.name for f in tfolds]
    groups = {"POOLED test1-3 (PRIMARY)": t_names[:3]}
    res = summarize(preds, groups)
    
    cols = ["group", "model", "n", "pos_rate", "brier", "log_loss", "pr_auc", "roc_auc", "within_oblast_roc_auc"]
    for n in HORIZONS_H:
        label = f"y_{n}h"
        section(f"2. Results, label {label}")
        r = res[res["label"] == label][cols]
        print(r.round(4).to_string(index=False))

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(METRICS_PATH, index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = open(out_path, "w", encoding="utf-8")
    try:
        main()
    finally:
        if sys.stdout is not sys.__stdout__:
            sys.stdout.close()
            sys.stdout = sys.__stdout__