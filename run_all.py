#!/usr/bin/env python
"""Run the entire Telco‑Churn pipeline from the command line."""

import pathlib, joblib, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from src.data import load_raw, clean, save_processed
from src.modeling import build_logreg_pipeline, build_xgboost_pipeline, train_and_save
from src.evaluation import get_metrics, plot_confusion, plot_roc_curve

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROC = ROOT / "data" / "processed" / "telco_processed.csv"
MODEL_DIR = ROOT / "models"

# ---- Load & clean ---------------------------------------------------------
df_raw = load_raw(RAW)
df = clean(df_raw)
save_processed(df, PROC)

# ---- Baseline Logistic Regression -----------------------------------------
pipe_lr = build_logreg_pipeline(df)
X_test_lr, y_test_lr = train_and_save(pipe_lr, df, MODEL_DIR / "baseline.pkl")
pred_lr = pipe_lr.predict(X_test_lr)
prob_lr = pipe_lr.predict_proba(X_test_lr)[:, 1]
print("Baseline LR metrics:", get_metrics(y_test_lr, pred_lr, prob_lr))

# ---- XGBoost (default) --------------------------------------------------
pipe_xgb = build_xgboost_pipeline(df)
X_test_xgb, y_test_xgb = train_and_save(pipe_xgb, df, MODEL_DIR / "xgboost_default.pkl")
pred_xgb = pipe_xgb.predict(X_test_xgb)
prob_xgb = pipe_xgb.predict_proba(X_test_xgb)[:, 1]
print("XGBoost default metrics:", get_metrics(y_test_xgb, pred_xgb, prob_xgb))

# ---- Save a few visualisations -----------------------------------------
plot_confusion(y_test_lr, pred_lr, "Baseline LR").savefig(
    ROOT / "notebooks" / "plots" / "baseline_lr_cm.png"
)
plot_roc_curve(y_test_lr, prob_lr, "Baseline LR ROC").savefig(
    ROOT / "notebooks" / "plots" / "baseline_lr_roc.png"
)

plot_confusion(y_test_xgb, pred_xgb, "XGBoost").savefig(
    ROOT / "notebooks" / "plots" / "xgb_cm.png"
)
plot_roc_curve(y_test_xgb, prob_xgb, "XGBoost ROC").savefig(
    ROOT / "notebooks" / "plots" / "xgb_roc.png"
)
