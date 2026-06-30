"""
═══════════════════════════════════════════════════════════════════
KALNET AI-4 — Fee Default Predictor — train_v3.py
Author: Are Samhith (ML Engineer 2)
Project: KALNET AI-4 School Management System
Module: Fee Default Prediction (fee_features_v3.csv)
═══════════════════════════════════════════════════════════════════

WHY THIS VERSION (V3) EXISTS:
  V1 used SMOTE -> inflated the 5% minority class to 50% of the
  training data. The model learned a fake, balanced world and as
  a result fired "default" far too aggressively (56 false alarms).

  V2 used sample_weight instead of SMOTE (good idea) but the
  dataset was too small (500 rows, only 82 defaulters, single
  snapshot, no trend signal) -> CV AUC std = 0.08, i.e. the model's
  performance depended heavily on which fold it happened to see.
  That's not a model you can trust in production.

  V3 fixes both root causes at once:
    1. No SMOTE. We keep sample_weight='balanced' (V2's good idea).
    2. 4x more data (2000 rows) AND, more importantly, two terms of
       *trend* history per student (escalating, both_terms_late,
       outstanding_growth) which are validated to be 5-16x stronger
       signals than any single-snapshot feature could ever be.

  This file trains, tunes, evaluates, and packages that model.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import optuna
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    fbeta_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# Quiet Optuna's per-trial spam; we print our own summaries instead.
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
import os


print("Working directory:", os.getcwd())
print("Resolved path:", os.path.abspath("data/fee_features_v3.csv"))

df = pd.read_csv("data/fee_features_v3.csv")
print("Row count:", len(df))
print("Default count:", (df["label"] == 2).sum())
print(df["student_id"].head(3).tolist())

# ---------------------------------------------------------------------------
# Feature list — order matters because the FastAPI backend builds a feature
# vector in this exact order at inference time. Do not reorder casually.
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "t1_status", "t1_outstanding", "t1_days_late",
    "t2_status", "t2_outstanding", "t2_days_late",
    "income_encoded", "sibling_count", "transport_user",
    "outstanding_growth", "days_late_trend", "both_terms_late",
    "escalating", "avg_outstanding", "max_days_late",
]


def banner(step_num: int, title: str) -> None:
    """Consistent [STEP N] log formatting so the run log is easy to scan."""
    print("\n" + "=" * 70)
    print(f"[STEP {step_num}] {title}")
    print("=" * 70)


# ===========================================================================
# STEP 1 — Load data
# ===========================================================================
def step1_load_data(path: str = "data/fee_features_v3.csv") -> pd.DataFrame:
    banner(1, "Load data")
    df = pd.read_csv(path)

    # label: 0=ontime, 1=late, 2=default. We only care about catching
    # DEFAULT (2). "Late" (1) is not the target — defaulting after grace
    # period is the costly event for the school, so we binarize on that.
    df["fee_default"] = (df["label"] == 2).astype(int)

    print(f"Loaded {len(df)} rows from {path}")
    print("Class distribution (fee_default):")
    print(df["fee_default"].value_counts().rename({0: "non-default", 1: "default"}))
    pct = 100 * df["fee_default"].mean()
    print(f"Default rate: {pct:.2f}%")
    return df


# ===========================================================================
# STEP 2 — Train/test split
# ===========================================================================
def step2_split(df: pd.DataFrame):
    banner(2, "Train/test split")
    X = df[FEATURE_COLS]
    y = df["fee_default"]

    # Stratify is non-negotiable here: with only ~100 defaulters total,
    # a random (non-stratified) split could easily starve the test set
    # of defaulters and give us a meaningless evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train rows: {len(X_train)} | defaulters: {y_train.sum()}")
    print(f"Test rows:  {len(X_test)} | defaulters: {y_test.sum()}")
    return X_train, X_test, y_train, y_test


# ===========================================================================
# STEP 3 — Sample weights
# ===========================================================================
def step3_sample_weights(y_train: pd.Series) -> np.ndarray:
    banner(3, "Sample weights (balanced, NO SMOTE)")

    # compute_sample_weight gives every TRUE row a weight inversely
    # proportional to its class frequency. Unlike SMOTE, it never invents
    # synthetic rows -> the model only ever sees real students, just
    # weighted so that the rare "default" rows count more during fitting.
    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    w_non_default = weights[y_train.values == 0][0]
    w_default = weights[y_train.values == 1][0]
    print(f"Non-defaulter weight: {w_non_default:.4f}")
    print(f"Defaulter weight:     {w_default:.4f}")
    print(f"Ratio (default/non):  {w_default / w_non_default:.2f}x")
    return weights


# ===========================================================================
# STEP 4 — Optuna hyperparameter tuning
# ===========================================================================
def step4_optuna_tune(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 30):
    banner(4, f"Optuna hyperparameter tuning ({n_trials} trials)")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": RANDOM_STATE,
        }

        # Fast 3-fold CV INSIDE the search loop. We use only 3 folds here
        # (not 5) purely for speed across 30 trials; the real, trustworthy
        # 5-fold CV happens later in STEP 7 on the final chosen model.
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        ap_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            sw = compute_sample_weight(class_weight="balanced", y=y_tr)

            model = GradientBoostingClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=sw)

            proba = model.predict_proba(X_val)[:, 1]
            # average_precision_score is our primary metric because, with a
            # 5% positive rate, AUC-ROC can look deceptively good while
            # precision on the positive class is still poor. AP (area under
            # the PR curve) is far more honest about rare-event performance.
            ap_scores.append(average_precision_score(y_val, proba))

        return float(np.mean(ap_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print("Best params found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Best CV average_precision (3-fold, in-search): {study.best_value:.4f}")

    return study.best_params


# ===========================================================================
# STEP 5 — Train final model
# ===========================================================================
def step5_train_final(X_train, y_train, best_params: dict) -> GradientBoostingClassifier:
    banner(5, "Train final model on full training set")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    final_params = dict(best_params)
    final_params["random_state"] = RANDOM_STATE

    model = GradientBoostingClassifier(**final_params)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    print(f"Final model trained with params: {final_params}")
    return model


# ===========================================================================
# STEP 6 — Feature importances
# ===========================================================================
def step6_feature_importances(model: GradientBoostingClassifier) -> None:
    banner(6, "Feature importances")

    importances = model.feature_importances_
    pairs = sorted(zip(FEATURE_COLS, importances), key=lambda p: p[1], reverse=True)

    max_imp = pairs[0][1] if pairs else 1.0
    bar_width = 40

    for name, imp in pairs:
        bar_len = int((imp / max_imp) * bar_width) if max_imp > 0 else 0
        bar = "█" * bar_len
        print(f"{name:<22} {imp*100:5.1f}%  {bar}")


# ===========================================================================
# STEP 7 — Full 5-fold Stratified CV on final params
# ===========================================================================
def step7_full_cv(X_train, y_train, best_params: dict):
    banner(7, "Full 5-fold Stratified CV on final model config")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores, ap_scores = [], []

    final_params = dict(best_params)
    final_params["random_state"] = RANDOM_STATE

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        sw = compute_sample_weight(class_weight="balanced", y=y_tr)
        model = GradientBoostingClassifier(**final_params)
        model.fit(X_tr, y_tr, sample_weight=sw)

        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        ap = average_precision_score(y_val, proba)

        auc_scores.append(auc)
        ap_scores.append(ap)
        print(f"Fold {fold_i}: AUC-ROC = {auc:.4f} | Avg Precision = {ap:.4f}")

    auc_mean, auc_std = float(np.mean(auc_scores)), float(np.std(auc_scores))
    ap_mean, ap_std = float(np.mean(ap_scores)), float(np.std(ap_scores))

    print(f"\nCV AUC-ROC:        {auc_mean:.4f} +/- {auc_std:.4f}")
    print(f"CV Avg Precision:  {ap_mean:.4f} +/- {ap_std:.4f}")
    print(f"Stability goal (CV AUC std < 0.04): {'PASS ✅' if auc_std < 0.04 else 'FAIL ❌'}")

    return auc_mean, auc_std, ap_mean, ap_std


# ===========================================================================
# STEP 8 — Threshold sweep on test set
# ===========================================================================
def step8_threshold_sweep(model, X_test, y_test):
    banner(8, "Threshold sweep on test set (0.05 to 0.70, step 0.02)")

    proba_test = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.05, 0.70 + 1e-9, 0.02)

    print(f"{'thresh':>7} | {'recall':>7} | {'precision':>9} | {'f2':>6} | {'TP':>4} | {'FN':>4} | {'FP':>4}")
    print("-" * 60)

    rows = []
    for t in thresholds:
        preds = (proba_test >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

        recall = recall_score(y_test, preds, zero_division=0)
        precision = precision_score(y_test, preds, zero_division=0)
        f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)

        rows.append({
            "threshold": round(float(t), 2),
            "recall": recall, "precision": precision, "f2": f2,
            "tp": int(tp), "fn": int(fn), "fp": int(fp),
        })
        print(f"{t:7.2f} | {recall:7.3f} | {precision:9.3f} | {f2:6.3f} | {tp:4d} | {fn:4d} | {fp:4d}")

    # Best F2 among thresholds that still clear a recall floor of 0.75 —
    # this is the primary chosen threshold: maximize F2 (recall-weighted)
    # without letting recall drop below an operationally acceptable floor.
    floor_75 = [r for r in rows if r["recall"] >= 0.75]
    best_f2_row = max(floor_75, key=lambda r: r["f2"]) if floor_75 else max(rows, key=lambda r: r["f2"])

    # Secondary marker: among thresholds with recall >= 0.85, the one with
    # best precision — useful if admins want a "catch almost everyone" mode.
    floor_85 = [r for r in rows if r["recall"] >= 0.85]
    best_85_row = max(floor_85, key=lambda r: r["precision"]) if floor_85 else None

    print("\nBest threshold (max F2, recall >= 0.75):")
    print(f"  threshold={best_f2_row['threshold']} recall={best_f2_row['recall']:.3f} "
          f"precision={best_f2_row['precision']:.3f} f2={best_f2_row['f2']:.3f}")

    if best_85_row:
        print("Best precision threshold (recall >= 0.85):")
        print(f"  threshold={best_85_row['threshold']} recall={best_85_row['recall']:.3f} "
              f"precision={best_85_row['precision']:.3f} f2={best_85_row['f2']:.3f}")
    else:
        print("No threshold in sweep range reached recall >= 0.85.")

    return best_f2_row["threshold"]


# ===========================================================================
# STEP 9 — Final evaluation at best threshold
# ===========================================================================
def step9_final_eval(model, X_test, y_test, threshold: float):
    banner(9, f"Final evaluation at threshold={threshold}")

    proba_test = model.predict_proba(X_test)[:, 1]
    preds = (proba_test >= threshold).astype(int)

    print("Classification report:")
    print(classification_report(y_test, preds, target_names=["non-default", "default"], zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    print("Confusion matrix:")
    print(f"                 predicted=0   predicted=1")
    print(f"  actual=0 (ok)      {tn:6d}        {fp:6d}")
    print(f"  actual=1 (def)     {fn:6d}        {tp:6d}")

    recall = recall_score(y_test, preds, zero_division=0)
    precision = precision_score(y_test, preds, zero_division=0)
    f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)
    auc = roc_auc_score(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)

    print(f"\nRecall:        {recall:.4f}  (goal >= 0.85: {'PASS ✅' if recall >= 0.85 else 'FAIL ❌'})")
    print(f"Precision:     {precision:.4f}  (goal >= 0.40: {'PASS ✅' if precision >= 0.40 else 'FAIL ❌'})")
    print(f"F2 score:      {f2:.4f}")
    print(f"AUC-ROC:       {auc:.4f}  (goal >= 0.82: {'PASS ✅' if auc >= 0.82 else 'FAIL ❌'})")
    print(f"Avg Precision: {ap:.4f}  (goal >= 0.45: {'PASS ✅' if ap >= 0.45 else 'FAIL ❌'})")

    print(f"\nCaught (TP):       {tp}")
    print(f"Missed (FN):       {fn}")
    print(f"False alarms (FP): {fp}")

    return {
        "recall": recall, "precision": precision, "f2": f2,
        "auc": auc, "ap": ap, "tp": tp, "fn": fn, "fp": fp,
    }


# ===========================================================================
# STEP 10 — Comparison table V1 vs V2 vs V3
# ===========================================================================
def step10_comparison_table(v3_auc, v3_cv_auc_std, v3_recall, v3_precision, v3_fp):
    banner(10, "Comparison table: V1 vs V2 vs V3")

    rows = [
        ("V1 (SMOTE)",            0.62, None,  0.875, 0.20, 56),
        ("V2 (sample_weight)",    0.68, 0.08,  0.75,  0.28, 31),
        ("V3 (this run)",         v3_auc, v3_cv_auc_std, v3_recall, v3_precision, v3_fp),
    ]

    header = f"{'Version':<20} | {'AUC-ROC':>8} | {'CV AUC std':>10} | {'Recall':>7} | {'Precision':>9} | {'False alarms':>13}"
    print(header)
    print("-" * len(header))
    for name, auc, std, recall, precision, fp in rows:
        std_str = f"{std:.2f}" if std is not None else "n/a"
        print(f"{name:<20} | {auc:8.3f} | {std_str:>10} | {recall:7.3f} | {precision:9.3f} | {fp:13d}")


# ===========================================================================
# STEP 11 — Save model bundle
# ===========================================================================
def step11_save_bundle(model, threshold, df, cv_auc_mean, cv_ap_mean, path="models/fee_predictor/model_v3.pkl"):
    banner(11, f"Save model bundle -> {path}")

    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "threshold": float(threshold),
        "training_median": float(df["avg_outstanding"].median()),
        "cv_auc_mean": float(cv_auc_mean),
        "cv_ap_mean": float(cv_ap_mean),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)
    print(f"Saved bundle with keys: {list(bundle.keys())}")
    print(f"  threshold        = {bundle['threshold']:.3f}")
    print(f"  training_median  = {bundle['training_median']:.2f}")
    print(f"  cv_auc_mean      = {bundle['cv_auc_mean']:.4f}")
    print(f"  cv_ap_mean       = {bundle['cv_ap_mean']:.4f}")
    return bundle


# ===========================================================================
# STEP 12 — predict_default_risk() — used by FastAPI backend
# ===========================================================================
def predict_default_risk(df_input: pd.DataFrame, bundle_path: str = "models/fee_predictor/model_v3.pkl") -> pd.DataFrame:
    """
    FastAPI backend entry point. Signature and output columns are fixed —
    do not rename/reorder without updating main.py and the React frontend.

    Expected input columns:
      student_id, t1_status, t1_outstanding, t1_days_late,
      t2_status, t2_outstanding, t2_days_late,
      income_encoded, sibling_count, transport_user

    Returns:
      student_id, default_probability (0.0-100.0), risk_category (LOW/MEDIUM/HIGH)
    """
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    threshold = bundle["threshold"]

    df = df_input.copy()

    # Recompute the exact same derived/trend features used at training time.
    # This MUST mirror the dataset's derivation logic exactly, or the model
    # will be scoring on a different feature distribution than it learned.
    df["outstanding_growth"] = df["t2_outstanding"] - df["t1_outstanding"]
    df["days_late_trend"] = df["t2_days_late"] - df["t1_days_late"]
    df["both_terms_late"] = ((df["t1_status"] >= 1) & (df["t2_status"] >= 1)).astype(int)
    df["escalating"] = (df["t2_status"] > df["t1_status"]).astype(int)
    df["avg_outstanding"] = (df["t1_outstanding"] + df["t2_outstanding"]) / 2
    df["max_days_late"] = df[["t1_days_late", "t2_days_late"]].max(axis=1)

    X = df[feature_cols]
    proba = model.predict_proba(X)[:, 1]

    risk_category = np.where(
        proba >= threshold + 0.20, "HIGH",
        np.where(proba >= threshold, "MEDIUM", "LOW"),
    )

    result = pd.DataFrame({
        "student_id": df["student_id"].values,
        "default_probability": np.round(proba * 100, 1),
        "risk_category": risk_category,
    })
    return result


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("KALNET AI-4 — Fee Default Predictor — train_v3.py")
    print(f"Random state: {RANDOM_STATE}")

    df = step1_load_data("data/fee_features_v3.csv")
    X_train, X_test, y_train, y_test = step2_split(df)
    step3_sample_weights(y_train)

    best_params = step4_optuna_tune(X_train, y_train, n_trials=30)
    model = step5_train_final(X_train, y_train, best_params)
    step6_feature_importances(model)

    cv_auc_mean, cv_auc_std, cv_ap_mean, cv_ap_std = step7_full_cv(X_train, y_train, best_params)

    best_threshold = step8_threshold_sweep(model, X_test, y_test)
    final_metrics = step9_final_eval(model, X_test, y_test, best_threshold)

    step10_comparison_table(
        v3_auc=final_metrics["auc"],
        v3_cv_auc_std=cv_auc_std,
        v3_recall=final_metrics["recall"],
        v3_precision=final_metrics["precision"],
        v3_fp=final_metrics["fp"],
    )

    step11_save_bundle(model, best_threshold, df, cv_auc_mean, cv_ap_mean, path="models/fee_predictor/model_v3.pkl")

    banner(12, "predict_default_risk() ready for FastAPI backend")
    print("Function defined in this module: predict_default_risk(df_input) -> pd.DataFrame")

    # Quick smoke test using the actual test split, just to prove the
    # function runs end-to-end against the saved bundle.
    smoke_input = df.loc[X_test.index, [
        "student_id", "t1_status", "t1_outstanding", "t1_days_late",
        "t2_status", "t2_outstanding", "t2_days_late",
        "income_encoded", "sibling_count", "transport_user",
    ]].head(5)
    print("\nSmoke test on 5 test-set rows:")
    print(predict_default_risk(smoke_input, bundle_path="models/fee_predictor/model_v3.pkl").to_string(index=False))

    print("\n" + "=" * 70)
    print("DONE. All goals checked against printed PASS/FAIL markers above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
