# train.py
import os
import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ============================================================
# Configuration
# ============================================================
BASE_DIR   = os.getcwd()
DATA_PATH  = os.path.join(BASE_DIR, "fee_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
THRESHOLD_PATH  = os.path.join(BASE_DIR, "threshold.pkl")
FEAT_COLS_PATH  = os.path.join(BASE_DIR, "feature_cols.pkl")

# high_outstanding removed — feature importance was 0.0000
# It was contributing nothing to predictions
# All other 10 features kept
FEATURE_COLS = [
    "days_since_last_payment",
    "previous_term_status",
    "total_outstanding",
    "income_encoded",
    "transport_user",
    "sibling_count",
    "is_low_income",
    "has_many_siblings",
    "was_late_or_worse",
    "days_x_outstanding",
]

DECISION_THRESHOLD = 0.20
RANDOM_STATE       = 42


# ============================================================
# Data Loading
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[DATA] {len(df)} rows loaded")
    print(f"[DATA] Columns: {df.columns.tolist()}")
    return df


# ============================================================
# Feature Engineering
#
# high_outstanding is removed because:
#   - It had 0.0000 feature importance in the trained model
#   - The model was not using it at all
#   - It also required saving a training_median.pkl which
#     added unnecessary complexity
#   - total_outstanding (raw value) is already in FEATURE_COLS
#     and has 24.8% importance — the raw number is enough
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary target: 1 if defaulted, 0 otherwise
    df["fee_default"] = (df["label"] == 2).astype(int)

    # Flag: Low income family — more financially vulnerable
    df["is_low_income"] = (df["income_encoded"] == 2).astype(int)

    # Flag: 3+ siblings — family financially stretched
    df["has_many_siblings"] = (df["sibling_count"] >= 3).astype(int)

    # Flag: Was late or defaulted last term — strong early warning
    df["was_late_or_worse"] = (df["previous_term_status"] >= 1).astype(int)

    # Interaction: how overdue × how much owed
    # 90 days overdue + ₹5000 outstanding is very different
    # from 90 days overdue + ₹200 outstanding
    df["days_x_outstanding"] = (
        df["days_since_last_payment"] * df["total_outstanding"] / 10_000
    )

    return df


# ============================================================
# Model Training
# ============================================================
def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    print(f"\n[SMOTE] Balancing training set...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"[SMOTE] Before: {len(X_train)} rows ({y_train.sum()} defaults)")
    print(f"[SMOTE] After : {len(X_train_sm)} rows ({y_train_sm.sum()} defaults)")

    print(f"\n[TRAINING] Fitting GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.7,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_sm, y_train_sm)
    print(f"[TRAINING] Done")

    return model


# ============================================================
# Feature Importances
# ============================================================
def print_feature_importances(model, feature_cols):
    print("\n[FEATURE IMPORTANCES]")
    print("-" * 52)
    importances = dict(zip(feature_cols, model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 60)
        print(f"  {feat:<26}  {imp:.4f}  {bar}")
    top = max(importances, key=importances.get)
    print(f"\n  → TOP PREDICTOR: '{top}'")


# ============================================================
# Evaluation
# ============================================================
def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    auc       = roc_auc_score(y_test, y_prob)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n[EVALUATION]")
    print("=" * 50)
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Recall    : {recall * 100:.1f}%  {'✓ Target met' if recall >= 0.70 else '✗ Below target'}")
    print(f"  Precision : {precision * 100:.1f}%")
    print(f"  Threshold : {DECISION_THRESHOLD}")

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    print("[CONFUSION MATRIX]")
    print(f"                       Predicted No-Default   Predicted Default")
    print(f"  Actual No-Default         {tn:<5}                  {fp}")
    print(f"  Actual Default            {fn:<5}                  {tp}")
    print(f"\n  ✅ Caught       : {tp} actual defaulters")
    print(f"  ❌ Missed       : {fn} defaulters")
    print(f"  ⚠️  False alarms: {fp} non-defaulters flagged")

    return {
        "auc":       auc,
        "recall":    recall,
        "precision": precision,
    }


# ============================================================
# Prediction Utility
#
# FIX: Previously used df_input["total_outstanding"].median()
# which caused data leakage — the median changed depending
# on which students were passed in, giving different feature
# values for the same student in different batch sizes.
#
# Now fixed: high_outstanding is removed entirely since it
# had 0.0000 importance. No median needed anywhere.
# Feature engineering is consistent regardless of batch size.
# ============================================================
def predict_default_risk(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Predict default risk for a batch of students.

    Input columns required:
        student_id, days_since_last_payment, previous_term_status,
        total_outstanding, income_encoded, transport_user, sibling_count

    Returns DataFrame with:
        student_id | default_probability | risk_category
    """
    # Load saved artifacts
    model      = joblib.load(MODEL_PATH)
    feat_cols  = joblib.load(FEAT_COLS_PATH)
    threshold  = joblib.load(THRESHOLD_PATH)

    # Apply feature engineering
    # No median needed — high_outstanding removed
    df_in = df_input.copy()
    df_in["is_low_income"]      = (df_in["income_encoded"] == 2).astype(int)
    df_in["has_many_siblings"]  = (df_in["sibling_count"] >= 3).astype(int)
    df_in["was_late_or_worse"]  = (df_in["previous_term_status"] >= 1).astype(int)
    df_in["days_x_outstanding"] = (
        df_in["days_since_last_payment"] * df_in["total_outstanding"] / 10_000
    )

    # Predict using saved feature columns — consistent with training
    probs = model.predict_proba(df_in[feat_cols])[:, 1]

    def risk_category(p):
        if p >= 0.60:   return "HIGH"
        elif p >= 0.30: return "MEDIUM"
        return "LOW"

    return pd.DataFrame({
        "student_id"          : df_input["student_id"].values,
        "default_probability" : (probs * 100).round(1),
        "risk_category"       : [risk_category(p) for p in probs],
        "will_default"        : (probs >= threshold).astype(int),
    })


# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("=" * 60)
    print("KALNET — Fee Default Prediction System")
    print("Are Samhith | ML Engineer 2")
    print("=" * 60)

    # Step 1: Load
    df = load_data(DATA_PATH)

    # Step 2: Feature engineering
    df = engineer_features(df)

    total_defaulters = df["fee_default"].sum()
    default_pct      = df["fee_default"].mean() * 100
    print(f"\n[TARGET] Total students   : {len(df)}")
    print(f"[TARGET] Total defaulters : {total_defaulters}")
    print(f"[TARGET] Default %        : {default_pct:.1f}%")

    # Step 3: Split
    X = df[FEATURE_COLS]
    y = df["fee_default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\n[SPLIT] Train : {len(X_train)} rows ({y_train.sum()} defaults)")
    print(f"[SPLIT] Test  : {len(X_test)}  rows ({y_test.sum()} defaults)")

    # Step 4: Train
    model = train_model(X_train, y_train)

    # Step 5: Feature importances
    print_feature_importances(model, FEATURE_COLS)

    # Step 6: Evaluate
    evaluate_model(model, X_test, y_test)

    # Step 7: Save all artifacts
    joblib.dump(model,             MODEL_PATH)
    joblib.dump(DECISION_THRESHOLD, THRESHOLD_PATH)
    joblib.dump(FEATURE_COLS,      FEAT_COLS_PATH)

    print(f"\n[SAVE] model.pkl        → {MODEL_PATH}")
    print(f"[SAVE] threshold.pkl    → {THRESHOLD_PATH}")
    print(f"[SAVE] feature_cols.pkl → {FEAT_COLS_PATH}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
