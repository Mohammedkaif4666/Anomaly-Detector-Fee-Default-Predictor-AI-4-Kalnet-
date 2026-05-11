import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    classification_report, recall_score, precision_score,
    f1_score, confusion_matrix, accuracy_score
)

def run_evaluation():
    """
    Vodyati Sai Phanindra — Model Evaluation Script
    -----------------------------------------------
    Evaluates both trained models against the synthetic labels,
    prints detailed metrics to console, and writes docs/model_evaluation.md
    """
    print("=" * 60)
    print("  KALNET AI-4 — Model Evaluation (Vodyati Sai Phanindra)")
    print("=" * 60)

    if not os.path.exists('models/attendance_anomaly/model.pkl') or \
       not os.path.exists('models/fee_predictor/model.pkl'):
        print("[ERROR] Models not found. Run training scripts first.")
        return

    # ── 1. ATTENDANCE ANOMALY DETECTION ──────────────────────────
    print("\n[ 1 / 2 ]  Evaluating Attendance Anomaly Model (IsolationForest)")

    df_att = pd.read_csv('data/attendance_features.csv')
    model_att = joblib.load('models/attendance_anomaly/model.pkl')
    scaler_att = joblib.load('models/attendance_anomaly/scaler.pkl')

    X_att = df_att.drop(['student_id', 'is_anomaly'], axis=1)
    y_true_att = df_att['is_anomaly']
    X_scaled = scaler_att.transform(X_att)

    raw_pred = model_att.predict(X_scaled)
    y_pred_att = [1 if x == -1 else 0 for x in raw_pred]

    decision_scores = model_att.decision_function(X_scaled)
    risk_scores = np.clip(100 * (1 - (decision_scores + 0.5) / 1.0), 0, 100)

    att_recall    = recall_score(y_true_att, y_pred_att)
    att_precision = precision_score(y_true_att, y_pred_att, zero_division=0)
    att_f1        = f1_score(y_true_att, y_pred_att, zero_division=0)
    att_accuracy  = accuracy_score(y_true_att, y_pred_att)
    att_cm        = confusion_matrix(y_true_att, y_pred_att)
    att_report    = classification_report(y_true_att, y_pred_att,
                        target_names=["Normal", "Anomaly"])

    print(f"  Recall    : {att_recall:.2%}")
    print(f"  Precision : {att_precision:.2%}")
    print(f"  F1-Score  : {att_f1:.2%}")
    print(f"  Accuracy  : {att_accuracy:.2%}")
    print(f"  Target Recall ≥ 60% → {'PASS ✅' if att_recall >= 0.60 else 'FAIL ❌'}")

    df_att['predicted_anomaly'] = y_pred_att
    df_att['risk_score'] = risk_scores
    flagged_att = df_att[df_att['predicted_anomaly'] == 1].sort_values('risk_score', ascending=False).head(5)

    print("\n  Top 5 Flagged Attendance Anomalies:")
    print(f"  {'Student ID':<12} {'Attendance %':<15} {'Risk Score':<12} {'Absence Streak':<16} {'Absences (30d)'}")
    print("  " + "-" * 68)
    for _, row in flagged_att.iterrows():
        print(f"  {row['student_id']:<12} {row['attendance_rate']:.1%}          {row['risk_score']:<12.1f} {row['longest_absence_streak']:<16} {row['absence_in_last_30_days']}")

    # ── 2. FEE DEFAULT PREDICTION ────────────────────────────────
    print("\n[ 2 / 2 ]  Evaluating Fee Default Prediction Model (GradientBoosting)")

    df_fee = pd.read_csv('data/fee_features.csv')
    model_fee = joblib.load('models/fee_predictor/model.pkl')

    X_fee = df_fee.drop(['student_id', 'label'], axis=1)
    y_true_fee = df_fee['label']
    y_pred_fee = model_fee.predict(X_fee)
    probs_fee  = model_fee.predict_proba(X_fee)[:, 2]

    fee_recall_default = recall_score(y_true_fee, y_pred_fee, labels=[2], average='macro')
    fee_report         = classification_report(y_true_fee, y_pred_fee,
                             target_names=["On Time", "Late", "Default"])

    print(f"  Recall (Default class) : {fee_recall_default:.2%}")
    print(f"  Target Recall ≥ 70%    → {'PASS ✅' if fee_recall_default >= 0.70 else 'FAIL ❌'}")

    df_fee['default_prob'] = probs_fee
    high_risk_fee = df_fee.sort_values('default_prob', ascending=False).head(5)

    print("\n  Top 5 High-Risk Fee Predictions (Likely Defaulters):")
    print(f"  {'Student ID':<12} {'Default Prob':<15} {'Outstanding':<14} {'Days Late':<12} {'Prev Status'}")
    print("  " + "-" * 64)
    for _, row in high_risk_fee.iterrows():
        prev = {0: 'On Time', 1: 'Late', 2: 'Default'}.get(int(row['previous_term_status']), '?')
        print(f"  {row['student_id']:<12} {row['default_prob']:.1%}           ₹{int(row['total_outstanding']):<13,} {int(row['days_since_last_payment']):<12} {prev}")

    # Feature importances
    importances = pd.Series(
        model_fee.feature_importances_,
        index=X_fee.columns
    ).sort_values(ascending=False)
    print("\n  Fee Model Feature Importances:")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<28} {imp:.4f}  {bar}")

    # ── Write docs/model_evaluation.md ───────────────────────────
    os.makedirs('docs', exist_ok=True)
    with open('docs/model_evaluation.md', 'w', encoding='utf-8') as f:
        f.write("# KALNET AI-4 — Model Evaluation Report\n")
        f.write("> Authored by: **Vodyati Sai Phanindra**\n\n")
        f.write("---\n\n")

        # Section 1
        f.write("## 1. Attendance Anomaly Detection — IsolationForest\n\n")
        f.write("### Algorithm\n")
        f.write("- `sklearn.ensemble.IsolationForest(contamination=0.1, n_estimators=100, random_state=42)`\n")
        f.write("- Features normalized with `StandardScaler` before training\n")
        f.write("- Risk score derived from `decision_function()`, normalized to 0–100 scale\n\n")
        f.write("### Performance Metrics\n")
        f.write(f"| Metric | Value | Target |\n|--------|-------|--------|\n")
        f.write(f"| Recall (Anomaly Class) | **{att_recall:.2%}** | ≥ 60% |\n")
        f.write(f"| Precision | {att_precision:.2%} | — |\n")
        f.write(f"| F1-Score | {att_f1:.2%} | — |\n")
        f.write(f"| Overall Accuracy | {att_accuracy:.2%} | — |\n\n")
        target_str = "✅ PASS" if att_recall >= 0.60 else "❌ FAIL"
        f.write(f"**Recall Target (≥ 60%): {target_str}**\n\n")
        f.write("### Classification Report\n```\n")
        f.write(att_report)
        f.write("```\n\n")
        f.write("### Confusion Matrix\n```\n")
        f.write(f"              Predicted Normal  Predicted Anomaly\n")
        f.write(f"Actual Normal      {att_cm[0][0]:<18} {att_cm[0][1]}\n")
        f.write(f"Actual Anomaly     {att_cm[1][0]:<18} {att_cm[1][1]}\n")
        f.write("```\n\n")
        f.write("### Top 5 Flagged Anomalies (Verification)\n\n")
        f.write("| Student ID | Attendance Rate | Risk Score | Absence Streak | Absences (Last 30d) | Verdict |\n")
        f.write("|------------|-----------------|------------|----------------|---------------------|---------|\n")
        for _, row in flagged_att.iterrows():
            verdict = "✅ Correctly Flagged" if row['is_anomaly'] == 1 else "⚠️ False Positive"
            f.write(f"| {row['student_id']} | {row['attendance_rate']:.1%} | {row['risk_score']:.1f}/100 | {row['longest_absence_streak']} days | {row['absence_in_last_30_days']} days | {verdict} |\n")
        f.write("\n> **Interpretation:** Students with attendance below 75% and a high risk score are flagged. ")
        f.write("A sudden drop pattern (high absence in last 30 days + long absence streak) is the key signal.\n\n")
        f.write("---\n\n")

        # Section 2
        f.write("## 2. Fee Default Prediction — GradientBoostingClassifier\n\n")
        f.write("### Algorithm\n")
        f.write("- `sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, random_state=42)`\n")
        f.write("- 80/20 stratified train-test split — critical for the imbalanced 5% default class\n")
        f.write("- Predicts 3 classes: `0 = On Time`, `1 = Late`, `2 = Default`\n\n")
        f.write("### Performance Metrics\n")
        f.write(f"| Metric | Value | Target |\n|--------|-------|--------|\n")
        f.write(f"| Recall (Default Class) | **{fee_recall_default:.2%}** | ≥ 70% |\n\n")
        target_fee = "✅ PASS" if fee_recall_default >= 0.70 else "❌ FAIL"
        f.write(f"**Recall Target (≥ 70%): {target_fee}**\n\n")
        f.write("### Classification Report\n```\n")
        f.write(fee_report)
        f.write("```\n\n")
        f.write("### Feature Importances\n\n")
        f.write("| Feature | Importance | Role |\n|---------|------------|------|\n")
        feat_desc = {
            'days_since_last_payment': 'Days overdue — strongest predictor of default',
            'total_outstanding': 'Total unpaid amount — direct financial risk signal',
            'previous_term_status': 'Past behaviour predicts future behaviour',
            'income_encoded': 'Low income → higher default risk (H=0, M=1, L=2)',
            'transport_user': 'Transport costs add financial burden',
            'sibling_count': 'More siblings → more financial strain',
        }
        for feat, imp in importances.items():
            desc = feat_desc.get(feat, '—')
            f.write(f"| `{feat}` | {imp:.4f} | {desc} |\n")
        f.write("\n### Top 5 High-Risk Fee Predictions\n\n")
        f.write("| Student ID | Default Probability | Outstanding | Days Late | Prev Status | Verdict |\n")
        f.write("|------------|---------------------|-------------|-----------|-------------|----------|\n")
        for _, row in high_risk_fee.iterrows():
            prev = {0: 'On Time', 1: 'Late', 2: 'Default'}.get(int(row['previous_term_status']), '?')
            verdict = "✅ True Default" if row['label'] == 2 else "⚠️ Check Needed"
            f.write(f"| {row['student_id']} | {row['default_prob']:.1%} | ₹{int(row['total_outstanding']):,} | {int(row['days_since_last_payment'])} days | {prev} | {verdict} |\n")
        f.write("\n---\n\n")

        # Sales demo
        f.write("## Plain-English Model Description (Sales / Demo)\n\n")
        f.write("### What Does This AI Do?\n\n")
        f.write("Our system is an **Early Warning Radar** for school administrators — built entirely with Scikit-learn, no paid APIs.\n\n")
        f.write("**1. The Attendance Watchman (IsolationForest)**\n\n")
        f.write("Every student has a *normal* attendance pattern. The AI learns this pattern over 200 school days. ")
        f.write("When a student like Rahul Sharma suddenly drops from 92% → 34% attendance in 3 weeks, the model ")
        f.write("detects that this is *statistically impossible* for a normal student and flags it. ")
        f.write("The admin is alerted the same week — not a month later.\n\n")
        f.write("**2. The Financial Forecast (GradientBoosting)**\n\n")
        f.write("Fee defaults don't happen overnight. The AI studies payment history, income brackets, ")
        f.write("family size, and transport costs to predict — with high recall — which students are likely ")
        f.write("to miss next term's payment. Early outreach prevents one default at a time.\n\n")
        f.write("**Why it matters to a school:**\n\n")
        f.write("| Impact | Without AI | With KALNET AI-4 |\n")
        f.write("|--------|------------|------------------|\n")
        f.write("| Attendance crisis detected | After 1–2 months | Within 1 week |\n")
        f.write("| Fee default intervention | After default occurs | 4–6 weeks before |\n")
        f.write("| Admin workload | Manual review of 500 students | AI flags top 50 to review |\n")

    print(f"\n✅ Evaluation report saved → docs/model_evaluation.md")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
