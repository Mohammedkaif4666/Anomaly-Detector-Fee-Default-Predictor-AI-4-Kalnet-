import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    classification_report, recall_score, precision_score,
    f1_score, confusion_matrix, accuracy_score, roc_auc_score,
    average_precision_score
)

def run_evaluation():
    """
    Vodyati Sai Phanindra — Model Evaluation Script
    -----------------------------------------------
    Evaluates both trained models against the synthetic labels,
    prints detailed metrics to console, and writes docs/model_evaluation.md

    NOTE (fee model section updated to V3):
      - Old script assumed a 3-class GradientBoostingClassifier
        (0=On Time, 1=Late, 2=Default) trained on fee_features.csv
        with columns like previous_term_status / total_outstanding /
        days_since_last_payment.
      - V3 replaced this with a BINARY model (0=no default, 1=default)
        trained on fee_features_v3.csv with two-term trend features,
        saved as a bundle dict in models/fee_predictor/model_v3.pkl
        (keys: model, feature_cols, threshold, training_median,
        cv_auc_mean, cv_ap_mean) per train_v3.py / Are Samhith.
      - This script now loads that bundle, recomputes the same
        feature set, and applies the bundle's tuned threshold
        instead of relying on model.predict()'s default 0.5 cutoff.
    """
    print("=" * 60)
    print("  KALNET AI-4 — Model Evaluation (Vodyati Sai Phanindra)")
    print("=" * 60)

    if not os.path.exists('models/attendance_anomaly/model.pkl') or \
       not os.path.exists('models/fee_predictor/model_v3.pkl'):
        print("[ERROR] Models not found. Run training scripts first.")
        print("        Expected: models/attendance_anomaly/model.pkl")
        print("        Expected: models/fee_predictor/model_v3.pkl")
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

    # ── 2. FEE DEFAULT PREDICTION (V3 — binary, threshold-based) ──
    print("\n[ 2 / 2 ]  Evaluating Fee Default Prediction Model (GradientBoosting V3)")

    df_fee = pd.read_csv('data/fee_features_v3.csv')
    bundle_fee = joblib.load('models/fee_predictor/model_v3.pkl')

    model_fee     = bundle_fee['model']
    feature_cols  = bundle_fee['feature_cols']
    threshold     = bundle_fee['threshold']
    cv_auc_mean   = bundle_fee['cv_auc_mean']
    cv_ap_mean    = bundle_fee['cv_ap_mean']

    # Binary target: label==2 (default) is the only positive class we
    # evaluate against now — "late" (label==1) is no longer a separate
    # class in V3, it's just a non-default outcome.
    df_fee['fee_default'] = (df_fee['label'] == 2).astype(int)
    y_true_fee = df_fee['fee_default']

    X_fee = df_fee[feature_cols]
    probs_fee = model_fee.predict_proba(X_fee)[:, 1]
    y_pred_fee = (probs_fee >= threshold).astype(int)

    fee_recall    = recall_score(y_true_fee, y_pred_fee, zero_division=0)
    fee_precision = precision_score(y_true_fee, y_pred_fee, zero_division=0)
    fee_f1        = f1_score(y_true_fee, y_pred_fee, zero_division=0)
    fee_auc       = roc_auc_score(y_true_fee, probs_fee)
    fee_ap        = average_precision_score(y_true_fee, probs_fee)
    fee_cm        = confusion_matrix(y_true_fee, y_pred_fee, labels=[0, 1])
    fee_report    = classification_report(y_true_fee, y_pred_fee,
                         target_names=["No Default", "Default"], zero_division=0)

    print(f"  Threshold (from bundle) : {threshold:.3f}")
    print(f"  Recall (Default class)  : {fee_recall:.2%}")
    print(f"  Precision (Default)     : {fee_precision:.2%}")
    print(f"  AUC-ROC                 : {fee_auc:.4f}")
    print(f"  Avg Precision (PR-AUC)  : {fee_ap:.4f}")
    print(f"  CV AUC (train-time)     : {cv_auc_mean:.4f}")
    print(f"  Target Recall ≥ 70%     → {'PASS ✅' if fee_recall >= 0.70 else 'FAIL ❌'}")

    df_fee['default_prob'] = probs_fee
    high_risk_fee = df_fee.sort_values('default_prob', ascending=False).head(5)

    print("\n  Top 5 High-Risk Fee Predictions (Likely Defaulters):")
    print(f"  {'Student ID':<12} {'Default Prob':<15} {'Avg Outstanding':<16} {'Max Days Late':<14} {'Escalating'}")
    print("  " + "-" * 70)
    for _, row in high_risk_fee.iterrows():
        esc = "Yes" if row['escalating'] == 1 else "No"
        print(f"  {row['student_id']:<12} {row['default_prob']:.1%}           ₹{row['avg_outstanding']:<15,.0f} {int(row['max_days_late']):<14} {esc}")

    # Feature importances
    importances = pd.Series(
        model_fee.feature_importances_,
        index=feature_cols
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

        # Section 2 — V3
        f.write("## 2. Fee Default Prediction — GradientBoostingClassifier (V3)\n\n")
        f.write("### Algorithm\n")
        f.write("- `sklearn.ensemble.GradientBoostingClassifier`, hyperparameters tuned via Optuna (30 trials)\n")
        f.write("- Binary target: `0 = no default`, `1 = default` (was 3-class in V1/V2; "
                 "\"late\" payments are no longer a separate prediction class)\n")
        f.write("- Imbalance handled with `compute_sample_weight('balanced')` — no SMOTE\n")
        f.write("- 80/20 stratified train-test split; decision threshold tuned via F2-score "
                 "sweep with a recall floor of 0.75, stored in the model bundle (not the default 0.5 cutoff)\n")
        f.write(f"- Bundle reports 5-fold CV AUC-ROC = **{cv_auc_mean:.4f}**, CV Avg Precision = **{cv_ap_mean:.4f}** "
                 "from training time\n\n")
        f.write("### Performance Metrics (this evaluation run)\n")
        f.write(f"| Metric | Value | Target |\n|--------|-------|--------|\n")
        f.write(f"| Recall (Default Class) | **{fee_recall:.2%}** | ≥ 70% |\n")
        f.write(f"| Precision (Default Class) | {fee_precision:.2%} | — |\n")
        f.write(f"| F1-Score (Default Class) | {fee_f1:.2%} | — |\n")
        f.write(f"| AUC-ROC | {fee_auc:.4f} | — |\n")
        f.write(f"| Avg Precision (PR-AUC) | {fee_ap:.4f} | — |\n")
        f.write(f"| Decision Threshold (bundle) | {threshold:.3f} | — |\n\n")
        target_fee = "✅ PASS" if fee_recall >= 0.70 else "❌ FAIL"
        f.write(f"**Recall Target (≥ 70%): {target_fee}**\n\n")
        f.write("### Classification Report\n```\n")
        f.write(fee_report)
        f.write("```\n\n")
        f.write("### Confusion Matrix\n```\n")
        f.write(f"              Predicted No-Default  Predicted Default\n")
        f.write(f"Actual No-Default  {fee_cm[0][0]:<21} {fee_cm[0][1]}\n")
        f.write(f"Actual Default     {fee_cm[1][0]:<21} {fee_cm[1][1]}\n")
        f.write("```\n\n")
        f.write("### Feature Importances\n\n")
        f.write("| Feature | Importance | Role |\n|---------|------------|------|\n")
        feat_desc = {
            't1_status': 'Term 1 payment status — early-term behaviour baseline',
            't1_outstanding': 'Term 1 unpaid amount',
            't1_days_late': 'Term 1 days overdue',
            't2_status': 'Term 2 payment status — most recent behaviour',
            't2_outstanding': 'Term 2 unpaid amount — direct financial risk signal',
            't2_days_late': 'Term 2 days overdue — strongest single-term predictor',
            'income_encoded': 'Low income → higher default risk (0=High, 1=Medium, 2=Low)',
            'sibling_count': 'More siblings → more financial strain',
            'transport_user': 'Transport costs add financial burden',
            'outstanding_growth': 'Change in unpaid amount term-to-term — rising debt signal',
            'days_late_trend': 'Change in days-overdue term-to-term — worsening lateness',
            'both_terms_late': 'Late in both terms — persistent risk pattern',
            'escalating': 'Status got worse term-to-term — early warning of default trajectory',
            'avg_outstanding': 'Average unpaid amount across both terms',
            'max_days_late': 'Worst days-overdue figure across both terms',
        }
        for feat, imp in importances.items():
            desc = feat_desc.get(feat, '—')
            f.write(f"| `{feat}` | {imp:.4f} | {desc} |\n")
        f.write("\n### Top 5 High-Risk Fee Predictions\n\n")
        f.write("| Student ID | Default Probability | Avg Outstanding | Max Days Late | Escalating | Verdict |\n")
        f.write("|------------|---------------------|------------------|---------------|------------|----------|\n")
        for _, row in high_risk_fee.iterrows():
            esc = "Yes" if row['escalating'] == 1 else "No"
            verdict = "✅ True Default" if row['label'] == 2 else "⚠️ Check Needed"
            f.write(f"| {row['student_id']} | {row['default_prob']:.1%} | ₹{row['avg_outstanding']:,.0f} | {int(row['max_days_late'])} days | {esc} | {verdict} |\n")
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
        f.write("**2. The Financial Forecast (GradientBoosting, V3)**\n\n")
        f.write("Fee defaults don't happen overnight. The AI studies two terms of payment history — outstanding ")
        f.write("balance, days overdue, and whether a family's situation is improving or worsening — to predict, ")
        f.write("with high recall, which students are on track to default next term. A family that goes from ")
        f.write("\"slightly late\" to \"escalating and growing debt\" is flagged well before the grace period runs out, ")
        f.write("so outreach can happen 4-6 weeks early instead of after the fact.\n\n")
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