# KALNET AI-4 — Model Evaluation Report
> Authored by: **Vodyati Sai Phanindra**

---

## 1. Attendance Anomaly Detection — IsolationForest

### Algorithm
- `sklearn.ensemble.IsolationForest(contamination=0.1, n_estimators=100, random_state=42)`
- Features normalized with `StandardScaler` before training
- Risk score derived from `decision_function()`, normalized to 0–100 scale

### Performance Metrics
| Metric | Value | Target |
|--------|-------|--------|
| Recall (Anomaly Class) | **62.86%** | ≥ 60% |
| Precision | 88.00% | — |
| F1-Score | 73.33% | — |
| Overall Accuracy | 93.60% | — |

**Recall Target (≥ 60%): ✅ PASS**

### Classification Report
```
              precision    recall  f1-score   support

      Normal       0.94      0.99      0.96       430
     Anomaly       0.88      0.63      0.73        70

    accuracy                           0.94       500
   macro avg       0.91      0.81      0.85       500
weighted avg       0.93      0.94      0.93       500
```

### Confusion Matrix
```
              Predicted Normal  Predicted Anomaly
Actual Normal      424                6
Actual Anomaly     26                 44
```

### Top 5 Flagged Anomalies (Verification)

| Student ID | Attendance Rate | Risk Score | Absence Streak | Absences (Last 30d) | Verdict |
|------------|-----------------|------------|----------------|---------------------|---------|
| STU_378 | 66.0% | 67.4/100 | 19.0 days | 27.0 days | ✅ Correctly Flagged |
| STU_375 | 65.3% | 65.8/100 | 15.0 days | 26.0 days | ✅ Correctly Flagged |
| STU_389 | 72.2% | 63.3/100 | 7.0 days | 14.0 days | ✅ Correctly Flagged |
| STU_441 | 68.1% | 62.2/100 | 15.0 days | 26.0 days | ✅ Correctly Flagged |
| STU_149 | 66.7% | 62.2/100 | 7.0 days | 21.0 days | ✅ Correctly Flagged |

> **Interpretation:** Students with attendance below 75% and a high risk score are flagged. A sudden drop pattern (high absence in last 30 days + long absence streak) is the key signal.

---

## 2. Fee Default Prediction — GradientBoostingClassifier (V3)

### Algorithm
- `sklearn.ensemble.GradientBoostingClassifier`, hyperparameters tuned via Optuna (30 trials)
- Binary target: `0 = no default`, `1 = default` (was 3-class in V1/V2; "late" payments are no longer a separate prediction class)
- Imbalance handled with `compute_sample_weight('balanced')` — no SMOTE
- 80/20 stratified train-test split; decision threshold tuned via F2-score sweep with a recall floor of 0.75, stored in the model bundle (not the default 0.5 cutoff)
- Bundle reports 5-fold CV AUC-ROC = **0.9972**, CV Avg Precision = **0.9466** from training time

### Performance Metrics (this evaluation run)
| Metric | Value | Target |
|--------|-------|--------|
| Recall (Default Class) | **80.00%** | ≥ 70% |
| Precision (Default Class) | 22.47% | — |
| F1-Score (Default Class) | 35.09% | — |
| AUC-ROC | 0.9021 | — |
| Avg Precision (PR-AUC) | 0.3925 | — |
| Decision Threshold (bundle) | 0.550 | — |

**Recall Target (≥ 70%): ✅ PASS**

### Classification Report
```
              precision    recall  f1-score   support

  No Default       0.99      0.85      0.92       475
     Default       0.22      0.80      0.35        25

    accuracy                           0.85       500
   macro avg       0.61      0.83      0.63       500
weighted avg       0.95      0.85      0.89       500
```

### Confusion Matrix
```
              Predicted No-Default  Predicted Default
Actual No-Default  406                   69
Actual Default     5                     20
```

### Feature Importances

| Feature | Importance | Role |
|---------|------------|------|
| `t2_outstanding` | 0.2989 | Term 2 unpaid amount — direct financial risk signal |
| `avg_outstanding` | 0.2453 | Average unpaid amount across both terms |
| `t2_status` | 0.1568 | Term 2 payment status — most recent behaviour |
| `both_terms_late` | 0.1055 | Late in both terms — persistent risk pattern |
| `t2_days_late` | 0.0563 | Term 2 days overdue — strongest single-term predictor |
| `outstanding_growth` | 0.0467 | Change in unpaid amount term-to-term — rising debt signal |
| `max_days_late` | 0.0347 | Worst days-overdue figure across both terms |
| `t1_outstanding` | 0.0213 | Term 1 unpaid amount |
| `days_late_trend` | 0.0192 | Change in days-overdue term-to-term — worsening lateness |
| `t1_days_late` | 0.0053 | Term 1 days overdue |
| `income_encoded` | 0.0048 | Low income → higher default risk (0=High, 1=Medium, 2=Low) |
| `escalating` | 0.0027 | Status got worse term-to-term — early warning of default trajectory |
| `t1_status` | 0.0012 | Term 1 payment status — early-term behaviour baseline |
| `sibling_count` | 0.0011 | More siblings → more financial strain |
| `transport_user` | 0.0002 | Transport costs add financial burden |

### Top 5 High-Risk Fee Predictions

| Student ID | Default Probability | Avg Outstanding | Max Days Late | Escalating | Verdict |
|------------|---------------------|------------------|---------------|------------|----------|
| STU_127 | 100.0% | ₹4,102 | 120 days | Yes | ⚠️ Check Needed |
| STU_194 | 100.0% | ₹8,000 | 120 days | Yes | ✅ True Default |
| STU_388 | 100.0% | ₹4,348 | 120 days | Yes | ⚠️ Check Needed |
| STU_067 | 100.0% | ₹5,042 | 107 days | Yes | ⚠️ Check Needed |
| STU_252 | 100.0% | ₹5,412 | 120 days | Yes | ⚠️ Check Needed |

---

## Plain-English Model Description (Sales / Demo)

### What Does This AI Do?

Our system is an **Early Warning Radar** for school administrators — built entirely with Scikit-learn, no paid APIs.

**1. The Attendance Watchman (IsolationForest)**

Every student has a *normal* attendance pattern. The AI learns this pattern over 200 school days. When a student like Rahul Sharma suddenly drops from 92% → 34% attendance in 3 weeks, the model detects that this is *statistically impossible* for a normal student and flags it. The admin is alerted the same week — not a month later.

**2. The Financial Forecast (GradientBoosting, V3)**

Fee defaults don't happen overnight. The AI studies two terms of payment history — outstanding balance, days overdue, and whether a family's situation is improving or worsening — to predict, with high recall, which students are on track to default next term. A family that goes from "slightly late" to "escalating and growing debt" is flagged well before the grace period runs out, so outreach can happen 4-6 weeks early instead of after the fact.

**Why it matters to a school:**

| Impact | Without AI | With KALNET AI-4 |
|--------|------------|------------------|
| Attendance crisis detected | After 1–2 months | Within 1 week |
| Fee default intervention | After default occurs | 4–6 weeks before |
| Admin workload | Manual review of 500 students | AI flags top 50 to review |
