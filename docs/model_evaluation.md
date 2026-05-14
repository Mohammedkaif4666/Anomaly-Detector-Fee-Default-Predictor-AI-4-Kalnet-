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

## 2. Fee Default Prediction — GradientBoostingClassifier

### Algorithm
- `sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, random_state=42)`
- 80/20 stratified train-test split — critical for the imbalanced 5% default class
- Predicts 3 classes: `0 = On Time`, `1 = Late`, `2 = Default`

### Performance Metrics
| Metric | Value | Target |
|--------|-------|--------|
| Recall (Default Class) | **76.83%** | ≥ 70% |

**Recall Target (≥ 70%): ✅ PASS**

### Classification Report
```
              precision    recall  f1-score   support

     On Time       0.88      0.88      0.88       333
        Late       0.84      0.54      0.66        85
     Default       0.56      0.77      0.65        82

    accuracy                           0.81       500
   macro avg       0.76      0.73      0.73       500
weighted avg       0.82      0.81      0.81       500
```

### Feature Importances

| Feature | Importance | Role |
|---------|------------|------|
| `total_outstanding` | 0.3794 | Total unpaid amount — direct financial risk signal |
| `days_since_last_payment` | 0.2672 | Days overdue — strongest predictor of default |
| `sibling_count` | 0.1358 | More siblings → more financial strain |
| `income_encoded` | 0.1356 | Low income → higher default risk (H=0, M=1, L=2) |
| `transport_user` | 0.0632 | Transport costs add financial burden |
| `previous_term_status` | 0.0189 | Past behaviour predicts future behaviour |

### Top 5 High-Risk Fee Predictions

| Student ID | Default Probability | Outstanding | Days Late | Prev Status | Verdict |
|------------|---------------------|-------------|-----------|-------------|----------|
| STU_165 | 99.6% | ₹4,974 | 112 days | Default | ✅ True Default |
| STU_343 | 99.2% | ₹5,078 | 88 days | Default | ✅ True Default |
| STU_124 | 99.1% | ₹5,044 | 92 days | Default | ⚠️ Check Needed |
| STU_397 | 98.7% | ₹5,340 | 92 days | Default | ✅ True Default |
| STU_284 | 98.4% | ₹4,515 | 43 days | Default | ✅ True Default |

---

## Plain-English Model Description (Sales / Demo)

### What Does This AI Do?

Our system is an **Early Warning Radar** for school administrators — built entirely with Scikit-learn, no paid APIs.

**1. The Attendance Watchman (IsolationForest)**

Every student has a *normal* attendance pattern. The AI learns this pattern over 200 school days. When a student like Rahul Sharma suddenly drops from 92% → 34% attendance in 3 weeks, the model detects that this is *statistically impossible* for a normal student and flags it. The admin is alerted the same week — not a month later.

**2. The Financial Forecast (GradientBoosting)**

Fee defaults don't happen overnight. The AI studies payment history, income brackets, family size, and transport costs to predict — with high recall — which students are likely to miss next term's payment. Early outreach prevents one default at a time.

**Why it matters to a school:**

| Impact | Without AI | With KALNET AI-4 |
|--------|------------|------------------|
| Attendance crisis detected | After 1–2 months | Within 1 week |
| Fee default intervention | After default occurs | 4–6 weeks before |
| Admin workload | Manual review of 500 students | AI flags top 50 to review |
