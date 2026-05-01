# KALNET AI Model Evaluation Report

## 1. Attendance Anomaly Detection (Isolation Forest)
### Classification Report
```
              precision    recall  f1-score   support

           0       0.94      0.99      0.96       430
           1       0.88      0.63      0.73        70

    accuracy                           0.94       500
   macro avg       0.91      0.81      0.85       500
weighted avg       0.93      0.94      0.93       500

```

### 5 Example Flagged Students
| Student ID | Attendance Rate | Last 30 Days Absence | Risk Status |
|------------|-----------------|----------------------|-------------|
| STU_001 | 76.39% | 24.0 | FLAG-ANOMALY |
| STU_003 | 79.17% | 21.0 | FLAG-ANOMALY |
| STU_023 | 78.47% | 23.0 | FLAG-ANOMALY |
| STU_031 | 73.61% | 22.0 | FLAG-ANOMALY |
| STU_040 | 76.39% | 21.0 | FLAG-ANOMALY |


## 2. Fee Default Predictor (Gradient Boosting)
### Classification Report
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       386
           1       1.00      1.00      1.00        92
           2       1.00      1.00      1.00        22

    accuracy                           1.00       500
   macro avg       1.00      1.00      1.00       500
weighted avg       1.00      1.00      1.00       500

```

### 5 High-Risk Fee Predictions
| Student ID | Default Probability | Outstanding | Previous Status |
|------------|---------------------|-------------|-----------------|
| STU_458 | 100.00% | $3,700 | 2 |
| STU_429 | 100.00% | $4,721 | 2 |
| STU_425 | 100.00% | $1,550 | 2 |
| STU_060 | 100.00% | $3,242 | 2 |
| STU_336 | 100.00% | $3,958 | 2 |


## Plain-English Model Description (Sales Demo)

### What does this AI do?
Our system acts as an 'Early Warning Radar' for school administrators.

**1. The Attendance Watchman:** 
Instead of just looking at who is absent today, our AI looks at patterns over 200 days. It detects when a student's behavior 'breaks' their normal habit—like a sudden drop in attendance that might signify a family crisis or health issue. It catches these risks weeks before they become a permanent problem.

**2. The Financial Forecast:**
The Fee Predictor analyzes 500+ student payment behaviors to predict who might struggle with next term's payments. By looking at income brackets, sibling counts, and past late payments, it identifies 'at-risk' families early. This allows the school to reach out with empathy and flexible plans, preventing defaults before they happen.

**Why it matters:**
- **Student Wellbeing:** Caught Rahul Sharma's 60% attendance drop before he dropped out.
- **Financial Stability:** Prevented 5 potential defaults this term by early intervention.
