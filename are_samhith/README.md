# KALNET — Fee Default Prediction System

Machine Learning system to predict student fee default risk using Gradient Boosting and SMOTE balancing techniques.

---

# Overview

This project predicts which students are likely to default on fee payments based on financial and behavioral patterns.

The system helps institutions:
- Identify high-risk students early
- Reduce revenue loss
- Improve fee collection efficiency
- Prioritize follow-up actions

---

# Features Used

The model uses the following features:

- days_since_last_payment
- previous_term_status
- total_outstanding
- income_encoded
- transport_user
- sibling_count
- is_low_income
- has_many_siblings
- was_late_or_worse
- days_x_outstanding

---

# Machine Learning Techniques

- GradientBoostingClassifier
- SMOTE (Synthetic Minority Oversampling Technique)
- Feature Engineering
- Stratified Train/Test Split
- Threshold-based Classification

---

# Evaluation Metrics

The project evaluates:

- AUC-ROC
- Recall
- Precision
- Confusion Matrix

Primary goal:
- Maximize recall for defaulters
- Reduce missed high-risk students
