# KALNET AI-4: Individual Contribution & Technical Report

**Project:** Anomaly Detector + Fee Default Predictor (System 5)  
**Date:** April 2026  

This document provides a detailed breakdown of the technical contributions made by each engineer on the AI-4 Pod. It is designed to be used during management reviews to explain the specific logic, algorithms, and value delivered by each individual.

---

## 1. Mohammed Kaif
**Role:** Pod Lead & Data Engineer  

### Key Responsibilities:
*   **Synthetic Data Architecture:** Designed the logic for creating realistic school datasets without relying on external APIs.
*   **Anomalous Pattern Injection:** Engineered specific "data triggers" (sudden attendance drops) to ensure the AI models had clear patterns to learn from.

### Technical Implementation:
*   **Script:** `mohammed_kaif/generate_raw_data.py`
*   **Attendance Simulation:** Generated a 100,000-record dataset for 500 students. He implemented a two-phase simulation: a "Normal Phase" (80-97% attendance) and an "Anomalous Phase" (sudden drop to 20-40%) for 14% of the student population.
*   **Fee Simulation:** Created a multi-term payment history including features like `family_income_bracket`, `transport_user`, and `sibling_count`. He ensured a 5% default rate to represent realistic financial risk.

---

## 2. Rohith Koppu
**Role:** Data Pipeline Engineer  

### Key Responsibilities:
*   **Feature Engineering:** Transformed raw logs into high-value mathematical features that machine learning models can understand.
*   **Data Quality Assurance:** Built the validation layer to ensure no "garbage" data entered the training pipeline.

### Technical Implementation:
*   **Scripts:** `rohith_koppu/feature_engineering.py`, `rohith_koppu/data_quality_checks.py`
*   **Attendance Features:** Engineered `longest_absence_streak` and `day_of_week_variance`. These features are critical because they detect *how* a student is absent (e.g., missing every Friday vs. missing a solid week).
*   **Fee Features:** Created `total_outstanding` and `previous_term_status` encoders.
*   **Validation:** Implemented automated assertions to check for NaN values and distribution shifts, ensuring the dataset was balanced and clean.

---

## 3. Om Dattatray Wabale
**Role:** ML Engineer 1 (Attendance Anomaly)  

### Key Responsibilities:
*   **Unsupervised Learning Implementation:** Developed the anomaly detection engine using the Isolation Forest algorithm.
*   **Risk Scoring:** Created a normalized 0-100 risk scale to make technical AI outputs understandable for school principals.

### Technical Implementation:
*   **Script:** `om_dattatray_wabale/train.py`
*   **Algorithm:** **Isolation Forest** (contamination=0.1, n_estimators=100).
*   **Scaling:** Utilized `StandardScaler` to ensure features like "Attendance Rate" (0.0 to 1.0) and "Absence Streak" (0 to 200) were weighted equally by the model.
*   **Outcome:** Achieved a **64% Recall**, successfully identifying the majority of at-risk students while maintaining a high precision of 90%.

---

## 4. Are Samhith
**Role:** ML Engineer 2 (Fee Default Predictor)  

### Key Responsibilities:
*   **Supervised Classification:** Built the predictive engine for financial defaults.
*   **Class Imbalance Handling:** Solved the "needle in a haystack" problem where only 5% of students default.

### Technical Implementation:
*   **Script:** `are_samhith/train.py`
*   **Algorithm:** **Gradient Boosting Classifier**.
*   **Optimization:** Implemented `compute_sample_weight` to give more importance to the minority "default" class. He also tuned the decision threshold to **0.3** to prioritize **Recall (80%)** over Precision, ensuring the school misses as few potential defaults as possible.
*   **Insight:** Generated **Feature Importance** reports, identifying that `total_outstanding` is the #1 predictor of future defaults.

---

## 5. Jyothsna Lenka
**Role:** API Developer  

### Key Responsibilities:
*   **Model Deployment:** Wrapped complex Python models into a high-performance REST API.
*   **System Integration:** Ensured the AI backend could talk to the frontend Admin Dashboard.

### Technical Implementation:
*   **Script:** `jyothsna_lenka/app.py`
*   **Framework:** **FastAPI**.
*   **Efficiency:** Designed an **Asynchronous Startup Handler** that loads models into RAM exactly once. This prevents the server from slowing down during requests.
*   **Endpoints:** Developed `POST /ai/anomalies` and `POST /ai/fee-risk`, capable of processing batches of student IDs and returning JSON data including risk levels ("High", "Medium", "Low").

---

## 6. VODYATI SAI PHANINDRA
**Role:** QA + Evaluator  

### Key Responsibilities:
*   **Model Auditing:** Verified that the AI's "flags" were actually intuitive and logical.
*   **Stakeholder Communication:** Translated technical metrics into a plain-English report for non-technical users.

### Technical Implementation:
*   **Scripts:** `vodyati_sai_phanindra/evaluate_models.py`, `vodyati_sai_phanindra/model_evaluation.md`
*   **Testing:** Ran full classification reports and extracted real-world examples of flagged students (e.g., Student #489 flagged with 22 absences in 30 days).
*   **Reporting:** Authored the **Model Evaluation Document**, which explains the "Why" behind each model, making the system ready for a sales demo or board presentation.
