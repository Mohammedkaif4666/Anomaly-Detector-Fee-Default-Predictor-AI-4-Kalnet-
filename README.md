# KALNET AI-4: Anomaly Detector + Fee Default Predictor

**System:** System 5  
**Track:** AI / ML  
**Issued by:** Rishav Raj, CTO & Co-Founder | KALNET  

## Overview
This repository contains the complete machine learning backend for KALNET System 5. It features two specialized models built to run natively in schools without requiring massive cloud infrastructure.

1. **Attendance Anomaly Detector:** Watches student attendance patterns and flags sudden drops or unusual absence streaks, allowing early intervention for students facing hidden issues.
2. **Fee Default Predictor:** Analyzes historical fee payments, family income brackets, and past terms to predict which students are at high risk of defaulting on the upcoming term's fees.

## Tech Stack
- **Python 3.11**
- **Scikit-learn** (Machine Learning algorithms)
- **Pandas / NumPy** (Data generation and feature engineering)
- **FastAPI / Uvicorn** (RESTful API development)
- **Joblib** (Model serialization)

## Team & Folder Structure
To maintain a strict separation of concerns, the project is structured according to the team members responsible for each phase of the pipeline:

```
KALNET-AI-4/
├── mohammed_kaif/          # Pod Lead + Data Engineer
│   ├── generate_raw_data.py   # Generates realistic synthetic attendance & fee data
│   ├── raw_attendance.csv
│   └── raw_fee.csv
├── rohith_koppu/           # Data Pipeline Engineer
│   ├── feature_engineering.py # Extracts ML-ready features (streaks, variances, outstanding fees)
│   ├── data_quality_checks.py # Asserts constraints and validates data integrity
│   ├── attendance_features.csv
│   └── fee_features.csv
├── om_dattatray_wabale/    # ML Engineer 1 (Attendance)
│   ├── exploration.ipynb      # EDA and algorithm tuning for Anomaly Detection
│   ├── train.py               # Trains the IsolationForest model
│   └── attendance_model.pkl   # Serialized model & scaler
├── are_samhith/            # ML Engineer 2 (Fee Prediction)
│   ├── exploration.ipynb      # EDA and class imbalance handling for Fee Prediction
│   ├── train.py               # Trains the GradientBoostingClassifier
│   └── fee_model.pkl          # Serialized model
├── jyothsna_lenka/         # API Developer
│   ├── app.py                 # FastAPI application serving both models
│   └── requirements.txt       # Project dependencies
├── vodyati_sai_phanindra/  # QA + Evaluator
│   ├── evaluate_models.py     # Generates classification reports and sample extractions
│   └── model_evaluation.md    # Plain-English documentation for sales demos
└── README.md               # You are here
```

## Performance & Algorithms

### 1. Attendance Anomaly Detector
- **Algorithm:** Isolation Forest
- **Target Recall:** $\ge$ 60%
- **Actual Recall:** **64%** (Precision: 90%)
- **Data Normalization:** `StandardScaler` applied prior to training.
- **Scoring:** Uses `decision_function(X)` mapped to a 0-100 continuous risk score.

### 2. Fee Default Predictor
- **Algorithm:** Gradient Boosting Classifier
- **Target Recall:** $\ge$ 70%
- **Actual Recall:** **80%** (via custom thresholding to prioritize catching defaulters)
- **Class Balancing:** Utilized `compute_sample_weight('balanced')` to account for the highly skewed 5% default rate.

## How to Run the Project Locally

### 1. Setup Environment
Ensure you have Python 3.11+ installed.
```bash
pip install -r jyothsna_lenka/requirements.txt
```

### 2. Data Pipeline
First, generate the raw synthetic data, then engineer the ML features, and finally run data quality checks to assert data integrity:
```bash
python mohammed_kaif/generate_raw_data.py
python rohith_koppu/feature_engineering.py
python rohith_koppu/data_quality_checks.py
```

### 3. Train Models
Train the Anomaly Detector and Fee Predictor models sequentially:
```bash
python om_dattatray_wabale/train.py
python are_samhith/train.py
```

### 4. Quality Assurance (Optional)
To verify the models meet recall targets and to generate intuitive examples:
```bash
python vodyati_sai_phanindra/evaluate_models.py
```

### 5. Start the FastAPI Server
Launch the REST API to expose the models to the FS-3 Admin Dashboard:
```bash
uvicorn jyothsna_lenka.app:app --reload
```
- The server will boot at `http://127.0.0.1:8000`
- Available endpoints: `POST /ai/anomalies` and `POST /ai/fee-risk`
- Swagger UI Documentation is available at `http://127.0.0.1:8000/docs`

## Final Notes
- Both models are loaded into memory asynchronously during the FastAPI `startup` event, ensuring zero file I/O blocking inside the route handlers.
- The repository is fully prepared to be pushed directly to GitHub as requested.
