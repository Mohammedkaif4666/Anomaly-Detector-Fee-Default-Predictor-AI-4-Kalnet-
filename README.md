# KALNET AI-4: Anomaly Detector & Fee Default Predictor

This project implements an advanced Machine Learning system designed for schools to proactively detect students at risk due to attendance anomalies and predict potential fee defaults.

## 🚀 Project Vision
The system acts as an **"Early Warning Radar"** for school administrators. 

*   **Attendance Anomaly**: Instead of just tracking daily presence, the AI analyzes patterns over 200 days to detect sudden drops (e.g., from 92% to 34% in 3 weeks) that might signal family issues or health crises, catching risks weeks before they become permanent problems.
*   **Fee Default Predictor**: By analyzing payment history, income brackets, and household factors, the AI identifies students at risk of missing next term's payment, allowing schools to reach out early with flexible support.

## 🛠️ Modular Structure (Team Roles)

The project is organized into modular directories representing the contributions of the KALNET AI-4 Team:

*   **`mohammed_kaif/`**: Data Generation & Core Feature Engineering.
    *   `data_generator.py`: Generates synthetic data for 500 students across 200 days.
    *   `feature_engineering.py`: Processes raw logs into student-level ML features.
*   **`om_dattatray_wabale/`**: Attendance Anomaly Detection.
    *   `train.py`: Trains an `IsolationForest` model to detect unusual attendance patterns.
*   **`are_samhith/`**: Fee Default Prediction.
    *   `train.py`: Trains a `GradientBoostingClassifier` to predict payment risks.
*   **`jyothsna_lenka/`**: FastAPI Deployment.
    *   `main.py`: Production-ready API for real-time risk assessment.
*   **`rohith_koppu/`**: Feature Verification & Quality Control.
*   **`vodyati_sai_phanindra/`**: Evaluation & Documentation.
    *   `evaluate.py`: Generates performance reports and plain-English documentation.

## 🏃‍♂️ How to Run

Follow these steps to set up and execute the project locally:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the ML Pipeline (Data -> Features -> Training -> Evaluation)
Execute the following commands in order to prepare the data and models:
```bash
# Generate raw data and engineer features
python mohammed_kaif/data_generator.py
python mohammed_kaif/feature_engineering.py

# Train both AI models
python om_dattatray_wabale/train.py
python are_samhith/train.py

# Generate final evaluation reports
python vodyati_sai_phanindra/evaluate.py
```

### 3. Start the API Server
Launch the FastAPI backend:
```bash
uvicorn jyothsna_lenka.main:app --reload
```

### 4. Interactive API Documentation & Demo
Once the server is running, open your web browser and navigate to:
**`http://localhost:8000/docs`**

Type **`/docs`** in your browser's address bar to access the interactive Swagger UI. Here you can:
*   View all available endpoints.
*   Test the model predictions with custom `student_ids`.
*   See live responses from the AI models.

## 📄 Documentation
For a detailed analysis of model performance and plain-English descriptions of the AI logic, refer to:
`docs/model_evaluation.md`
