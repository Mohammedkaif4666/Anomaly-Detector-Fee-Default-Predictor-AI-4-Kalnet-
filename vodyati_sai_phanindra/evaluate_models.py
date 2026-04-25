import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
import os
import sys

# Change to project root if needed
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
os.chdir(project_root)

def evaluate_attendance_model():
    print("=== Attendance Anomaly Model Evaluation ===")
    
    df = pd.read_csv("rohith_koppu/attendance_features.csv")
    artifacts = joblib.load("om_dattatray_wabale/attendance_model.pkl")
    
    scaler = artifacts['scaler']
    model = artifacts['model']
    features = artifacts['features']
    
    X = df[features]
    y_true = df['is_anomalous']
    
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    y_pred = np.where(preds == -1, 1, 0)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # 5 Examples
    df['is_flagged'] = y_pred
    anomalies = df[df['is_flagged'] == 1].head(5)
    
    print("\n5 Example Flagged Attendance Anomalies:")
    for _, row in anomalies.iterrows():
        print(f"Student {int(row['student_id'])} | Actual Anomaly: {int(row['is_anomalous'])} | "
              f"Attendance Rate: {row['attendance_rate']:.2f} | "
              f"Longest Streak: {int(row['longest_absence_streak'])} | "
              f"Last 30 Days Absence: {int(row['absence_in_last_30_days'])}")
    print("\n")

def evaluate_fee_model():
    print("=== Fee Default Predictor Model Evaluation ===")
    
    df = pd.read_csv("rohith_koppu/fee_features.csv")
    artifacts = joblib.load("are_samhith/fee_model.pkl")
    
    model = artifacts['model']
    features = artifacts['features']
    
    X = df[features]
    y_true = df['will_default']
    
    probs = model.predict_proba(X)[:, 1]
    # Using the adjusted threshold 0.3
    y_pred = (probs >= 0.3).astype(int)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # 5 Examples
    df['default_probability'] = probs
    df['predicted_default'] = y_pred
    high_risk = df[df['predicted_default'] == 1].sort_values(by='default_probability', ascending=False).head(5)
    
    print("\n5 Example High-Risk Fee Predictions:")
    for _, row in high_risk.iterrows():
        print(f"Student {int(row['student_id'])} | Actual Default: {int(row['will_default'])} | "
              f"Probability: {row['default_probability']:.2f} | "
              f"Total Outstanding: {row['total_outstanding']} | "
              f"Previous Status: {row['previous_term_status']}")
              
if __name__ == "__main__":
    evaluate_attendance_model()
    evaluate_fee_model()
