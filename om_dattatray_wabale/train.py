import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, classification_report

def train_attendance_model():
    print("Training Attendance Anomaly Model...")
    
    # Load features
    if not os.path.exists('data/attendance_features.csv'):
        print("Features not found. Run feature_engineering.py first.")
        return

    df = pd.read_csv('data/attendance_features.csv')
    X = df.drop(['student_id', 'is_anomaly'], axis=1)
    y_true = df['is_anomaly']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    model = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
    model.fit(X_scaled)
    
    # Get predictions and scores
    # IsolationForest returns -1 for anomalies, 1 for normal
    y_pred = model.predict(X_scaled)
    y_pred = [1 if x == -1 else 0 for x in y_pred] # Convert to our label format
    
    # Risk score: more negative is higher risk
    decision_scores = model.decision_function(X_scaled)
    # Normalize to 0-100 (where 100 is highest risk)
    # decision_function returns values in roughly [-0.5, 0.5]
    # Let's map it: higher risk score for more anomalous
    risk_scores = 100 * (1 - (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min()))
    
    # Evaluate
    recall = recall_score(y_true, y_pred)
    print(f"Model Recall: {recall:.2f}")
    print(classification_report(y_true, y_pred))
    
    # Save model and scaler
    os.makedirs('models/attendance_anomaly', exist_ok=True)
    joblib.dump(model, 'models/attendance_anomaly/model.pkl')
    joblib.dump(scaler, 'models/attendance_anomaly/scaler.pkl')
    print("Model saved to models/attendance_anomaly/model.pkl")

def flag_anomalies(df):
    """
    Function to be used in production/API
    Returns {student_id, risk_score, is_flagged, risk_level}
    """
    model = joblib.load('models/attendance_anomaly/model.pkl')
    scaler = joblib.load('models/attendance_anomaly/scaler.pkl')
    
    X = df.drop(['student_id'], axis=1, errors='ignore')
    X_scaled = scaler.transform(X)
    
    y_pred = model.predict(X_scaled)
    decision_scores = model.decision_function(X_scaled)
    
    # Normalize risk score to 0-100
    # For simplicity in this function, we'll use a fixed mapping or re-calculate based on training range
    # In a real app, we'd store the min/max from training
    risk_scores = 100 * (1 - (decision_scores + 0.5) / 1.0) # Rough normalization
    risk_scores = np.clip(risk_scores, 0, 100)
    
    results = []
    for i, stu_id in enumerate(df['student_id']):
        risk = risk_scores[i]
        level = "High" if risk > 70 else "Medium" if risk > 40 else "Low"
        results.append({
            'student_id': stu_id,
            'risk_score': round(float(risk), 2),
            'is_flagged': bool(y_pred[i] == -1),
            'risk_level': level
        })
    return results

if __name__ == "__main__":
    train_attendance_model()
