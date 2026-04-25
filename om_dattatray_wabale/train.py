import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score
import os

# Features to use for anomaly detection
FEATURE_COLS = [
    'attendance_rate', 
    'longest_absence_streak', 
    'absence_in_last_30_days', 
    'day_of_week_variance'
]

def train_model():
    print("Training Attendance Anomaly Detection Model...")
    
    # Load data
    df = pd.read_csv("../rohith_koppu/attendance_features.csv")
    X = df[FEATURE_COLS]
    y_true = df['is_anomalous']
    
    # Normalise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train IsolationForest
    # contamination=0.1 since we expect roughly 10-14% anomalies
    model = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
    model.fit(X_scaled)
    
    # Evaluate
    # IsolationForest returns -1 for anomaly, 1 for normal
    preds = model.predict(X_scaled)
    y_pred = np.where(preds == -1, 1, 0)
    
    recall = recall_score(y_true, y_pred)
    print(f"Model Recall: {recall:.2f}")
    
    if recall >= 0.60:
        print("Success: Recall target met!")
    else:
        print("Warning: Recall target NOT met.")
        
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Save the scaler and model together
    artifacts = {
        'scaler': scaler,
        'model': model,
        'features': FEATURE_COLS
    }
    
    os.makedirs(os.path.dirname("attendance_model.pkl") or ".", exist_ok=True)
    joblib.dump(artifacts, "attendance_model.pkl")
    print("Saved attendance_model.pkl")

def flag_anomalies(df):
    """
    Returns DataFrame with risk scores and flags for the given students.
    """
    artifacts = joblib.load("attendance_model.pkl")
    scaler = artifacts['scaler']
    model = artifacts['model']
    features = artifacts['features']
    
    # Ensure all features are present
    X = df[features]
    X_scaled = scaler.transform(X)
    
    # Predict (-1 is anomaly, 1 is normal)
    preds = model.predict(X_scaled)
    
    # Get risk score: decision_function
    # Negative scores are anomalies, positive are normal.
    # Lower value means more anomalous.
    scores = model.decision_function(X_scaled)
    
    # Normalize risk score to 0-100 scale where 100 is highest risk
    # decision_function typically ranges between -0.5 and 0.5 roughly
    # We can use MinMax scaling on the inverted scores, or a fixed threshold.
    # Let's invert the scores so positive is higher risk
    inverted_scores = -scores
    
    # To normalize 0-100, let's map the min observed to 0 and max to 100
    # For a robust approach, we can map 0 decision_function to 50 risk score.
    # But for simplicity, we'll just use a MinMax approach
    min_score, max_score = inverted_scores.min(), inverted_scores.max()
    
    if max_score > min_score:
        risk_scores_100 = ((inverted_scores - min_score) / (max_score - min_score)) * 100
    else:
        risk_scores_100 = np.zeros_like(inverted_scores)
        
    results = pd.DataFrame({
        'student_id': df['student_id'],
        'risk_score': np.round(risk_scores_100, 2),
        'is_flagged': np.where(preds == -1, True, False)
    })
    
    # Assign risk level
    def get_risk_level(row):
        if row['is_flagged']:
            return "High"
        elif row['risk_score'] > 60:
            return "Medium"
        return "Low"
        
    results['risk_level'] = results.apply(get_risk_level, axis=1)
    
    return results

if __name__ == "__main__":
    # Change cwd if running directly to ensure relative paths work
    import sys
    if os.path.basename(os.getcwd()) != 'om_dattatray_wabale':
        try:
            os.chdir('om_dattatray_wabale')
        except FileNotFoundError:
            pass
            
    train_model()
