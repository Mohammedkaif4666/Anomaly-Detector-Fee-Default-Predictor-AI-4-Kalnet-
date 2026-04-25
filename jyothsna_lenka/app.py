from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="KALNET AI-4 API", description="Anomaly Detector and Fee Default Predictor")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to KALNET AI-4 API",
        "documentation": "To test the API, go to /docs",
        "endpoints": {
            "Attendance Anomalies": "/ai/anomalies [POST]",
            "Fee Default Risk": "/ai/fee-risk [POST]"
        }
    }


# Global variables for models and data
attendance_artifacts = None
fee_artifacts = None
attendance_data = None
fee_data = None

# Pydantic models
class StudentRequest(BaseModel):
    student_ids: List[int]

@app.on_event("startup")
def load_models():
    global attendance_artifacts, fee_artifacts, attendance_data, fee_data
    
    # Define paths assuming we run from project root or jyothsna_lenka folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    attendance_model_path = os.path.join(project_root, "om_dattatray_wabale", "attendance_model.pkl")
    fee_model_path = os.path.join(project_root, "are_samhith", "fee_model.pkl")
    attendance_data_path = os.path.join(project_root, "rohith_koppu", "attendance_features.csv")
    fee_data_path = os.path.join(project_root, "rohith_koppu", "fee_features.csv")
    
    try:
        attendance_artifacts = joblib.load(attendance_model_path)
        fee_artifacts = joblib.load(fee_model_path)
        
        # Load feature data into memory so we can query it quickly
        attendance_data = pd.read_csv(attendance_data_path)
        fee_data = pd.read_csv(fee_data_path)
        print("Models and data loaded successfully at startup.")
    except Exception as e:
        print(f"Error loading models or data: {e}")

@app.post("/ai/anomalies")
def get_anomalies(request: StudentRequest):
    if attendance_artifacts is None or attendance_data is None:
        raise HTTPException(status_code=500, detail="Attendance model or data not loaded.")
        
    student_ids = request.student_ids
    
    # Filter data for requested students
    df = attendance_data[attendance_data['student_id'].isin(student_ids)].copy()
    if df.empty:
        return {"results": []}
        
    scaler = attendance_artifacts['scaler']
    model = attendance_artifacts['model']
    features = attendance_artifacts['features']
    
    X = df[features]
    X_scaled = scaler.transform(X)
    
    preds = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    
    # Normalize risk score to 0-100
    inverted_scores = -scores
    min_score, max_score = inverted_scores.min(), inverted_scores.max()
    
    if max_score > min_score:
        risk_scores_100 = ((inverted_scores - min_score) / (max_score - min_score)) * 100
    else:
        risk_scores_100 = np.zeros_like(inverted_scores)
        
    df['risk_score'] = np.round(risk_scores_100, 2)
    df['is_flagged'] = np.where(preds == -1, True, False)
    
    def get_risk_level(row):
        if row['is_flagged']:
            return "High"
        elif row['risk_score'] > 60:
            return "Medium"
        return "Low"
        
    df['risk_level'] = df.apply(get_risk_level, axis=1)
    
    results = df[['student_id', 'risk_score', 'is_flagged', 'risk_level']].to_dict(orient="records")
    return {"results": results}

@app.post("/ai/fee-risk")
def get_fee_risk(request: StudentRequest):
    if fee_artifacts is None or fee_data is None:
        raise HTTPException(status_code=500, detail="Fee predictor model or data not loaded.")
        
    student_ids = request.student_ids
    
    df = fee_data[fee_data['student_id'].isin(student_ids)].copy()
    if df.empty:
        return {"results": []}
        
    model = fee_artifacts['model']
    features = fee_artifacts['features']
    
    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    
    df['default_probability'] = np.round(probs, 2)
    
    def get_category(prob):
        if prob > 0.6:
            return "High Risk"
        elif prob > 0.3:
            return "Medium Risk"
        return "Low Risk"
        
    df['risk_category'] = df['default_probability'].apply(get_category)
    
    results = df[['student_id', 'default_probability', 'risk_category']].to_dict(orient="records")
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
