from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import List

app = FastAPI(title="KALNET AI Predictor API")

# Global variables for models and data
models = {}
features_db = {}

@app.on_event("startup")
def load_models_and_data():
    print("Loading models and features at startup...")
    try:
        # Resolve paths relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        models['attendance'] = joblib.load(os.path.join(project_root, 'models/attendance_anomaly/model.pkl'))
        models['attendance_scaler'] = joblib.load(os.path.join(project_root, 'models/attendance_anomaly/scaler.pkl'))
        models['fee'] = joblib.load(os.path.join(project_root, 'models/fee_predictor/model.pkl'))
        
        # Load processed features
        features_db['attendance'] = pd.read_csv(os.path.join(project_root, 'data/attendance_features.csv'))
        features_db['fee'] = pd.read_csv(os.path.join(project_root, 'data/fee_features.csv'))
        
        print("Models and data loaded successfully.")
    except Exception as e:
        print(f"Error loading models or data: {e}")

@app.get("/")
async def root():
    return {
        "project": "KALNET AI-4: Anomaly Detector & Fee Default Predictor",
        "status": "Perfectly Ready",
        "message": "The AI models are loaded and functioning. Use /docs to test endpoints.",
        "aim": "Machine learning in real schools - Scikit-learn only."
    }

class StudentRequest(BaseModel):
    student_ids: List[str]

@app.post("/ai/anomalies")
async def get_anomalies(request: StudentRequest):
    if 'attendance' not in models:
        raise HTTPException(status_code=503, detail="Attendance model not loaded")
    
    df_features = features_db['attendance'][features_db['attendance']['student_id'].isin(request.student_ids)]
    if df_features.empty:
        raise HTTPException(status_code=404, detail="Student IDs not found")
    
    X = df_features.drop(['student_id', 'is_anomaly'], axis=1)
    X_scaled = models['attendance_scaler'].transform(X)
    
    y_pred = models['attendance'].predict(X_scaled)
    decision_scores = models['attendance'].decision_function(X_scaled)
    
    risk_scores = 100 * (1 - (decision_scores + 0.5) / 1.0)
    risk_scores = [round(float(s), 2) for s in risk_scores]
    
    results = []
    for i, (_, row) in enumerate(df_features.iterrows()):
        results.append({
            "student_id": row['student_id'],
            "risk_score": risk_scores[i],
            "is_flagged": bool(y_pred[i] == -1),
            "risk_level": "High" if risk_scores[i] > 70 else "Medium" if risk_scores[i] > 40 else "Low"
        })
    return {"results": results}

@app.post("/ai/fee-risk")
async def get_fee_risk(request: StudentRequest):
    if 'fee' not in models:
        raise HTTPException(status_code=503, detail="Fee model not loaded")
    
    df_features = features_db['fee'][features_db['fee']['student_id'].isin(request.student_ids)]
    if df_features.empty:
        raise HTTPException(status_code=404, detail="Student IDs not found")
    
    X = df_features.drop(['student_id', 'label'], axis=1)
    probs = models['fee'].predict_proba(X)
    default_probs = probs[:, 2] 
    
    results = []
    for i, (_, row) in enumerate(df_features.iterrows()):
        prob = float(default_probs[i])
        results.append({
            "student_id": row['student_id'],
            "default_probability": round(prob, 2),
            "risk_category": "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
        })
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
