from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import List

app = FastAPI(title="KALNET AI-4")

models = {}
features_db = {}
students_cache = []

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(ROOT, "templates")

NAMES = [
    "Aarav Sharma","Priya Patel","Rahul Gupta","Anjali Singh","Vikram Reddy",
    "Sanya Iyer","Arjun Nair","Meera Joshi","Rohan Verma","Kavya Menon",
    "Dev Malhotra","Isha Rao","Aditya Kumar","Pooja Bhat","Karan Mishra",
    "Sneha Das","Harsh Agarwal","Riya Thakur","Nikhil Pandey","Divya Chauhan",
]

@app.on_event("startup")
def startup():
    global students_cache
    try:
        models['att'] = joblib.load(os.path.join(ROOT, 'models/attendance_anomaly/model.pkl'))
        models['scaler'] = joblib.load(os.path.join(ROOT, 'models/attendance_anomaly/scaler.pkl'))
        models['fee'] = joblib.load(os.path.join(ROOT, 'models/fee_predictor/model.pkl'))
        thresh_path = os.path.join(ROOT, 'models/fee_predictor/threshold.pkl')
        models['fee_thresh'] = joblib.load(thresh_path) if os.path.exists(thresh_path) else 0.5

        att = pd.read_csv(os.path.join(ROOT, 'data/attendance_features.csv')).reset_index(drop=True)
        fee = pd.read_csv(os.path.join(ROOT, 'data/fee_features.csv')).reset_index(drop=True)
        features_db['att'] = att
        features_db['fee'] = fee

        X_att = att.drop(['student_id', 'is_anomaly'], axis=1).values
        X_scaled = models['scaler'].transform(X_att)
        att_preds = models['att'].predict(X_scaled)
        att_scores = np.clip(100 * (1 - (models['att'].decision_function(X_scaled) + 0.5) / 1.0), 0, 100)

        X_fee = fee.drop(['student_id', 'label'], axis=1).values
        fee_proba_all = models['fee'].predict_proba(X_fee)
        default_col = list(models['fee'].classes_).index(2)
        fee_probs = fee_proba_all[:, default_col]
        fee_thresh = models.get('fee_thresh', 0.5)
        # Use threshold-adjusted labels so UI badges match model decisions
        fee_labels_pred = np.where(fee_probs >= fee_thresh, 2,
                          np.argmax(fee_proba_all[:, [i for i in range(fee_proba_all.shape[1]) if i != default_col]], axis=1))
        fee_classes_no_default = [c for c in models['fee'].classes_ if c != 2]
        fee_labels_adjusted = []
        for lp in fee_labels_pred:
            if lp == 2:
                fee_labels_adjusted.append(2)
            else:
                fee_labels_adjusted.append(int(fee_classes_no_default[lp]))

        fee_map = {fee.loc[i, 'student_id']: {
            'prob': float(fee_probs[i]),
            'label': int(fee_labels_adjusted[i]),
            'outstanding': float(fee.loc[i, 'total_outstanding']),
            'days_late': int(fee.loc[i, 'days_since_last_payment']),
        } for i in range(len(fee))}

        # Load class info if available
        labels_path = os.path.join(ROOT, 'data/student_labels.csv')
        class_map = {}
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            if 'class' in labels_df.columns:
                class_map = dict(zip(labels_df['student_id'], labels_df['class']))

        for i, row in att.iterrows():
            sid = row['student_id']
            num = int(sid.split('_')[1]) - 1
            name = NAMES[num % len(NAMES)]
            fd = fee_map.get(sid, {'prob': 0, 'label': 0, 'outstanding': 0, 'days_late': 0})
            classes = ["6A","6B","7A","7B","8A","8B","9A","9B","10A","10B"]
            students_cache.append({
                "id": sid,
                "name": name,
                "student_class": class_map.get(sid, classes[num % len(classes)]),
                "attendance_rate": round(float(row['attendance_rate']) * 100, 1),
                "is_anomaly": bool(att_preds[i] == -1),
                "risk_score": round(float(att_scores[i]), 1),
                "absence_streak": int(row['longest_absence_streak']),
                "absence_30d": int(row['absence_in_last_30_days']),
                "fee_label": fd['label'],
                "fee_prob": round(fd['prob'] * 100, 1),
                "outstanding": fd['outstanding'],
                "days_late": fd['days_late'],
            })
        print(f"✅ KALNET AI-4 ready — {len(students_cache)} students loaded.")
    except Exception as e:
        print(f"❌ Startup error: {e}")


# ── Dashboard summary ──────────────────────────────────────────
@app.get("/api/summary")
def summary():
    return {
        "total": len(students_cache),
        "anomalies": sum(1 for s in students_cache if s['is_anomaly']),
        "fee_defaults": sum(1 for s in students_cache if s['fee_label'] == 2),
        "fee_late": sum(1 for s in students_cache if s['fee_label'] == 1),
    }


# ── Student list with optional search & filter ─────────────────
@app.get("/api/students")
def list_students(search: str = "", filter: str = "all"):
    data = students_cache
    if search:
        q = search.lower()
        data = [s for s in data if q in s['name'].lower() or q in s['id'].lower()]
    if filter == "anomaly":
        data = [s for s in data if s['is_anomaly']]
    elif filter == "fee_risk":
        data = [s for s in data if s['fee_label'] == 2]
    elif filter == "safe":
        data = [s for s in data if not s['is_anomaly'] and s['fee_label'] == 0]
    return data


# ── Individual student detail with 30-day history ──────────────
@app.get("/api/student/{student_id}")
def get_student(student_id: str):
    s = next((x for x in students_cache if x['id'] == student_id), None)
    if not s:
        raise HTTPException(404, "Student not found")
    raw_path = os.path.join(ROOT, 'data/attendance_raw.csv')
    raw = pd.read_csv(raw_path)
    hist = raw[raw['student_id'] == student_id].tail(30)
    result = dict(s)
    result['history'] = hist[['date', 'is_present']].to_dict('records')
    return result


# ── Original POST endpoints (kept intact) ─────────────────────
class StudentRequest(BaseModel):
    student_ids: List[str]


@app.post("/ai/anomalies")
def get_anomalies(req: StudentRequest):
    df = features_db['att'][features_db['att']['student_id'].isin(req.student_ids)]
    if df.empty:
        return {"results": []}
    X = df.drop(['student_id', 'is_anomaly'], axis=1).values
    Xs = models['scaler'].transform(X)
    preds = models['att'].predict(Xs)
    scores = np.clip(100 * (1 - (models['att'].decision_function(Xs) + 0.5) / 1.0), 0, 100)
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append({
            "student_id": row['student_id'],
            "risk_score": round(float(scores[i]), 2),
            "is_flagged": bool(preds[i] == -1),
            "risk_level": "High" if scores[i] > 70 else "Medium" if scores[i] > 40 else "Low",
        })
    return {"results": results}


@app.post("/ai/fee-risk")
def get_fee_risk(req: StudentRequest):
    df = features_db['fee'][features_db['fee']['student_id'].isin(req.student_ids)]
    if df.empty:
        return {"results": []}
    X = df.drop(['student_id', 'label'], axis=1).values
    probs = models['fee'].predict_proba(X)[:, 2]
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append({
            "student_id": row['student_id'],
            "default_probability": round(float(probs[i]), 2),
            "risk_category": "High" if probs[i] > 0.6 else "Medium" if probs[i] > 0.3 else "Low",
        })
    return {"results": results}


# ── Serve UI ───────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse(os.path.join(TEMPLATES, "index.html"))

app.mount("/static", StaticFiles(directory=TEMPLATES), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
