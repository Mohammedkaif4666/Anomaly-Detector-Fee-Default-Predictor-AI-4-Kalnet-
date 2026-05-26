from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import secrets
from typing import List

app = FastAPI(title="KALNET AI-4")

security = HTTPBasic()
REQUIRE_ADMIN_AUTH = True

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if not REQUIRE_ADMIN_AUTH:
        return "admin"
    
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "kalnet2026")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

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
        print(f"[OK] KALNET AI-4 ready -- {len(students_cache)} students loaded.")
    except Exception as e:
        print(f"[ERROR] Startup error: {e}")


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


# ── Analytics API — used by new Admin Dashboard ───────────────
@app.get("/api/analytics")
def analytics():
    """Aggregate analytics data from the pre-computed student cache.
    Returns chart-ready JSON for the Admin Dashboard visualisations."""
    if not students_cache:
        return {}

    total = len(students_cache)

    # Risk distribution (based on risk_score)
    high = sum(1 for s in students_cache if s['risk_score'] > 70)
    medium = sum(1 for s in students_cache if 40 < s['risk_score'] <= 70)
    low = total - high - medium

    # Fee distribution
    on_time = sum(1 for s in students_cache if s['fee_label'] == 0)
    late    = sum(1 for s in students_cache if s['fee_label'] == 1)
    default = sum(1 for s in students_cache if s['fee_label'] == 2)

    # Attendance rate histogram (10-point buckets)
    buckets = {
        "0-50%": 0, "50-60%": 0, "60-70%": 0,
        "70-80%": 0, "80-90%": 0, "90-100%": 0
    }
    for s in students_cache:
        r = s['attendance_rate']
        if r < 50:   buckets["0-50%"] += 1
        elif r < 60: buckets["50-60%"] += 1
        elif r < 70: buckets["60-70%"] += 1
        elif r < 80: buckets["70-80%"] += 1
        elif r < 90: buckets["80-90%"] += 1
        else:        buckets["90-100%"] += 1
    attendance_histogram = [
        {"bucket": k, "count": v} for k, v in buckets.items()
    ]

    # Class-wise breakdown (anomalies + defaults per class)
    class_stats = {}
    for s in students_cache:
        cls = s.get('student_class', 'Unknown')
        if cls not in class_stats:
            class_stats[cls] = {"class": cls, "total": 0,
                                "anomalies": 0, "defaults": 0, "late": 0}
        class_stats[cls]["total"] += 1
        if s['is_anomaly']:      class_stats[cls]["anomalies"] += 1
        if s['fee_label'] == 2:  class_stats[cls]["defaults"]  += 1
        if s['fee_label'] == 1:  class_stats[cls]["late"]      += 1
    class_breakdown = sorted(class_stats.values(), key=lambda x: x['class'])

    # Top 10 highest-risk students (combined attendance + fee risk)
    def combined_risk(s):
        return s['risk_score'] * 0.5 + s['fee_prob'] * 0.5
    top_risk = sorted(students_cache, key=combined_risk, reverse=True)[:10]
    top_risk_students = [{
        "id":              s['id'],
        "name":            s['name'],
        "student_class":   s.get('student_class', '—'),
        "attendance_rate": s['attendance_rate'],
        "risk_score":      s['risk_score'],
        "fee_prob":        s['fee_prob'],
        "is_anomaly":      s['is_anomaly'],
        "fee_label":       s['fee_label'],
    } for s in top_risk]

    # Recent activity: latest 8 flagged students (anomaly or default)
    flagged = [s for s in students_cache if s['is_anomaly'] or s['fee_label'] == 2]
    recent_activity = [{
        "id":     s['id'],
        "name":   s['name'],
        "type":   "Attendance Anomaly" if s['is_anomaly'] else "Fee Default",
        "detail": f"Risk {s['risk_score']}" if s['is_anomaly'] else f"{s['fee_prob']}% default prob",
    } for s in flagged[:8]]

    # Overall health score (0-100, higher = better)
    health = round(100 - (high / total * 40) - (default / total * 60), 1)

    return {
        "total_students":      total,
        "anomaly_count":       sum(1 for s in students_cache if s['is_anomaly']),
        "fee_default_count":   default,
        "fee_late_count":      late,
        "risk_distribution":   {"High": high, "Medium": medium, "Low": low},
        "fee_distribution":    {"On Time": on_time, "Late": late, "Default": default},
        "attendance_histogram": attendance_histogram,
        "class_breakdown":     class_breakdown,
        "top_risk_students":   top_risk_students,
        "recent_activity":     recent_activity,
        "system_health":       max(0, health),
        "model_metrics": {
            "anomaly_model":    "IsolationForest",
            "anomaly_recall":   63,
            "fee_model":        "GradientBoosting",
            "fee_recall":       70,
            "total_features":   10,
            "training_records": 72000,
        },
    }


# ── Serve UI / Landing Page ────────────────────────────────────
@app.get("/")
def root():
    """Serves the public-facing landing/marketing page as the home page."""
    return FileResponse(os.path.join(TEMPLATES, "landing.html"))


# ── Serve Student Directory ───────────────────────────────────
@app.get("/students")
def student_directory():
    """Serves the student list dashboard table."""
    return FileResponse(os.path.join(TEMPLATES, "index.html"))


# ── Serve Admin Analytics Dashboard ───────────────────────────
@app.get("/dashboard")
def admin_dashboard(username: str = Depends(verify_admin)):
    """Serves the Admin Analytics Dashboard with charts and insights."""
    return FileResponse(os.path.join(TEMPLATES, "dashboard.html"))


app.mount("/static", StaticFiles(directory=TEMPLATES), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
