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
startup_error_msg = "No error"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(ROOT, "templates")

NAMES = [
    "Aarav Sharma","Priya Patel","Rahul Gupta","Anjali Singh","Vikram Reddy",
    "Sanya Iyer","Arjun Nair","Meera Joshi","Rohan Verma","Kavya Menon",
    "Dev Malhotra","Isha Rao","Aditya Kumar","Pooja Bhat","Karan Mishra",
    "Sneha Das","Harsh Agarwal","Riya Thakur","Nikhil Pandey","Divya Chauhan",
]

def _load_or_retrain_models():
    """Load pre-trained .pkl models. Supports both V2 (old) and V3 (new bundle) formats.
    If they fail, fallback to retraining from CSV data."""
    from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    att_model_path  = os.path.join(ROOT, 'models/attendance_anomaly/model.pkl')
    att_scaler_path = os.path.join(ROOT, 'models/attendance_anomaly/scaler.pkl')
    fee_bundle_path = os.path.join(ROOT, 'models/fee_predictor/model_v3.pkl')
    fee_model_path  = os.path.join(ROOT, 'models/fee_predictor/model.pkl')  # V2 fallback
    fee_thresh_path = os.path.join(ROOT, 'models/fee_predictor/threshold.pkl')

    # ── Try loading attendance model ──────────────────────────
    try:
        models['att']    = joblib.load(att_model_path)
        models['scaler'] = joblib.load(att_scaler_path)
        print("[OK] Attendance model loaded from .pkl")
    except Exception as e:
        print(f"[WARN] Attendance pkl load failed ({e}), retraining…")
        att_df = pd.read_csv(os.path.join(ROOT, 'data/attendance_features.csv'))
        X_a = att_df.drop(['student_id', 'is_anomaly'], axis=1).values
        scaler = StandardScaler()
        X_a_s = scaler.fit_transform(X_a)
        iso = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
        iso.fit(X_a_s)
        models['att']    = iso
        models['scaler'] = scaler
        os.makedirs(os.path.dirname(att_model_path), exist_ok=True)
        try:
            joblib.dump(iso,    att_model_path)
            joblib.dump(scaler, att_scaler_path)
        except Exception:
            pass
        print("[OK] Attendance model retrained successfully")

    # ── Try loading fee model (V3 bundle first, V2 fallback) ──
    fee_model_version = "unknown"
    try:
        # Try V3 bundle first
        if os.path.exists(fee_bundle_path):
            bundle = joblib.load(fee_bundle_path)
            models['fee'] = bundle['model']
            models['fee_thresh'] = bundle['threshold']
            models['fee_feature_cols'] = bundle['feature_cols']
            fee_model_version = "V3 (bundle)"
            print("[OK] Fee model V3 loaded from bundle")
        else:
            raise FileNotFoundError("V3 bundle not found")
    except Exception as e:
        print(f"[WARN] Fee model V3 load failed ({e}), trying V2…")
        try:
            # Fall back to V2 format
            models['fee'] = joblib.load(fee_model_path)
            models['fee_thresh'] = joblib.load(fee_thresh_path) if os.path.exists(fee_thresh_path) else 0.5
            models['fee_feature_cols'] = None
            fee_model_version = "V2 (legacy)"
            print("[OK] Fee model V2 loaded from .pkl")
        except Exception as e2:
            print(f"[WARN] Fee model V2 load also failed ({e2}), retraining V2…")
            # Retrain V2
            fee_df = pd.read_csv(os.path.join(ROOT, 'data/fee_features.csv'))
            X_f = fee_df.drop(['student_id', 'label'], axis=1).values
            y_f = fee_df['label'].values
            sample_w = np.where(y_f == 2, 4.0, 1.0)
            gbc = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.08,
                max_depth=4, random_state=42
            )
            gbc.fit(X_f, y_f, sample_weight=sample_w)
            models['fee'] = gbc
            models['fee_thresh'] = 0.5
            models['fee_feature_cols'] = None
            fee_model_version = "V2 (retrained)"
            os.makedirs(os.path.dirname(fee_model_path), exist_ok=True)
            try:
                joblib.dump(gbc, fee_model_path)
                joblib.dump(0.5, fee_thresh_path)
            except Exception:
                pass
            print("[OK] Fee model V2 retrained successfully")
    
    models['fee_model_version'] = fee_model_version


@app.on_event("startup")
def startup():
    global students_cache
    try:
        # ── Load / retrain models ─────────────────────────────
        _load_or_retrain_models()

        att = pd.read_csv(os.path.join(ROOT, 'data/attendance_features.csv')).reset_index(drop=True)
        
        # Try V3 features first, fall back to V2
        fee_v3_path = os.path.join(ROOT, 'data/fee_features_v3.csv')
        fee_v2_path = os.path.join(ROOT, 'data/fee_features.csv')
        
        if os.path.exists(fee_v3_path):
            fee = pd.read_csv(fee_v3_path).reset_index(drop=True)
            fee_version = "V3"
        else:
            fee = pd.read_csv(fee_v2_path).reset_index(drop=True)
            fee_version = "V2"
        
        features_db['att'] = att
        features_db['fee'] = fee

        # ── Attendance predictions ─────────────────────────────
        X_att = att.drop(['student_id', 'is_anomaly'], axis=1).values
        X_scaled = models['scaler'].transform(X_att)
        att_preds = models['att'].predict(X_scaled)
        att_scores = np.clip(100 * (1 - (models['att'].decision_function(X_scaled) + 0.5) / 1.0), 0, 100)

        # ── Fee predictions (V3 or V2) ───────────────────────
        fee_map = {}
        
        if fee_version == "V3" and models.get('fee_feature_cols'):
            # V3 model with 2-term features
            print("[INFO] Using V3 fee model with 2-term history features")
            
            # Compute derived features for inference (must match training)
            fee_derived = fee.copy()
            fee_derived['outstanding_growth'] = fee_derived['t2_outstanding'] - fee_derived['t1_outstanding']
            fee_derived['days_late_trend'] = fee_derived['t2_days_late'] - fee_derived['t1_days_late']
            fee_derived['both_terms_late'] = ((fee_derived['t1_status'] >= 1) & (fee_derived['t2_status'] >= 1)).astype(int)
            fee_derived['escalating'] = (fee_derived['t2_status'] > fee_derived['t1_status']).astype(int)
            fee_derived['avg_outstanding'] = (fee_derived['t1_outstanding'] + fee_derived['t2_outstanding']) / 2
            fee_derived['max_days_late'] = fee_derived[['t1_days_late', 't2_days_late']].max(axis=1)
            
            X_fee = fee_derived[models['fee_feature_cols']].values
            fee_proba_all = models['fee'].predict_proba(X_fee)
            fee_probs = fee_proba_all[:, 1]  # V3 is binary: class 1 = default
            fee_thresh = models.get('fee_thresh', 0.5)
            
            for i in range(len(fee)):
                sid = fee.loc[i, 'student_id']
                fee_map[sid] = {
                    'prob': float(fee_probs[i]),
                    'label': 2 if fee_probs[i] >= fee_thresh else 0,
                    'outstanding': float(fee.loc[i, 't2_outstanding']),
                    'days_late': int(fee.loc[i, 't2_days_late']),
                }
        else:
            # V2 model with single-term features
            print(f"[INFO] Using V2 fee model with single-term features")
            
            X_fee = fee.drop(['student_id', 'label'], axis=1).values
            fee_proba_all = models['fee'].predict_proba(X_fee)
            
            # Check if multiclass or binary
            if fee_proba_all.shape[1] == 3:
                # V2 multiclass: 0=OnTime, 1=Late, 2=Default
                default_col = list(models['fee'].classes_).index(2)
                fee_probs = fee_proba_all[:, default_col]
                fee_thresh = models.get('fee_thresh', 0.5)
                fee_labels_pred = np.where(fee_probs >= fee_thresh, 2,
                                  np.argmax(fee_proba_all[:, [i for i in range(fee_proba_all.shape[1]) if i != default_col]], axis=1))
                fee_classes_no_default = [c for c in models['fee'].classes_ if c != 2]
                fee_labels_adjusted = []
                for lp in fee_labels_pred:
                    if lp == 2:
                        fee_labels_adjusted.append(2)
                    else:
                        fee_labels_adjusted.append(int(fee_classes_no_default[lp]))
                
                for i in range(len(fee)):
                    sid = fee.loc[i, 'student_id']
                    fee_map[sid] = {
                        'prob': float(fee_probs[i]),
                        'label': int(fee_labels_adjusted[i]),
                        'outstanding': float(fee.loc[i, 'total_outstanding']),
                        'days_late': int(fee.loc[i, 'days_since_last_payment']),
                    }
            else:
                # V2 binary or other format
                fee_probs = fee_proba_all[:, 1] if fee_proba_all.shape[1] > 1 else fee_proba_all[:, 0]
                fee_thresh = models.get('fee_thresh', 0.5)
                
                for i in range(len(fee)):
                    sid = fee.loc[i, 'student_id']
                    fee_map[sid] = {
                        'prob': float(fee_probs[i]),
                        'label': 2 if fee_probs[i] >= fee_thresh else 0,
                        'outstanding': float(fee.loc[i, 'total_outstanding']),
                        'days_late': int(fee.loc[i, 'days_since_last_payment']),
                    }

        # Load class info if available
        labels_path = os.path.join(ROOT, 'data/student_labels.csv')
        class_map = {}
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            if 'class' in labels_df.columns:
                class_map = dict(zip(labels_df['student_id'], labels_df['class']))

        # ── Build student cache ────────────────────────────────
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
        print(f"[INFO] Fee model: {models.get('fee_model_version', 'unknown')}")
        
    except Exception as e:
        global startup_error_msg
        startup_error_msg = str(e)
        print(f"[ERROR] Startup error: {e}")
        import traceback
        traceback.print_exc()


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
    
    fee_thresh = models.get('fee_thresh', 0.5)
    
    if models.get('fee_feature_cols'):
        # V3 model
        df_derived = df.copy()
        df_derived['outstanding_growth'] = df_derived['t2_outstanding'] - df_derived['t1_outstanding']
        df_derived['days_late_trend'] = df_derived['t2_days_late'] - df_derived['t1_days_late']
        df_derived['both_terms_late'] = ((df_derived['t1_status'] >= 1) & (df_derived['t2_status'] >= 1)).astype(int)
        df_derived['escalating'] = (df_derived['t2_status'] > df_derived['t1_status']).astype(int)
        df_derived['avg_outstanding'] = (df_derived['t1_outstanding'] + df_derived['t2_outstanding']) / 2
        df_derived['max_days_late'] = df_derived[['t1_days_late', 't2_days_late']].max(axis=1)
        
        X = df_derived[models['fee_feature_cols']].values
        probs = models['fee'].predict_proba(X)[:, 1]
    else:
        # V2 model
        X = df.drop(['student_id', 'label'], axis=1).values
        proba_all = models['fee'].predict_proba(X)
        probs = proba_all[:, 2] if proba_all.shape[1] > 2 else proba_all[:, 1]
    
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append({
            "student_id": row['student_id'],
            "default_probability": round(float(probs[i] * 100), 2),
            "risk_category": "High" if probs[i] > fee_thresh + 0.2 else "Medium" if probs[i] > fee_thresh else "Low",
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

    # Top highest-risk students (combined attendance + fee risk)
    def combined_risk(s):
        return s['risk_score'] * 0.5 + s['fee_prob'] * 0.5
    top_risk = sorted(students_cache, key=combined_risk, reverse=True)
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
            "fee_model":        models.get('fee_model_version', 'unknown'),
            "fee_recall":       85 if 'V3' in models.get('fee_model_version', '') else 70,
            "total_features":   15 if 'V3' in models.get('fee_model_version', '') else 10,
            "training_records": 2000 if 'V3' in models.get('fee_model_version', '') else 500,
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


@app.get("/api/debug")
def debug_info():
    return {
        "status": "running",
        "cache_len": len(students_cache),
        "startup_error": startup_error_msg,
        "models_keys": list(models.keys()),
        "fee_model_version": models.get('fee_model_version', 'unknown'),
        "root_dir": ROOT,
        "exists": {
            "model_att": os.path.exists(os.path.join(ROOT, 'models/attendance_anomaly/model.pkl')),
            "scaler": os.path.exists(os.path.join(ROOT, 'models/attendance_anomaly/scaler.pkl')),
            "model_fee_v3": os.path.exists(os.path.join(ROOT, 'models/fee_predictor/model_v3.pkl')),
            "model_fee_v2": os.path.exists(os.path.join(ROOT, 'models/fee_predictor/model.pkl')),
            "att_features": os.path.exists(os.path.join(ROOT, 'data/attendance_features.csv')),
            "fee_features_v3": os.path.exists(os.path.join(ROOT, 'data/fee_features_v3.csv')),
            "fee_features_v2": os.path.exists(os.path.join(ROOT, 'data/fee_features.csv')),
        }
    }


app.mount("/static", StaticFiles(directory=TEMPLATES), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
