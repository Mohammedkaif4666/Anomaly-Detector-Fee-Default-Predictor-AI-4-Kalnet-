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


# ── Global state ───────────────────────────────────────────────
models         = {}
features_db    = {}
students_cache = []
startup_error_msg = "No error"

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(ROOT, "templates")

NAMES = [
    "Aarav Sharma","Priya Patel","Rahul Gupta","Anjali Singh","Vikram Reddy",
    "Sanya Iyer","Arjun Nair","Meera Joshi","Rohan Verma","Kavya Menon",
    "Dev Malhotra","Isha Rao","Aditya Kumar","Pooja Bhat","Karan Mishra",
    "Sneha Das","Harsh Agarwal","Riya Thakur","Nikhil Pandey","Divya Chauhan",
]


def _compute_v3_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived/trend features for V3 inference. Must match train_v3.py exactly.
    Input needs: t1_status, t1_outstanding, t1_days_late,
                 t2_status, t2_outstanding, t2_days_late
    """
    d = df.copy()
    d["outstanding_growth"] = d["t2_outstanding"] - d["t1_outstanding"]
    d["days_late_trend"] = d["t2_days_late"] - d["t1_days_late"]
    d["both_terms_late"] = ((d["t1_status"] >= 1) & (d["t2_status"] >= 1)).astype(int)
    d["escalating"] = (d["t2_status"] > d["t1_status"]).astype(int)
    d["avg_outstanding"] = (d["t1_outstanding"] + d["t2_outstanding"]) / 2
    d["max_days_late"] = d[["t1_days_late", "t2_days_late"]].max(axis=1)
    return d


def _load_models():
    """Load the attendance IsolationForest and the V3 fee model bundle."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    att_model_path  = os.path.join(ROOT, "models/attendance_anomaly/model.pkl")
    att_scaler_path = os.path.join(ROOT, "models/attendance_anomaly/scaler.pkl")
    fee_v3_path     = os.path.join(ROOT, "models/fee_predictor/model_v3.pkl")

    # ── Attendance model ──────────────────────────────────────
    try:
        models["att"]    = joblib.load(att_model_path)
        models["scaler"] = joblib.load(att_scaler_path)
        print("[OK] Attendance model loaded from .pkl")
    except Exception as e:
        print(f"[WARN] Attendance pkl failed ({e}), retraining...")
        att_df = pd.read_csv(os.path.join(ROOT, "data/attendance_features.csv"))
        X_a    = att_df.drop(["student_id", "is_anomaly"], axis=1).values
        scaler = StandardScaler()
        X_a_s  = scaler.fit_transform(X_a)
        iso    = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
        iso.fit(X_a_s)
        models["att"]    = iso
        models["scaler"] = scaler
        os.makedirs(os.path.dirname(att_model_path), exist_ok=True)
        try:
            joblib.dump(iso,    att_model_path)
            joblib.dump(scaler, att_scaler_path)
        except Exception:
            pass
        print("[OK] Attendance model retrained")

    # ── Fee model: V3 bundle (required) ───────────────────────
    if not os.path.exists(fee_v3_path):
        raise FileNotFoundError(
            f"model_v3.pkl not found at {fee_v3_path} — run train_v3.py first"
        )

    bundle = joblib.load(fee_v3_path)
    required_keys = ["model", "feature_cols", "threshold",
                      "training_median", "cv_auc_mean", "cv_ap_mean"]
    missing = [k for k in required_keys if k not in bundle]
    if missing:
        raise KeyError(f"V3 bundle missing keys: {missing}")

    models["fee"]              = bundle["model"]
    models["fee_thresh"]       = bundle["threshold"]
    models["fee_feature_cols"] = bundle["feature_cols"]
    models["fee_cv_auc"]       = bundle["cv_auc_mean"]
    models["fee_cv_ap"]        = bundle["cv_ap_mean"]
    models["fee_train_median"] = bundle["training_median"]
    print(f"[OK] Fee model V3 loaded | threshold={bundle['threshold']:.3f} "
          f"| cv_auc={bundle['cv_auc_mean']:.4f}")


# ══════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════
@app.on_event("startup")
def startup():
    global students_cache
    try:
        _load_models()

        att = pd.read_csv(
            os.path.join(ROOT, "data/attendance_features.csv")
        ).reset_index(drop=True)
        fee = pd.read_csv(
            os.path.join(ROOT, "data/fee_features_v3.csv")
        ).reset_index(drop=True)

        features_db["att"] = att
        features_db["fee"] = fee
        print(f"[INFO] Fee CSV: {len(fee)} rows")

        # ── Attendance predictions ────────────────────────────
        X_att    = att.drop(["student_id", "is_anomaly"], axis=1).values
        X_scaled = models["scaler"].transform(X_att)
        att_preds  = models["att"].predict(X_scaled)
        att_scores = np.clip(
            100 * (1 - (models["att"].decision_function(X_scaled) + 0.5) / 1.0),
            0, 100
        )

        # ── Fee predictions (V3: binary, 0=safe / 2=default) ──
        fee_thresh  = models["fee_thresh"]
        fee_derived = _compute_v3_derived(fee)
        X_fee       = fee_derived[models["fee_feature_cols"]].values
        fee_probs   = models["fee"].predict_proba(X_fee)[:, 1]

        fee_map = {}
        for i in range(len(fee)):
            sid = fee.loc[i, "student_id"]
            fee_map[sid] = {
                "prob"       : float(fee_probs[i]),
                "label"      : 2 if fee_probs[i] >= fee_thresh else 0,
                "outstanding": float(fee.loc[i, "t2_outstanding"]),
                "days_late"  : int(fee.loc[i, "t2_days_late"]),
            }

        # ── Class map (optional labels CSV) ──────────────────
        labels_path = os.path.join(ROOT, "data/student_labels.csv")
        class_map   = {}
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            if "class" in labels_df.columns:
                class_map = dict(zip(labels_df["student_id"], labels_df["class"]))

        # ── Build student cache ───────────────────────────────
        classes = ["6A","6B","7A","7B","8A","8B","9A","9B","10A","10B"]
        for i, row in att.iterrows():
            sid = row["student_id"]
            num = int(sid.split("_")[1]) - 1
            fd  = fee_map.get(sid, {
                "prob": 0, "label": 0, "outstanding": 0, "days_late": 0
            })
            students_cache.append({
                "id"            : sid,
                "name"          : NAMES[num % len(NAMES)],
                "student_class" : class_map.get(sid, classes[num % len(classes)]),
                "attendance_rate": round(float(row["attendance_rate"]) * 100, 1),
                "is_anomaly"    : bool(att_preds[i] == -1),
                "risk_score"    : round(float(att_scores[i]), 1),
                "absence_streak": int(row["longest_absence_streak"]),
                "absence_30d"   : int(row["absence_in_last_30_days"]),
                "fee_label"     : fd["label"],
                "fee_prob"      : round(fd["prob"] * 100, 1),
                "outstanding"   : fd["outstanding"],
                "days_late"     : fd["days_late"],
            })

        print(f"[OK] KALNET AI-4 ready — {len(students_cache)} students loaded")
        print(f"[INFO] Fee threshold     : {fee_thresh:.3f}")

    except Exception as e:
        global startup_error_msg
        startup_error_msg = str(e)
        print(f"[ERROR] Startup failed: {e}")
        import traceback
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# DASHBOARD SUMMARY
# ══════════════════════════════════════════════════════════════
@app.get("/api/summary")
def summary():
    """fee_default = predicted at-risk (>= threshold), not historical label count."""
    return {
        "total"       : len(students_cache),
        "anomalies"   : sum(1 for s in students_cache if s["is_anomaly"]),
        "fee_defaults": sum(1 for s in students_cache if s["fee_label"] == 2),
    }


# ══════════════════════════════════════════════════════════════
# STUDENT LIST
# ══════════════════════════════════════════════════════════════
@app.get("/api/students")
def list_students(search: str = "", filter: str = "all"):
    """
    filter: all | anomaly | fee_risk (fee_label==2) | safe (no anomaly, no default)
    """
    data = students_cache
    if search:
        q    = search.lower()
        data = [s for s in data
                if q in s["name"].lower() or q in s["id"].lower()]
    if filter == "anomaly":
        data = [s for s in data if s["is_anomaly"]]
    elif filter == "fee_risk":
        data = [s for s in data if s["fee_label"] == 2]
    elif filter == "safe":
        data = [s for s in data if not s["is_anomaly"] and s["fee_label"] == 0]
    return data


# ══════════════════════════════════════════════════════════════
# INDIVIDUAL STUDENT
# ══════════════════════════════════════════════════════════════
@app.get("/api/student/{student_id}")
def get_student(student_id: str):
    s = next((x for x in students_cache if x["id"] == student_id), None)
    if not s:
        raise HTTPException(404, "Student not found")
    raw  = pd.read_csv(os.path.join(ROOT, "data/attendance_raw.csv"))
    hist = raw[raw["student_id"] == student_id].tail(30)
    result = dict(s)
    result["history"] = hist[["date", "is_present"]].to_dict("records")
    return result


# ══════════════════════════════════════════════════════════════
# POST — BATCH ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════
class StudentRequest(BaseModel):
    student_ids: List[str]


@app.post("/ai/anomalies")
def get_anomalies(req: StudentRequest):
    df = features_db["att"][features_db["att"]["student_id"].isin(req.student_ids)]
    if df.empty:
        return {"results": []}
    X      = df.drop(["student_id", "is_anomaly"], axis=1).values
    Xs     = models["scaler"].transform(X)
    preds  = models["att"].predict(Xs)
    scores = np.clip(
        100 * (1 - (models["att"].decision_function(Xs) + 0.5) / 1.0), 0, 100
    )
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append({
            "student_id": row["student_id"],
            "risk_score": round(float(scores[i]), 2),
            "is_flagged": bool(preds[i] == -1),
            "risk_level": ("High"   if scores[i] > 70
                           else "Medium" if scores[i] > 40
                           else "Low"),
        })
    return {"results": results}


# ══════════════════════════════════════════════════════════════
# POST — BATCH FEE RISK
# ══════════════════════════════════════════════════════════════
@app.post("/ai/fee-risk")
def get_fee_risk(req: StudentRequest):
    """
    risk_category thresholds:
      prob >= threshold + 0.20  → HIGH
      prob >= threshold          → MEDIUM
      prob < threshold           → LOW
    """
    df = features_db["fee"][
        features_db["fee"]["student_id"].isin(req.student_ids)
    ].copy()

    if df.empty:
        return {"results": []}

    fee_thresh = models["fee_thresh"]
    df_derived = _compute_v3_derived(df)
    X          = df_derived[models["fee_feature_cols"]].values
    probs      = models["fee"].predict_proba(X)[:, 1]

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        p = float(probs[i])
        results.append({
            "student_id"         : row["student_id"],
            "default_probability": round(p * 100, 2),
            "risk_category"      : ("HIGH"   if p >= fee_thresh + 0.20
                                    else "MEDIUM" if p >= fee_thresh
                                    else "LOW"),
        })
    return {"results": results}


# ══════════════════════════════════════════════════════════════
# ANALYTICS (Admin Dashboard)
# ══════════════════════════════════════════════════════════════
@app.get("/api/analytics")
def analytics():
    """All data comes from the pre-computed students_cache for speed."""
    if not students_cache:
        return {}

    total   = len(students_cache)
    high    = sum(1 for s in students_cache if s["risk_score"] > 70)
    medium  = sum(1 for s in students_cache if 40 < s["risk_score"] <= 70)
    low     = total - high - medium
    on_time = sum(1 for s in students_cache if s["fee_label"] == 0)
    default = sum(1 for s in students_cache if s["fee_label"] == 2)

    # Attendance rate histogram
    buckets = {
        "0-50%": 0, "50-60%": 0, "60-70%": 0,
        "70-80%": 0, "80-90%": 0, "90-100%": 0
    }
    for s in students_cache:
        r = s["attendance_rate"]
        if   r < 50: buckets["0-50%"]   += 1
        elif r < 60: buckets["50-60%"]  += 1
        elif r < 70: buckets["60-70%"]  += 1
        elif r < 80: buckets["70-80%"]  += 1
        elif r < 90: buckets["80-90%"]  += 1
        else:        buckets["90-100%"] += 1

    attendance_histogram = [{"bucket": k, "count": v} for k, v in buckets.items()]

    # Class-wise breakdown
    class_stats = {}
    for s in students_cache:
        cls = s.get("student_class", "Unknown")
        if cls not in class_stats:
            class_stats[cls] = {
                "class": cls, "total": 0,
                "anomalies": 0, "defaults": 0,
            }
        class_stats[cls]["total"] += 1
        if s["is_anomaly"]:     class_stats[cls]["anomalies"] += 1
        if s["fee_label"] == 2: class_stats[cls]["defaults"]  += 1
    class_breakdown = sorted(class_stats.values(), key=lambda x: x["class"])

    # Top risk students (combined attendance + fee score)
    def combined_risk(s):
        return s["risk_score"] * 0.5 + s["fee_prob"] * 0.5

    top_risk_students = [
        {
            "id"            : s["id"],
            "name"          : s["name"],
            "student_class" : s.get("student_class", "—"),
            "attendance_rate": s["attendance_rate"],
            "risk_score"    : s["risk_score"],
            "fee_prob"      : s["fee_prob"],
            "is_anomaly"    : s["is_anomaly"],
            "fee_label"     : s["fee_label"],
        }
        for s in sorted(students_cache, key=combined_risk, reverse=True)
    ]

    # Recent flagged activity (last 8)
    flagged = [s for s in students_cache if s["is_anomaly"] or s["fee_label"] == 2]
    recent_activity = [
        {
            "id"    : s["id"],
            "name"  : s["name"],
            "type"  : "Attendance Anomaly" if s["is_anomaly"] else "Fee Default",
            "detail": (f"Risk {s['risk_score']}" if s["is_anomaly"]
                       else f"{s['fee_prob']}% default prob"),
        }
        for s in flagged[:8]
    ]

    # System health score (0-100, higher = better)
    health = round(100 - (high / total * 40) - (default / total * 60), 1)

    return {
        "total_students"     : total,
        "anomaly_count"      : sum(1 for s in students_cache if s["is_anomaly"]),
        "fee_default_count"  : default,
        "risk_distribution"  : {"High": high, "Medium": medium, "Low": low},
        "fee_distribution"   : {"On Time": on_time, "Default": default},
        "attendance_histogram": attendance_histogram,
        "class_breakdown"    : class_breakdown,
        "top_risk_students"  : top_risk_students,
        "recent_activity"    : recent_activity,
        "system_health"      : max(0, health),
        "model_metrics": {
            "anomaly_model"    : "IsolationForest",
            "anomaly_recall"   : 63,
            "fee_model"        : "GradientBoostingClassifier (V3)",
            "fee_cv_auc"       : round(models["fee_cv_auc"], 4),
            "fee_cv_ap"        : round(models["fee_cv_ap"], 4),
            "total_features"   : len(models["fee_feature_cols"]),
            "training_records" : len(features_db["fee"]),
            "fee_threshold"    : round(float(models["fee_thresh"]), 3),
            "v3_note"          : "Binary model: ontime vs default.",
        },
    }


# ══════════════════════════════════════════════════════════════
# DEBUG ENDPOINT
# ══════════════════════════════════════════════════════════════
@app.get("/api/debug")
def debug_info():
    return {
        "status"        : "running",
        "cache_len"     : len(students_cache),
        "startup_error" : startup_error_msg,
        "models_keys"   : list(models.keys()),
        "fee_threshold" : models.get("fee_thresh"),
        "fee_cv_auc"    : models.get("fee_cv_auc"),
        "fee_cv_ap"     : models.get("fee_cv_ap"),
        "root_dir"      : ROOT,
        "exists": {
            "model_att"      : os.path.exists(os.path.join(ROOT, "models/attendance_anomaly/model.pkl")),
            "scaler"         : os.path.exists(os.path.join(ROOT, "models/attendance_anomaly/scaler.pkl")),
            "model_fee_v3"   : os.path.exists(os.path.join(ROOT, "models/fee_predictor/model_v3.pkl")),
            "att_features"   : os.path.exists(os.path.join(ROOT, "data/attendance_features.csv")),
            "fee_features_v3": os.path.exists(os.path.join(ROOT, "data/fee_features_v3.csv")),
        },
    }


# ══════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return FileResponse(os.path.join(TEMPLATES, "landing.html"))

@app.get("/students")
def student_directory():
    return FileResponse(os.path.join(TEMPLATES, "index.html"))

@app.get("/dashboard")
def admin_dashboard(username: str = Depends(verify_admin)):
    return FileResponse(os.path.join(TEMPLATES, "dashboard.html"))

app.mount("/static", StaticFiles(directory=TEMPLATES), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)