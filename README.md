# KALNET AI-4 — Anomaly Detector & Fee Default Predictor

> **Machine learning in real schools. Scikit-learn only. No paid APIs.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)](https://scikit-learn.org)

> 📋 **See [`UPDATES.md`](UPDATES.md)** for a full log of all bug fixes and improvements made during development.

---

## 🎯 Project Vision

The admin opens KALNET on Friday. A **red section** appears: *3 students flagged as high attendance risk this week.*

She clicks. **Rahul Sharma, Class 9A** — attendance dropped from 92% to 34% in 3 weeks. Flagged as anomalous. The admin calls his parents. Family issue discovered. **Without this system, no one would have noticed for another month.**

The finance section shows **5 students predicted to miss next term's payment.** The team reaches out early. One default is prevented.

This is machine learning in real schools.

---

## 👥 Team — KALNET AI-4

| Member | Role | Files |
|--------|------|-------|
| **Mohammed Kaif** | Data Generation & Feature Engineering | `mohammed_kaif/` |
| **Rohith Koppu** | Feature Verification & Quality Control | `rohith_koppu/` |
| **Om Dattatray Wabale** | Attendance Anomaly Model | `om_dattatray_wabale/train.py` |
| **Are Samhith** | Fee Default Prediction Model | `are_samhith/train.py` |
| **Jyothsna Lenka** | FastAPI Backend & API Design | `jyothsna_lenka/main.py` |
| **Vodyati Sai Phanindra** | Model Evaluation & Documentation | `vodyati_sai_phanindra/evaluate.py` |

---

## 🏗️ Project Architecture

```
kalnet AI-4 team project/
│
├── mohammed_kaif/
│   ├── data_generator.py        # Generates 72,000 attendance records + 1,500 fee records
│   └── feature_engineering.py  # Computes per-student ML features
│
├── rohith_koppu/                # Feature verification & data quality checks
│
├── om_dattatray_wabale/
│   └── train.py                 # IsolationForest — attendance anomaly detection
│
├── are_samhith/
│   └── train.py                 # GradientBoostingClassifier — fee default prediction
│
├── jyothsna_lenka/
│   ├── main.py                  # FastAPI server — backend + API + UI serving
│   └── jyothsna.md              # Full API documentation (read this for API details)
│
├── vodyati_sai_phanindra/
│   └── evaluate.py              # Evaluation metrics, reports, and model verification
│
├── templates/
│   ├── index.html               # Frontend dashboard — professional table layout
│   └── style.css                # Separate CSS for clean table design
│
├── data/                        # Generated CSVs (auto-created on first run)
│   ├── attendance_raw.csv       # 72,000 rows: student_id, date, is_present
│   ├── attendance_features.csv  # 500 rows: per-student ML features
│   ├── fee_raw.csv              # 1,500 rows: 500 students × 3 terms
│   ├── fee_features.csv         # 500 rows: per-student fee features
│   └── student_labels.csv       # Ground truth labels for evaluation
│
├── models/                      # Trained model files (auto-created after training)
│   ├── attendance_anomaly/
│   │   ├── model.pkl            # Trained IsolationForest
│   │   └── scaler.pkl           # Fitted StandardScaler
│   └── fee_predictor/
│       ├── model.pkl            # Trained GradientBoostingClassifier
│       └── threshold.pkl        # Tuned decision threshold for ≥70% default recall
│
├── docs/
│   └── model_evaluation.md     # Auto-generated evaluation report
│
├── requirements.txt
└── README.md
```

---

## 🗄️ Dataset Description

### Attendance Data (`attendance_raw.csv`)
- **Shape:** ~72,000 rows × 3 columns
- **500 students × ~144 school days** (weekdays only from 200 calendar days)
- **86% of students** have normal attendance (80–97% rate)
- **14% of students** are anomalous — sudden drop to 20–40% in final weeks

| Column | Type | Description |
|--------|------|-------------|
| `student_id` | string | e.g., `STU_001` to `STU_500` |
| `date` | string | YYYY-MM-DD format |
| `is_present` | int | 1 = Present, 0 = Absent |

### Fee Data (`fee_raw.csv`)
- **Shape:** 1,500 rows × 8 columns (500 students × 3 terms)
- **80%** pay on time, **15%** pay late, **5%** default

| Column | Type | Description |
|--------|------|-------------|
| `student_id` | string | Student identifier |
| `term` | int | 1, 2, or 3 |
| `status` | int | 0=On Time, 1=Late, 2=Default |
| `days_since_last_payment` | int | Days overdue |
| `family_income_bracket` | string | Low / Medium / High |
| `transport_user` | int | 1 if uses school transport |
| `sibling_count` | int | Number of siblings (0–3) |
| `total_outstanding` | float | Unpaid amount in ₹ |

### Engineered Features (`attendance_features.csv`)
Per-student summary computed from raw daily data:

| Feature | How Computed |
|---------|-------------|
| `attendance_rate` | Total present ÷ total days |
| `longest_absence_streak` | Max consecutive absent days |
| `absence_in_last_30_days` | Count of absences in last 30 records |
| `day_of_week_variance` | Variance of attendance rate per weekday |
| `is_anomaly` | Ground-truth label (1 = anomalous student) |

### Engineered Features (`fee_features.csv`)
| Feature | Encoding |
|---------|----------|
| `days_since_last_payment` | Raw integer |
| `previous_term_status` | 0/1/2 (best of earlier terms) |
| `total_outstanding` | Raw ₹ amount |
| `income_encoded` | H=0, M=1, L=2 |
| `transport_user` | 0 or 1 |
| `sibling_count` | 0–3 |
| `label` | 0=On Time, 1=Late, 2=Default |

---

## 🤖 Machine Learning Models

### Model 1 — Attendance Anomaly Detection (Om Dattatray Wabale)
```python
IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
```

| Detail | Value |
|--------|-------|
| Algorithm | IsolationForest (unsupervised) |
| Input | 4 attendance features, StandardScaler normalized |
| Output | Anomaly flag (-1/1) + risk score (0–100) |
| Recall Target | ≥ 60% |
| Achieved Recall | **~63%** ✅ |

**How it works:** IsolationForest isolates observations by randomly partitioning features. Anomalies are isolated faster (shorter paths in the tree), resulting in a lower decision score. Scores are normalized to a 0–100 risk scale where 100 = highest risk.

### Model 2 — Fee Default Prediction (Are Samhith)
```python
GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42)
```

| Detail | Value |
|--------|-------|
| Algorithm | Gradient Boosting (supervised, multi-class) |
| Input | 6 fee features from **past terms only** (no leakage) |
| Output | Class probabilities for On Time / Late / Default |
| Split | 80/20 stratified (critical for minority default class) |
| Sample Weights | 4× for default class to boost recall |
| Threshold Tuning | Auto-tuned threshold saved to `threshold.pkl` |
| Recall Target | ≥ 70% for Default class |
| Achieved Recall | **≥ 70%** ✅ |

**Key Design:** Features use only historical (past terms 1 & 2) data to predict term 3 outcome — prevents data leakage that would cause artificially perfect but useless predictions.

**Top Predictive Features:**
1. `total_outstanding` — average past outstanding amount
2. `days_since_last_payment` — average past days overdue
3. `income_encoded` — Low income families at higher risk

---

## 🌐 Technology Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| **Python 3.9+** | Core language |
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server (runs FastAPI) |
| **Scikit-learn** | IsolationForest, GradientBoosting, StandardScaler |
| **Joblib** | Save/load trained model files (.pkl) |
| **Pandas** | Data loading and processing |
| **NumPy** | Numerical computations |

### Frontend
| Technology | Purpose |
|-----------|---------|
| **HTML5** | Page structure |
| **CSS3** | White-theme styling, animations, responsive grid |
| **Vanilla JavaScript** | Fetch API calls, DOM manipulation |
| **Chart.js (CDN)** | 30-day attendance bar chart |
| **Google Fonts (Inter)** | Typography |

### How Frontend Connects to Backend
The frontend (`templates/index.html`) is served directly by FastAPI at `http://localhost:8000`. All API calls use the browser's native `fetch()` function to call REST endpoints on the **same origin**:

```javascript
// Page load — fetch summary stats
const summary = await fetch('/api/summary').then(r => r.json());

// Student card click — fetch full student data
const student = await fetch(`/api/student/STU_045`).then(r => r.json());
```

No separate frontend server is needed. **One command starts everything.**

---

## 🚀 How to Run the Project (Step by Step)

### Prerequisites
- Python 3.9 or higher installed
- pip package manager

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Mohammedkaif4666/Anomaly-Detector-Fee-Default-Predictor-AI-4-Kalnet-.git
cd "Anomaly-Detector-Fee-Default-Predictor-AI-4-Kalnet-"
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Generate Data (Run Once)
```bash
python mohammed_kaif/data_generator.py
```
> Generates `data/attendance_raw.csv` (~72,000 rows) and `data/fee_raw.csv` (1,500 rows)

### Step 4 — Engineer Features (Run Once)
```bash
python mohammed_kaif/feature_engineering.py
```
> Generates `data/attendance_features.csv` and `data/fee_features.csv`

### Step 5 — Verify Data Quality (Rohith Koppu)
```bash
python rohith_koppu/verify.py
```
> Runs intuition checks on engineered features — NaN check, value ranges, label distribution, intuition checks (e.g., anomalous students should have lower attendance). All checks must pass before training.

### Step 6 — Train Models (Run Once)
```bash
python om_dattatray_wabale/train.py
python are_samhith/train.py
```
> Saves trained models to `models/attendance_anomaly/` and `models/fee_predictor/`

### Step 7 — Run Evaluation (Optional but recommended)
```bash
python vodyati_sai_phanindra/evaluate.py
```
> Prints metrics to console and saves `docs/model_evaluation.md`

### Step 8 — Start the Application
```bash
uvicorn jyothsna_lenka.main:app --reload
```

### Step 9 — Open the Dashboard
- **Landing Page (Public):** `http://localhost:8000`
- **Student Directory (Public):** `http://localhost:8000/students`
- **Admin Analytics (Secured):** `http://localhost:8000/dashboard`
  - **Username:** `admin` *(Teammate access)*
  - **Password:** `kalnet2026` *(Teammate access)*
- **Interactive API Docs:** `http://localhost:8000/docs`
- **API JSON (direct):** `http://localhost:8000/api/summary`

> **Note:** Steps 3–7 only need to be run **once**. After that, only Step 8 is needed to start the app.

---

## 🖥️ Platform Interfaces (CEO Review & Admin Enhancements)

> *Following feedback from the CEO, the platform was overhauled to be market-ready. Further enhancements to the Admin page have given administrators full control and interactive inspection capabilities directly from the Analytics view.*

| Interface | Description |
|-----------|-------------|
| **Marketing Landing Page** | High-conversion entry point with live ML statistics, feature overviews, and team details. |
| **Admin Analytics Dashboard** | Secure dashboard with real-time KPIs, system health gauge, model metrics, and active notifications. |
| **Interactive Chart Segments** | All Chart.js segments (risk levels, fee default status, attendance buckets, class sections) are clickable. Opens a popup modal containing a real, live student list matching that segment. |
| **Double Modal Layering** | Clicking a student in any popup list inside the admin page opens their full student details card (30-day attendance chart, AI reason explanations) directly on top of the list modal, without context switching. |
| **Dynamic Top N Table** | Dynamic row limits (Top 10, 20, 30, 50, all) for highest-risk students can be adjusted on the fly. |
| **Autocomplete Student Lookup** | Global search input in the top navigation bar for immediate student lookups from anywhere on the admin page. |
| **Student Directory Table** | Clean rows with pagination, sorting, search, and filter tabs at `/students`. |
| **Unified Navigation** | Persistent `global-nav` that snaps to a bottom bar on mobile screens. |
| **Admin Security** | Secure `/dashboard` protected by Basic Authentication (`admin` / `kalnet2026`). Configurable via `REQUIRE_ADMIN_AUTH` in `main.py`. |

---

## 📖 API Reference

For complete API documentation including request/response examples and connection diagrams, see:

📄 [`jyothsna_lenka/jyothsna.md`](jyothsna_lenka/jyothsna.md)

Quick reference:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve public marketing landing page |
| `/students` | GET | Serve public student directory dashboard |
| `/dashboard` | GET | Serve secure admin analytics dashboard (Auth required) |
| `/api/summary` | GET | Aggregate stats (totals, anomaly counts) |
| `/api/students` | GET | Student list with search & filter |
| `/api/student/{id}` | GET | Full student detail + 30-day history |
| `/api/analytics` | GET | Chart-ready JSON data for the Admin Dashboard |
| `/ai/anomalies` | POST | Run IsolationForest on given student IDs |
| `/ai/fee-risk` | POST | Run GradientBoosting on given student IDs |
| `/docs` | GET | Interactive Swagger API documentation |

---

## 📊 Model Evaluation Results

| Model | Metric | Target | Result |
|-------|--------|--------|--------|
| IsolationForest | Recall (Anomaly) | ≥ 60% | **63% ✅** |
| GradientBoosting | Recall (Default) | ≥ 70% | **≥ 70% ✅** |

Full report: [`docs/model_evaluation.md`](docs/model_evaluation.md)

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
joblib
fastapi
uvicorn
```

Install: `pip install -r requirements.txt`
