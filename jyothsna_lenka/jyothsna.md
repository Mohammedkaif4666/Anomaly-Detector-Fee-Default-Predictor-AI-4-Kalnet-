# Jyothsna Lenka — API Documentation

> **Role:** Backend API Developer — KALNET AI-4 Team  
> **File:** `jyothsna_lenka/main.py`  
> **Framework:** FastAPI (Python)  
> **Coordination:** Connects AI models (Om & Are) to the frontend dashboard (templates/)

---

## Overview

The FastAPI application is the **central nervous system** of the KALNET AI-4 project. It:
1. Loads trained ML models at startup (never inside route handlers)
2. Serves all student risk data to the frontend via REST API
3. Hosts the frontend dashboard as a static file at the root URL
4. Exposes original POST endpoints for programmatic ML inference

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | **FastAPI** (Python) |
| ASGI Server | **Uvicorn** |
| ML Models | **Joblib** (model loading) |
| Data Processing | **Pandas, NumPy** |
| Frontend Serving | `fastapi.staticfiles.StaticFiles` + `FileResponse` |
| Communication | **REST API (HTTP/JSON)** — fetch() from browser |

---

## How Frontend & Backend Connect

```
Browser (templates/index.html)
        │
        │  fetch('/api/summary')         ← GET on page load
        │  fetch('/api/students?...')    ← GET for student grid
        │  fetch('/api/student/STU_001') ← GET on card click
        │
        ▼
FastAPI Server (jyothsna_lenka/main.py)  ← uvicorn port 8000
        │
        │  Reads pre-computed student cache (built at startup)
        │  Loads attendance_features.csv, fee_features.csv
        │  Runs IsolationForest + GradientBoosting predictions
        │
        ▼
Returns JSON response → JavaScript renders cards/modal/charts
```

The frontend uses the **Fetch API** (native browser JavaScript) to call these endpoints. No external HTTP client library is needed. All calls are to the **same origin** (`http://localhost:8000`), so no CORS issues.

---

## API Endpoints

### `GET /`
Serves the frontend dashboard (`templates/index.html`).

**Response:** HTML page

---

### `GET /api/summary`
Returns aggregate statistics for the dashboard header cards.

**Response:**
```json
{
  "total": 500,
  "anomalies": 50,
  "fee_defaults": 22,
  "fee_late": 92
}
```

**Frontend usage:**
```js
const r = await fetch('/api/summary');
const d = await r.json();
// d.anomalies → shown in red stat card
```

---

### `GET /api/students`
Returns a list of all students with their risk profile. Supports search and filter.

**Query Parameters:**
| Param | Type | Example | Description |
|-------|------|---------|-------------|
| `search` | string | `rahul` | Filter by name or student ID |
| `filter` | string | `anomaly` | One of: `all`, `anomaly`, `fee_risk`, `safe` |

**Example Request:**
```
GET /api/students?search=sharma&filter=anomaly
```

**Response (array):**
```json
[
  {
    "id": "STU_001",
    "name": "Aarav Sharma",
    "attendance_rate": 34.5,
    "is_anomaly": true,
    "risk_score": 88.2,
    "absence_streak": 21,
    "absence_30d": 18,
    "fee_label": 0,
    "fee_prob": 3.1,
    "outstanding": 0.0,
    "days_late": 0
  }
]
```

**Frontend usage:**
```js
const r = await fetch(`/api/students?search=${query}&filter=${activeFilter}`);
const students = await r.json();
// Rendered as student cards in the grid
```

---

### `GET /api/student/{student_id}`
Returns full detail for a single student, including last 30-day attendance history for the chart.

**Path Parameter:** `student_id` — e.g., `STU_045`

**Response:**
```json
{
  "id": "STU_045",
  "name": "Rahul Gupta",
  "attendance_rate": 34.5,
  "is_anomaly": true,
  "risk_score": 91.3,
  "absence_streak": 21,
  "absence_30d": 18,
  "fee_label": 2,
  "fee_prob": 97.4,
  "outstanding": 4200.0,
  "days_late": 87,
  "history": [
    {"date": "2024-01-15", "is_present": 1},
    {"date": "2024-01-16", "is_present": 0}
  ]
}
```

**Frontend usage:**
```js
const r = await fetch(`/api/student/${studentId}`);
const s = await r.json();
// s.history → rendered as Chart.js bar chart
// s.is_anomaly, s.risk_score → shown in modal with reason text
```

---

### `POST /ai/anomalies`
Original endpoint — accepts a list of student IDs, runs IsolationForest, returns risk scores.

**Request Body:**
```json
{
  "student_ids": ["STU_001", "STU_045", "STU_123"]
}
```

**Response:**
```json
{
  "results": [
    {
      "student_id": "STU_001",
      "risk_score": 88.2,
      "is_flagged": true,
      "risk_level": "High"
    }
  ]
}
```

**Test via Swagger:** `http://localhost:8000/docs` → `/ai/anomalies` → Try it out

---

### `POST /ai/fee-risk`
Original endpoint — accepts a list of student IDs, runs GradientBoosting, returns default probabilities.

**Request Body:**
```json
{
  "student_ids": ["STU_001", "STU_045"]
}
```

**Response:**
```json
{
  "results": [
    {
      "student_id": "STU_045",
      "default_probability": 0.97,
      "risk_category": "High"
    }
  ]
}
```

---

## Model Loading Strategy

As per requirements, **models are loaded once at startup**, never inside route handlers:

```python
@app.on_event("startup")
def startup():
    models['att']    = joblib.load('models/attendance_anomaly/model.pkl')
    models['scaler'] = joblib.load('models/attendance_anomaly/scaler.pkl')
    models['fee']    = joblib.load('models/fee_predictor/model.pkl')
    # Pre-computes ALL 500 student risk scores → stored in students_cache[]
    # Dashboard loads instantly — no inference on every request
```

The `students_cache` list is an in-memory store of pre-computed predictions for all 500 students. This ensures the UI loads with **zero latency** when scrolling through student cards.

---

## How to Test the API

1. Start the server:
   ```bash
   uvicorn jyothsna_lenka.main:app --reload
   ```

2. Open Swagger UI: `http://localhost:8000/docs`

3. Open the dashboard: `http://localhost:8000`

4. Example curl test:
   ```bash
   curl -X POST http://localhost:8000/ai/anomalies \
     -H "Content-Type: application/json" \
     -d '{"student_ids": ["STU_001", "STU_070"]}'
   ```
