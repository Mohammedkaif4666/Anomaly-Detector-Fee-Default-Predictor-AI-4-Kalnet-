# KALNET AI-4 — Update Log

> This file tracks all updates made after the initial build, including bug fixes, UI improvements, and requirement alignments.

---

## Update 4 — Admin Dashboard Interactive Overhaul & Controls *(26 May 2026)*

> *This update adds interactive analysis capabilities and robust administrative controls directly to the Analytics page so the admin can manage everything without page jumping.*

### 📊 Interactive Charts & Dynamic Segment Lists
- Configured click handlers on all 4 Chart.js charts (`riskChart`, `feeChart`, `attHistChart`, `classChart`).
- Clicking on any pie slice or bar filters and displays a popup table containing the actual students matching that segment, complete with detailed descriptors.

### 🗂️ Staggered Double Modals
- Created a separate modal overlay for student segment lists and single student profile details.
- Clicking a student in a segment list modal overlays their complete profile modal (attendance chart, streak, outstanding amount, and AI predictions) directly on top. Closing it returns the admin back to the segment list.

### ⚙️ Customizable High-Risk Limits
- Added a Top N configuration selector (dropdown menu) directly inside the High-Risk table card.
- Allows admins to instantly change table rows between 10, 20, 30, 50, or Show All, calculated in memory without page refreshes.

### 🔍 Autocomplete Global Navbar Search
- Added a student lookup input in the top header navbar on `/dashboard`.
- Performs quick lookups as you type, highlighting matched student names and IDs. Suggestions link directly to the student profile detail modal.

### 🧹 Deprecated Files Cleanup
- Marked deprecated `app.py` and `verify_endpoints.py` files as deleted (recommend manual IDE deletion due to host shell permissions).

---

## Update 3 — CEO Review: UX Overhaul & Security *(26 May 2026)*

> *These updates were implemented after consulting the CEO to ensure the product is market-ready, visually stunning, and highly secure.*

### 🎨 Marketing Landing Page Added
- Created a brand-new, high-conversion landing page (`/`) explaining the platform's value proposition, features, models, and team.
- Includes live dynamic system metrics pulled directly from the backend API.

### 📊 Admin Analytics Dashboard Added
- Built a comprehensive Analytics Dashboard (`/dashboard`).
- Features real-time KPI cards, Chart.js visualizations (Risk Distribution, Fee Status, Attendance Histograms, Class Breakdown), a System Health gauge, and a Recent Alerts feed.

### 📱 All-Device Responsiveness & Unified Navigation
- Built a unified, persistent navigation bar (`global-nav`) shared across the Landing Page, Student Directory, and Admin Dashboard.
- **Mobile Perfection:** On screens under 640px, the navigation bar smartly snaps to a sticky bottom-bar (app-like experience) to guarantee users are never trapped on a page without navigation.
- Added horizontal scrolling to the Student Directory table to prevent layout breaks on small mobile screens.

### 🔒 Admin Page Security (Hardcoded Authentication)
- Secured the `/dashboard` route with HTTP Basic Authentication to prevent unauthorized access to sensitive analytics.
- The Landing Page and Student Directory remain public.
- **Admin Credentials (Temporary for Teammates):**
  - **Username:** `admin`
  - **Password:** `kalnet2026`
- *Note: This can be easily disabled by setting `REQUIRE_ADMIN_AUTH = False` in `main.py`.*

---

## Update 2 — Fee Model & UI Overhaul *(14 May 2026)*

### 🐛 Critical Bug Fixed: Default Probability Showing 0% or 100% Only

**Root Cause — Data Leakage in Feature Engineering (`mohammed_kaif/feature_engineering.py`)**

The original code was using **Term 3's own data** (the term being predicted) as input features to the model. This caused the model to learn a trivial pattern:

- `status=0` → `days_since_last_payment=0`, `outstanding=0` → model outputs 0% default
- `status=2` → `days_since_last_payment=90`, `outstanding=5000` → model outputs 100% default

No genuine prediction was happening — the model was just reading the answer from its own features.

**Fix Applied:**

Features now use **only Terms 1 & 2 historical data** to predict Term 3's outcome:

```python
# Before (BROKEN — data leakage)
'days_since_last_payment': latest_record['days_since_last_payment'],  # Term 3 data!
'total_outstanding': latest_record['total_outstanding'],               # Term 3 data!

# After (FIXED — genuine prediction from history)
'days_since_last_payment': avg_days_late,    # Average of Terms 1 & 2
'total_outstanding':       avg_outstanding,  # Average of Terms 1 & 2
```

**Result:** Default probabilities now span the full range — e.g., 4%, 18%, 43%, 72%, 91%.

---

### 🐛 Bug Fixed: Fee Model Recall Below 70% Target

**Problem:** After fixing the data leakage, the model now faced a genuine prediction task, but the vanilla GradientBoosting only achieved 50% recall on the default class (target: ≥70%).

**Fix Applied in `are_samhith/train.py`:**

1. **Sample weights** — default class students given 4× weight during training so the model learns to prioritise catching them
2. **More estimators + tuned depth** — `n_estimators=200`, `max_depth=4`, `learning_rate=0.08`
3. **Automatic threshold tuning** — scans thresholds from 0.10–0.80 and selects the lowest threshold that achieves ≥70% default recall
4. **Saved threshold** — `models/fee_predictor/threshold.pkl` saved alongside the model and loaded by the API

**Result:** Default class recall now consistently ≥70% ✅

---

### 🐛 Bug Fixed: `class` Column Breaking ML Training

**Problem:** After adding the `class` column (e.g., "6A", "9B") to `student_labels.csv`, the feature engineering merge pulled it into `attendance_features.csv`. The scaler then failed because "6A" cannot be converted to float.

**Fix:** Feature engineering now explicitly selects only `is_anomaly` when merging:

```python
attendance_features = attendance_features.merge(
    df_labels[['student_id', 'is_anomaly']], on='student_id'
)
```

The `class` field is used only by the API and UI display, never by the ML pipeline.

---

### 🎨 UI Redesign — Card Grid → Professional Table Layout

**Feedback from Superior:** "It is looking very messy for the first impression — make a list of students rather than give each student a card."

**Changes made in `templates/index.html` + new `templates/style.css`:**

| Feature | Before | After |
|---------|--------|-------|
| Layout | Card grid (500 cards) | Clean table with columns |
| Columns | N/A | Name, Class, Attendance, Risk Score, Fee Status, Default Prob, Status |
| Pagination | None (all 500 loaded) | 20 per page with navigation |
| Sorting | None | Sort by risk, attendance, or default probability |
| Class column | Not shown | Shows class (e.g., 9A, 7B) per student |
| CSS | Inline in HTML | Separate `style.css` file |

---

### 🔄 Fee Data Generator Made Probabilistic

**Problem:** All students had fixed payment profiles (always on-time, always late, always default).

**Fix in `mohammed_kaif/data_generator.py`:** Each term's payment outcome is now a **random draw** from the student's probability distribution, influenced by income bracket and sibling count.

```
On-time tendency:  85% on-time, 12% late, 3% default — per term
Late tendency:     30% on-time, 50% late, 20% default — per term
Default tendency:  5% on-time, 15% late, 80% default — per term
```

---

### 📡 API Updated (`jyothsna_lenka/main.py`)

- Loads `models/fee_predictor/threshold.pkl` at startup alongside model
- Fee labels shown in UI are now threshold-adjusted (consistent with model training decisions)
- Added `student_class` field to all API responses for table display

---

## Update 1 — Initial Full Build *(11 May 2026)*

- Complete FastAPI backend with `/api/summary`, `/api/students`, `/api/student/{id}` endpoints
- Original POST endpoints `/ai/anomalies` and `/ai/fee-risk` retained (per Jyothsna's requirements)
- Frontend dashboard built in `templates/index.html` (white theme)
- Models load at startup, student cache pre-computed for instant UI response
- `render.yaml` added for free Render.com hosting
- Comprehensive `README.md` and `jyothsna_lenka/jyothsna.md` API docs written
- `vodyati_sai_phanindra/evaluate.py` enhanced with confusion matrix, feature importances, and markdown report generation
