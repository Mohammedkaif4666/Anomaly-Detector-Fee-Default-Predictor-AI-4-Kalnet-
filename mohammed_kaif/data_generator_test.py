"""
KALNET — Inference Dataset Generator
Generates 500 students (STU_001 to STU_500) with:
  - 5% default rate (~25 defaulters)
  - Spread of probabilities: 0-10%, 10-30%, 30-60%, 60-90%, 90-100%
  - Realistic feature values matching training distribution
"""

import pandas as pd
import numpy as np

np.random.seed(99)  # different seed from training (42) for variety

N          = 500
OUT_FILE   = "data/fee_features_v3.csv"

student_ids = [f"STU_{i:03d}" for i in range(1, N + 1)]

# ── We generate 5 risk BUCKETS with defined sizes ──────────────
# This guarantees probability spread on the website
#
# VERY LOW  (prob ~0-10%)  : 200 students → clearly safe
# LOW-MED   (prob ~10-30%) : 100 students → mild risk
# MEDIUM    (prob ~30-60%) : 100 students → moderate risk
# HIGH      (prob ~60-85%) :  75 students → high risk
# VERY HIGH (prob ~85-100%):  25 students → certain defaulters (5%)
#
# Total = 500 students, ~25 defaults (5%)

BUCKETS = [
    # (count, t1_status_probs, t2_status_probs, income_probs,
    #  out_scale, days_scale, label_default_prob)
    # VERY LOW RISK — pay on time, low outstanding, high income
    dict(
        n=200,
        t1_p=[0.92, 0.07, 0.01],
        t2_p_given_t1={0:[0.93,0.06,0.01], 1:[0.60,0.35,0.05], 2:[0.40,0.35,0.25]},
        inc_p=[0.55, 0.35, 0.10],
        sib_p=[0.40, 0.35, 0.15, 0.07, 0.02, 0.01],
        out_scale=400,
        days_scale=8,
        label_rate=0.00,   # zero defaults in this bucket
    ),
    # LOW-MEDIUM RISK — occasional late, medium outstanding
    dict(
        n=100,
        t1_p=[0.70, 0.25, 0.05],
        t2_p_given_t1={0:[0.72,0.24,0.04], 1:[0.38,0.45,0.17], 2:[0.25,0.40,0.35]},
        inc_p=[0.30, 0.50, 0.20],
        sib_p=[0.25, 0.35, 0.25, 0.10, 0.04, 0.01],
        out_scale=900,
        days_scale=18,
        label_rate=0.00,   # zero defaults in this bucket
    ),
    # MEDIUM RISK — mixed history, moderate outstanding
    dict(
        n=100,
        t1_p=[0.50, 0.35, 0.15],
        t2_p_given_t1={0:[0.55,0.33,0.12], 1:[0.30,0.42,0.28], 2:[0.18,0.35,0.47]},
        inc_p=[0.20, 0.45, 0.35],
        sib_p=[0.20, 0.28, 0.27, 0.15, 0.07, 0.03],
        out_scale=2000,
        days_scale=35,
        label_rate=0.00,   # zero defaults in this bucket
    ),
    # HIGH RISK — mostly late/default history, high outstanding
    dict(
        n=75,
        t1_p=[0.25, 0.40, 0.35],
        t2_p_given_t1={0:[0.35,0.40,0.25], 1:[0.18,0.38,0.44], 2:[0.10,0.28,0.62]},
        inc_p=[0.10, 0.35, 0.55],
        sib_p=[0.10, 0.20, 0.28, 0.22, 0.12, 0.08],
        out_scale=4000,
        days_scale=60,
        label_rate=0.00,   # zero defaults in this bucket
    ),
    # VERY HIGH RISK — chronic defaulters, max outstanding
    dict(
        n=25,
        t1_p=[0.10, 0.30, 0.60],
        t2_p_given_t1={0:[0.20,0.35,0.45], 1:[0.08,0.27,0.65], 2:[0.05,0.18,0.77]},
        inc_p=[0.05, 0.25, 0.70],
        sib_p=[0.05, 0.15, 0.25, 0.28, 0.17, 0.10],
        out_scale=6500,
        days_scale=90,
        label_rate=1.00,   # ALL of these are defaults (25 students = 5%)
    ),
]

rows = []
for bucket in BUCKETS:
    n         = bucket["n"]
    inc_enc   = np.random.choice([0,1,2], size=n, p=bucket["inc_p"])
    sib_cnt   = np.random.choice([0,1,2,3,4,5], size=n, p=bucket["sib_p"])
    transport = np.random.choice([0,1], size=n, p=[0.55,0.45])

    # income multiplier: low income → higher outstanding
    inc_mult = np.where(inc_enc==2, 1.7,
               np.where(inc_enc==1, 1.1, 0.65))

    # Term 1 status
    t1_status = np.array([
        np.random.choice([0,1,2], p=bucket["t1_p"]) for _ in range(n)
    ])

    # Term 2 status — Markov chain from T1
    t2_p_map = bucket["t2_p_given_t1"]
    t2_status = np.array([
        np.random.choice([0,1,2], p=t2_p_map[t1_status[i]]) for i in range(n)
    ])

    # T1 financial features
    t1_stat_mult = np.where(t1_status==2, 2.8,
                   np.where(t1_status==1, 1.5, 0.6))
    t1_out_base  = np.random.exponential(scale=bucket["out_scale"], size=n)
    t1_out       = np.clip(t1_out_base * inc_mult * t1_stat_mult * 0.7, 0, 8000).round(0)
    t1_out       = np.where(t1_status==0, np.random.uniform(0, 250, n), t1_out)

    t1_days_base = np.random.exponential(scale=bucket["days_scale"], size=n)
    t1_days      = np.clip(t1_days_base * t1_stat_mult * 0.8, 0, 120).round(0)
    t1_days      = np.where(t1_status==0, np.random.uniform(0, 12, n), t1_days)

    # T2 financial features (accumulated debt if both late)
    t2_stat_mult = np.where(t2_status==2, 2.8,
                   np.where(t2_status==1, 1.5, 0.6))
    t2_out_base  = np.random.exponential(scale=bucket["out_scale"], size=n)
    t2_out       = np.clip(t2_out_base * inc_mult * t2_stat_mult * 0.9, 0, 8000).round(0)
    t2_out       = np.where(t2_status==0, np.random.uniform(0, 200, n), t2_out)

    # Accumulate debt if both terms had issues
    accum = np.where((t1_status>=1) & (t2_status>=1), t1_out * 0.45, 0)
    t2_out = np.clip(t2_out + accum, 0, 8000).round(0)

    t2_days_base = np.random.exponential(scale=bucket["days_scale"], size=n)
    t2_days      = np.clip(t2_days_base * t2_stat_mult * 0.9, 0, 120).round(0)
    t2_days      = np.where(t2_status==0, np.random.uniform(0, 10, n), t2_days)

    # Derived trend features
    out_growth    = (t2_out - t1_out).astype(int)
    days_trend    = (t2_days - t1_days).astype(int)
    both_late     = ((t1_status>=1) & (t2_status>=1)).astype(int)
    escalating    = (t2_status > t1_status).astype(int)
    avg_out       = ((t1_out + t2_out) / 2).round(0).astype(int)
    max_days      = np.maximum(t1_days, t2_days).astype(int)

    # Label assignment
    labels = np.random.choice(
        [0, 2], size=n,
        p=[1 - bucket["label_rate"], bucket["label_rate"]]
    )

    for i in range(n):
        rows.append({
            "t1_status"         : int(t1_status[i]),
            "t1_outstanding"    : int(t1_out[i]),
            "t1_days_late"      : int(t1_days[i]),
            "t2_status"         : int(t2_status[i]),
            "t2_outstanding"    : int(t2_out[i]),
            "t2_days_late"      : int(t2_days[i]),
            "income_encoded"    : int(inc_enc[i]),
            "sibling_count"     : int(sib_cnt[i]),
            "transport_user"    : int(transport[i]),
            "outstanding_growth": int(out_growth[i]),
            "days_late_trend"   : int(days_trend[i]),
            "both_terms_late"   : int(both_late[i]),
            "escalating"        : int(escalating[i]),
            "avg_outstanding"   : int(avg_out[i]),
            "max_days_late"     : int(max_days[i]),
            "label"             : int(labels[i]),
        })

# Shuffle so buckets are mixed (not all safe students first)
np.random.shuffle(rows)

df = pd.DataFrame(rows)
df.insert(0, "student_id", student_ids)

# ── Validation ────────────────────────────────────────────────
print("=" * 55)
print("KALNET Inference Dataset — Validation")
print("=" * 55)

n_def  = (df["label"]==2).sum()
n_safe = (df["label"]==0).sum()
print(f"\nTotal students : {len(df)}")
print(f"Defaulters     : {n_def}  ({n_def/len(df)*100:.1f}%)")
print(f"Non-defaulters : {n_safe}")

# Load model and check probability spread
import joblib, os

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models/fee_predictor/model_v3.pkl"
)

if os.path.exists(model_path):
    print(f"\n[MODEL] Loading {model_path}")
    bundle      = joblib.load(model_path)
    model       = bundle["model"]
    feat_cols   = bundle["feature_cols"]
    threshold   = bundle["threshold"]

    X     = df[feat_cols].values
    probs = model.predict_proba(X)[:, 1] * 100

    df["pred_prob"] = probs.round(1)

    print(f"\n[PROBABILITY SPREAD]")
    print(f"  0-10%    : {((probs>=0)  & (probs<10)).sum():>4} students")
    print(f"  10-30%   : {((probs>=10) & (probs<30)).sum():>4} students")
    print(f"  30-50%   : {((probs>=30) & (probs<50)).sum():>4} students")
    print(f"  50-70%   : {((probs>=50) & (probs<70)).sum():>4} students")
    print(f"  70-90%   : {((probs>=70) & (probs<90)).sum():>4} students")
    print(f"  90-100%  : {((probs>=90)).sum():>4} students")
    print(f"\n  Min prob : {probs.min():.1f}%")
    print(f"  Max prob : {probs.max():.1f}%")
    print(f"  Mean prob: {probs.mean():.1f}%")

    flagged = (probs >= threshold * 100).sum()
    print(f"\n  Threshold         : {threshold:.3f} ({threshold*100:.1f}%)")
    print(f"  Students flagged  : {flagged} / {len(df)}")

    # Sample from each bucket for visual check
    print(f"\n[SAMPLE STUDENTS across probability ranges]")
    print(f"  {'Student':<10} {'T1→T2':>6} {'T2_out':>8} {'T2_days':>8} {'Prob':>8} {'Label':>7}")
    print("  " + "-" * 55)

    ranges = [(0,10,"VERY LOW"), (10,30,"LOW"), (30,60,"MEDIUM"),
              (60,85,"HIGH"), (85,101,"VERY HIGH")]
    for lo, hi, label in ranges:
        mask  = (probs >= lo) & (probs < hi)
        idxs  = np.where(mask)[0]
        if len(idxs) == 0:
            print(f"  {label}: no students in this range")
            continue
        pick  = idxs[len(idxs)//2]  # pick middle student
        row   = df.iloc[pick]
        t1t2  = f"{int(row.t1_status)}→{int(row.t2_status)}"
        actual = "DEFAULT" if row.label==2 else "safe"
        print(f"  {row.student_id:<10} {t1t2:>6} "
              f"Rs.{int(row.t2_outstanding):>6} "
              f"{int(row.t2_days_late):>6}d "
              f"{probs[pick]:>7.1f}%  {actual:>7}  [{label}]")

    df.drop(columns=["pred_prob"], inplace=True)
else:
    print(f"\n[WARN] model_v3.pkl not found at {model_path}")
    print("  Dataset saved without probability validation.")
    print("  Run this script from your project root.")

# ── Save ──────────────────────────────────────────────────────
df.to_csv(OUT_FILE, index=False)
print(f"\n[SAVE] {OUT_FILE} → {len(df)} rows saved")
print(f"[DONE] Place this file in your project's data/ folder")
print("=" * 55)