"""
========================================================
KALNET — Fee Default Dataset Generator (3-Term Version)
File   : models/fee_predictor/generate_dataset.py
Author : Are Samhith (ML Engineer 2)

WHAT THIS SCRIPT DOES:
  Generates a realistic synthetic dataset for training the
  Fee Default Predictor model in KALNET.

WHY 3 TERMS:
  A single snapshot (current V2) can't capture TREND.
  A student going:  ontime → late → DEFAULT
  is far more predictable than just seeing "late" today.
  
  Term 1 + Term 2 = history features (what happened before)
  Term 3 label    = what the model predicts (will they default?)

WHY 5% DEFAULT RATE:
  Team specification: "default class is only 5 percent"
  This is realistic — most schools have 5-8% fee defaulters.
  Makes the problem harder but more real-world accurate.

OUTPUT:
  fee_features_v3.csv — 2000 rows, ~100 defaulters (5%)
  One row per student (flattened from 3 terms)
========================================================
"""

import pandas as pd
import numpy as np
import os

# ========================================================
# CONFIG
# ========================================================
N_STUDENTS          = 2000
TARGET_DEFAULT_RATE = 0.05      # 5% as specified by team
RANDOM_SEED         = 42
OUTPUT_FILE         = "data/fee_features_v3.csv"

np.random.seed(RANDOM_SEED)

print("=" * 60)
print("KALNET — Fee Default Dataset Generator (3-Term)")
print("=" * 60)
print(f"\n[CONFIG] Students to generate : {N_STUDENTS}")
print(f"[CONFIG] Target default rate  : {TARGET_DEFAULT_RATE*100:.0f}%")
print(f"[CONFIG] Random seed          : {RANDOM_SEED}")

# ========================================================
# STEP 1: GENERATE STATIC STUDENT FEATURES
#
# These features don't change across terms.
# They represent the student's background/profile.
# ========================================================
print("\n[STEP 1] Generating static student features...")

student_ids = [f"STU_{i:03d}" for i in range(1, N_STUDENTS + 1)]

# income_encoded
#   0 = High income   → 30% of students
#   1 = Medium income → 45% of students
#   2 = Low income    → 25% of students
income_encoded = np.random.choice(
    [0, 1, 2],
    size=N_STUDENTS,
    p=[0.30, 0.45, 0.25]
)

# sibling_count: 0 to 5
# More siblings = more financial burden on family
sibling_count = np.random.choice(
    [0, 1, 2, 3, 4, 5],
    size=N_STUDENTS,
    p=[0.25, 0.30, 0.22, 0.13, 0.07, 0.03]
)

# transport_user: 1 = uses school bus (extra fee burden)
transport_user = np.random.choice(
    [0, 1],
    size=N_STUDENTS,
    p=[0.55, 0.45]
)

print(f"  income_encoded → High:{(income_encoded==0).sum()} | Medium:{(income_encoded==1).sum()} | Low:{(income_encoded==2).sum()}")
print(f"  sibling_count  → Mean:{sibling_count.mean():.2f} | Max:{sibling_count.max()}")
print(f"  transport_user → Uses bus: {transport_user.sum()} ({transport_user.mean()*100:.1f}%)")

# ========================================================
# STEP 2: GENERATE TERM 1 STATUS
#
# Term 1 is the starting point — mostly on-time students.
# p=[0.75, 0.18, 0.07] means:
#   75% pay on time in Term 1
#   18% are late in Term 1
#    7% default even in Term 1 (chronic defaulters)
# ========================================================
print("\n[STEP 2] Generating Term 1 payment status...")

t1_status = np.random.choice(
    [0, 1, 2],
    size=N_STUDENTS,
    p=[0.75, 0.18, 0.07]
)

print(f"  T1 On-time : {(t1_status==0).sum()} ({(t1_status==0).mean()*100:.1f}%)")
print(f"  T1 Late    : {(t1_status==1).sum()} ({(t1_status==1).mean()*100:.1f}%)")
print(f"  T1 Default : {(t1_status==2).sum()} ({(t1_status==2).mean()*100:.1f}%)")

# ========================================================
# STEP 3: GENERATE TERM 1 FINANCIAL FEATURES
#
# outstanding and days_late depend on payment status.
# Default student → higher outstanding, more days unpaid.
# ========================================================
print("\n[STEP 3] Generating Term 1 financial features...")

income_mult    = np.where(income_encoded == 2, 1.6,
                 np.where(income_encoded == 1, 1.1, 0.7))

t1_status_mult = np.where(t1_status == 2, 2.5,
                 np.where(t1_status == 1, 1.4, 0.8))

# Total outstanding for Term 1
t1_outstanding_base = np.random.exponential(scale=1800, size=N_STUDENTS)
t1_outstanding = np.clip(
    t1_outstanding_base * income_mult * (t1_status_mult * 0.8),
    0, 8000
).round(0)

# On-time students → outstanding near 0
t1_outstanding = np.where(
    t1_status == 0,
    np.random.uniform(0, 200, N_STUDENTS),
    t1_outstanding
)

# Days since last payment for Term 1
t1_days_base = np.random.exponential(scale=25, size=N_STUDENTS)
t1_days_late = np.clip(t1_days_base * t1_status_mult, 0, 120).round(0)
t1_days_late = np.where(
    t1_status == 0,
    np.random.uniform(0, 10, N_STUDENTS),
    t1_days_late
)

print(f"  T1 outstanding → Mean:Rs.{t1_outstanding.mean():.0f} | Max:Rs.{t1_outstanding.max():.0f}")
print(f"  T1 days_late   → Mean:{t1_days_late.mean():.1f} | Max:{t1_days_late.max():.0f}")

# ========================================================
# STEP 4: GENERATE TERM 2 STATUS — MARKOV CHAIN
#
# WHY MARKOV CHAIN:
#   Term 2 status is NOT independent of Term 1.
#   A student who defaulted in Term 1 is much more likely
#   to default again in Term 2.
#
#   Transition probabilities:
#   T1=ontime  → T2: 78% ontime, 17% late,  5% default
#   T1=late    → T2: 40% ontime, 40% late, 20% default
#   T1=default → T2: 20% ontime, 30% late, 50% default
#
#   A T1=default student is 10x more likely to default
#   in T2 compared to a T1=ontime student.
# ========================================================
print("\n[STEP 4] Generating Term 2 status (Markov chain from Term 1)...")

t2_status = np.zeros(N_STUDENTS, dtype=int)

transition = {
    0: [0.78, 0.17, 0.05],   # was on-time → likely stays on-time
    1: [0.40, 0.40, 0.20],   # was late    → could go either way
    2: [0.20, 0.30, 0.50],   # was default → likely defaults again
}

for i in range(N_STUDENTS):
    t2_status[i] = np.random.choice([0, 1, 2], p=transition[t1_status[i]])

print(f"  T2 On-time : {(t2_status==0).sum()} ({(t2_status==0).mean()*100:.1f}%)")
print(f"  T2 Late    : {(t2_status==1).sum()} ({(t2_status==1).mean()*100:.1f}%)")
print(f"  T2 Default : {(t2_status==2).sum()} ({(t2_status==2).mean()*100:.1f}%)")

# ========================================================
# STEP 5: GENERATE TERM 2 FINANCIAL FEATURES
#
# If T1 already had outstanding debt, T2 accumulates more.
# accumulated_debt carries forward 40% of T1 outstanding
# when both terms are late/default.
# ========================================================
print("\n[STEP 5] Generating Term 2 financial features...")

t2_status_mult = np.where(t2_status == 2, 2.5,
                 np.where(t2_status == 1, 1.4, 0.8))

t2_outstanding_base = np.random.exponential(scale=1800, size=N_STUDENTS)
t2_outstanding = np.clip(
    t2_outstanding_base * income_mult * (t2_status_mult * 0.9),
    0, 8000
).round(0)

t2_outstanding = np.where(
    t2_status == 0,
    np.random.uniform(0, 200, N_STUDENTS),
    t2_outstanding
)

# Accumulated debt: if both T1 and T2 have issues, debt compounds
accumulated_debt = np.where(
    (t1_status >= 1) & (t2_status >= 1),
    t1_outstanding * 0.4,
    0
)
t2_outstanding = np.clip(t2_outstanding + accumulated_debt, 0, 8000).round(0)

t2_days_base = np.random.exponential(scale=25, size=N_STUDENTS)
t2_days_late = np.clip(t2_days_base * t2_status_mult, 0, 120).round(0)
t2_days_late = np.where(
    t2_status == 0,
    np.random.uniform(0, 10, N_STUDENTS),
    t2_days_late
)

print(f"  T2 outstanding → Mean:Rs.{t2_outstanding.mean():.0f} | Max:Rs.{t2_outstanding.max():.0f}")
print(f"  T2 days_late   → Mean:{t2_days_late.mean():.1f} | Max:{t2_days_late.max():.0f}")

# ========================================================
# STEP 6: COMPUTE DERIVED TREND FEATURES
#
# outstanding_growth > 0  → debt is increasing = bad sign
# days_late_trend > 0     → paying later each term = bad
# escalating = 1          → status got worse term over term
# both_terms_late = 1     → consistently late = high risk
# avg_outstanding         → overall debt level across terms
# max_days_late           → worst payment delay seen
# ========================================================
print("\n[STEP 6] Computing derived trend features...")

outstanding_growth = t2_outstanding - t1_outstanding
days_late_trend    = t2_days_late - t1_days_late
both_terms_late    = ((t1_status >= 1) & (t2_status >= 1)).astype(int)
escalating         = (t2_status > t1_status).astype(int)
avg_outstanding    = ((t1_outstanding + t2_outstanding) / 2).round(0)
max_days_late      = np.maximum(t1_days_late, t2_days_late)

print(f"  outstanding_growth → Mean:{outstanding_growth.mean():.0f} (positive=debt growing)")
print(f"  days_late_trend    → Mean:{days_late_trend.mean():.1f} (positive=getting later)")
print(f"  both_terms_late    → {both_terms_late.sum()} students ({both_terms_late.mean()*100:.1f}%)")
print(f"  escalating         → {escalating.sum()} students ({escalating.mean()*100:.1f}%) worsening")

# ========================================================
# STEP 7: ASSIGN TERM 3 LABEL — RISK SCORE FORMULA
#
# Label is derived from a weighted risk score, NOT random.
# This ensures the model CAN learn real patterns.
#
# Weight rationale:
#   t2_status × 3.0     → most recent term = strongest signal
#   t1_status × 1.5     → older term = weaker but still matters
#   escalating × 2.0    → worsening trajectory = very high risk
#   both_terms_late×1.5 → consistency of lateness
#   t2_outstanding×2.0  → how much is currently owed
#   t2_days_late×2.0    → how long since last payment
#   income × 1.5        → financial capacity
#   siblings × 1.0      → family financial burden
#   transport × 0.5     → extra expense signal
#   noise(0, 0.8)       → real-world unpredictability
#
# Threshold = 95th percentile → exactly 5% labeled DEFAULT
# ========================================================
print("\n[STEP 7] Computing risk scores and assigning Term 3 labels...")

risk_score = (
    t1_status               * 1.5  +
    t2_status               * 3.0  +
    escalating              * 2.0  +
    both_terms_late         * 1.5  +
    t2_outstanding / 2000   * 2.0  +
    t2_days_late   / 30     * 2.0  +
    income_encoded          * 1.5  +
    sibling_count  / 2      * 1.0  +
    transport_user          * 0.5  +
    np.random.normal(0, 0.8, N_STUDENTS)
)

threshold_percentile = (1 - TARGET_DEFAULT_RATE) * 100
risk_threshold = np.percentile(risk_score, threshold_percentile)
label = np.where(risk_score >= risk_threshold, 2, 0)

actual_rate  = (label == 2).mean()
n_defaulters = (label == 2).sum()

print(f"  Risk score range  : {risk_score.min():.2f} to {risk_score.max():.2f}")
print(f"  Risk threshold    : {risk_threshold:.3f}  (top {TARGET_DEFAULT_RATE*100:.0f}%)")
print(f"  Actual defaulters : {n_defaulters} / {N_STUDENTS} ({actual_rate*100:.1f}%)")

# ========================================================
# STEP 8: ASSEMBLE FINAL DATAFRAME
# ========================================================
print("\n[STEP 8] Assembling final dataframe...")

df = pd.DataFrame({
    "student_id"         : student_ids,
    "t1_status"          : t1_status,
    "t1_outstanding"     : t1_outstanding.astype(int),
    "t1_days_late"       : t1_days_late.astype(int),
    "t2_status"          : t2_status,
    "t2_outstanding"     : t2_outstanding.astype(int),
    "t2_days_late"       : t2_days_late.astype(int),
    "income_encoded"     : income_encoded,
    "sibling_count"      : sibling_count,
    "transport_user"     : transport_user,
    "outstanding_growth" : outstanding_growth.astype(int),
    "days_late_trend"    : days_late_trend.astype(int),
    "both_terms_late"    : both_terms_late,
    "escalating"         : escalating,
    "avg_outstanding"    : avg_outstanding.astype(int),
    "max_days_late"      : max_days_late.astype(int),
    "label"              : label,
})

print(f"  Shape   : {df.shape}")
print(f"  Columns : {list(df.columns)}")

# ========================================================
# STEP 9: VALIDATION CHECKS
# ========================================================
print("\n" + "=" * 60)
print("[VALIDATION] Checking dataset quality...")
print("=" * 60)

def_mask   = df["label"] == 2
nodef_mask = df["label"] == 0

# Check 1: Default rate
print(f"\n[CHECK 1] Default rate")
print(f"  Target : {TARGET_DEFAULT_RATE*100:.1f}%")
print(f"  Actual : {actual_rate*100:.1f}%")
print(f"  Status : {'PASS' if abs(actual_rate - TARGET_DEFAULT_RATE) < 0.01 else 'CHECK'}")

# Check 2: Feature correlation
print(f"\n[CHECK 2] Defaulters vs Non-defaulters (defaulters must be higher)")
print(f"  {'Feature':<22} {'Defaulters':>12} {'Non-default':>12} {'Ratio':>8} {'Status':>6}")
print("  " + "-" * 65)

num_checks = [
    ("t2_outstanding",    "Rs.", 0),
    ("t2_days_late",      "d",   0),
    ("income_encoded",    "",    2),
    ("sibling_count",     "",    2),
    ("outstanding_growth","Rs.", 0),
    ("avg_outstanding",   "Rs.", 0),
    ("max_days_late",     "d",   0),
]

for col, unit, dec in num_checks:
    d_mean  = df.loc[def_mask,   col].mean()
    nd_mean = df.loc[nodef_mask, col].mean()
    ratio   = d_mean / nd_mean if nd_mean != 0 else 0
    status  = "PASS" if ratio > 1.2 else "WEAK" if ratio > 1.0 else "FAIL"
    print(f"  {col:<22} {d_mean:>10.{dec}f}{unit} {nd_mean:>10.{dec}f}{unit} {ratio:>7.2f}x {status:>6}")

print(f"\n  {'Feature':<22} {'Defaulters%':>12} {'Non-default%':>12} {'Status':>6}")
print("  " + "-" * 55)

for col in ["both_terms_late", "escalating", "transport_user"]:
    d_pct  = df.loc[def_mask,   col].mean() * 100
    nd_pct = df.loc[nodef_mask, col].mean() * 100
    status = "PASS" if d_pct > nd_pct * 1.2 else "WEAK"
    print(f"  {col:<22} {d_pct:>11.1f}% {nd_pct:>11.1f}% {status:>6}")

# Check 3: Markov chain validation
print(f"\n[CHECK 3] T1 status -> T3 default rate (Markov chain check)")
print(f"  {'T1 Status':<15} {'T3 Default %':>14} {'Count':>8}")
print("  " + "-" * 40)

for s, name in [(0,"T1=On-time"), (1,"T1=Late"), (2,"T1=Default")]:
    mask     = df["t1_status"] == s
    def_rate = df.loc[mask, "label"].apply(lambda x: 1 if x == 2 else 0).mean() * 100
    count    = mask.sum()
    print(f"  {name:<15} {def_rate:>13.1f}% {count:>8}")

print(f"  (T1=Default must have highest T3 default rate)")

# Check 4: Escalating trend
print(f"\n[CHECK 4] Escalating trend -> T3 default rate")
esc_rate    = df.loc[df["escalating"]==1, "label"].apply(lambda x: 1 if x==2 else 0).mean() * 100
nonesc_rate = df.loc[df["escalating"]==0, "label"].apply(lambda x: 1 if x==2 else 0).mean() * 100
status = "PASS" if esc_rate > nonesc_rate * 2 else "WEAK"
print(f"  Escalating     -> {esc_rate:.1f}% default in T3")
print(f"  Non-escalating -> {nonesc_rate:.1f}% default in T3")
print(f"  Status: {status}  (escalating must default 2x+ more)")

# ========================================================
# STEP 10: SAMPLE ROWS
# ========================================================
print(f"\n[SAMPLE] 3 defaulters and 3 non-defaulters")
print("-" * 60)

print("\n  DEFAULTERS (label=2):")
for _, row in df[df["label"]==2].head(3).iterrows():
    print(f"  {row['student_id']} | T1:{int(row['t1_status'])} -> T2:{int(row['t2_status'])} -> T3:DEFAULT | "
          f"T2_out:Rs.{int(row['t2_outstanding'])} | days:{int(row['t2_days_late'])} | "
          f"esc:{int(row['escalating'])} | inc:{int(row['income_encoded'])}")

print("\n  NON-DEFAULTERS (label=0):")
for _, row in df[df["label"]==0].head(3).iterrows():
    print(f"  {row['student_id']} | T1:{int(row['t1_status'])} -> T2:{int(row['t2_status'])} -> T3:ONTIME  | "
          f"T2_out:Rs.{int(row['t2_outstanding'])} | days:{int(row['t2_days_late'])} | "
          f"esc:{int(row['escalating'])} | inc:{int(row['income_encoded'])}")

# ========================================================
# STEP 11: SAVE
# ========================================================
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n[SAVE] {OUTPUT_FILE} saved")
print(f"[SAVE] {N_STUDENTS} rows | {n_defaulters} defaulters | {len(df.columns)} columns")

print("""
========================================================
NEXT STEP: train_improved_v2.py
========================================================

FEATURE_COLS = [
    't1_status', 't1_outstanding', 't1_days_late',
    't2_status', 't2_outstanding', 't2_days_late',
    'income_encoded', 'sibling_count', 'transport_user',
    'outstanding_growth', 'days_late_trend', 'both_terms_late',
    'escalating', 'avg_outstanding', 'max_days_late'
]

TARGET: df["fee_default"] = (df["label"] == 2).astype(int)

MODEL:
  GradientBoostingClassifier only
  compute_sample_weight('balanced') — NO SMOTE
  StratifiedKFold(n_splits=5)
  Optimize: average_precision_score + fbeta_score(beta=2)
  Goal: Recall >= 80%, Precision >= 30%, CV AUC std < 0.04

EXPECTED with 3-term features:
  CV AUC std  : 0.08 -> ~0.03  (stable)
  AUC-ROC     : 0.68 -> 0.82+
  Recall      : 75%  -> 85%+
  Precision   : 28%  -> 35%+
========================================================
""")

print("=" * 60)
print("COMPLETE")
print(f"File      : {OUTPUT_FILE}")
print(f"Rows      : {N_STUDENTS}")
print(f"Defaulters: {n_defaulters} ({actual_rate*100:.1f}%)")
print(f"Features  : {len(df.columns)-2} (excl. student_id, label)")
print("=" * 60)