"""
Rohith Koppu — Data Pipeline Engineer
=====================================
Task: Verify that engineered features make intuitive sense before handing
      to Om (attendance model) and Are (fee model) for training.
      Run data quality checks: no NaN values, correct value ranges, correct
      label distribution.

Run this AFTER feature_engineering.py and BEFORE training the models.
"""

import pandas as pd
import numpy as np
import os
import sys


def check_attendance_features(df_att: pd.DataFrame) -> bool:
    """Verify attendance feature CSV is clean and intuitive."""
    print("\n" + "=" * 60)
    print("  [1/2] Attendance Feature Verification")
    print("=" * 60)

    passed = True

    # ── Shape ─────────────────────────────────────────────────────
    print(f"\n  Shape          : {df_att.shape[0]} rows × {df_att.shape[1]} cols")
    expected_cols = {'student_id', 'attendance_rate', 'longest_absence_streak',
                     'absence_in_last_30_days', 'day_of_week_variance', 'is_anomaly'}
    missing = expected_cols - set(df_att.columns)
    if missing:
        print(f"  ❌ Missing columns: {missing}")
        passed = False
    else:
        print(f"  ✅ All required columns present")

    # ── NaN check ─────────────────────────────────────────────────
    nan_counts = df_att.isnull().sum()
    if nan_counts.any():
        print(f"  ❌ NaN values found:\n{nan_counts[nan_counts > 0]}")
        passed = False
    else:
        print(f"  ✅ No NaN values")

    # ── Value ranges ──────────────────────────────────────────────
    att_min, att_max = df_att['attendance_rate'].min(), df_att['attendance_rate'].max()
    if not (0 <= att_min and att_max <= 1):
        print(f"  ❌ attendance_rate out of range [0,1]: min={att_min:.3f}, max={att_max:.3f}")
        passed = False
    else:
        print(f"  ✅ attendance_rate range OK: [{att_min:.3f}, {att_max:.3f}]")

    streak_max = df_att['longest_absence_streak'].max()
    if streak_max > 200:
        print(f"  ❌ longest_absence_streak seems too high: {streak_max}")
        passed = False
    else:
        print(f"  ✅ longest_absence_streak max: {streak_max} days")

    # ── Label distribution ────────────────────────────────────────
    label_dist = df_att['is_anomaly'].value_counts().sort_index()
    total = len(df_att)
    anomaly_pct = label_dist.get(1, 0) / total * 100
    normal_pct  = label_dist.get(0, 0) / total * 100
    print(f"\n  Label Distribution:")
    print(f"    Normal  (0): {label_dist.get(0,0):>4}  ({normal_pct:.1f}%)")
    print(f"    Anomaly (1): {label_dist.get(1,0):>4}  ({anomaly_pct:.1f}%)")
    if not (10 <= anomaly_pct <= 20):
        print(f"  ⚠️  Anomaly % is {anomaly_pct:.1f}% — expected ~14%. Check generator.")
    else:
        print(f"  ✅ Anomaly rate {anomaly_pct:.1f}% is within expected 10–20% range")

    # ── Intuition check: anomalous students should have lower attendance ──
    mean_normal  = df_att[df_att['is_anomaly'] == 0]['attendance_rate'].mean()
    mean_anomaly = df_att[df_att['is_anomaly'] == 1]['attendance_rate'].mean()
    print(f"\n  Intuition Check — Attendance Rate:")
    print(f"    Normal students avg   : {mean_normal:.1%}")
    print(f"    Anomalous students avg: {mean_anomaly:.1%}")
    if mean_anomaly < mean_normal:
        print(f"  ✅ Anomalous students have lower attendance — makes sense!")
    else:
        print(f"  ❌ Anomalous students have HIGHER attendance — data issue!")
        passed = False

    # ── Intuition check: anomalous students should have higher absence streaks ──
    streak_normal  = df_att[df_att['is_anomaly'] == 0]['longest_absence_streak'].mean()
    streak_anomaly = df_att[df_att['is_anomaly'] == 1]['longest_absence_streak'].mean()
    print(f"\n  Intuition Check — Absence Streaks:")
    print(f"    Normal students avg   : {streak_normal:.1f} days")
    print(f"    Anomalous students avg: {streak_anomaly:.1f} days")
    if streak_anomaly > streak_normal:
        print(f"  ✅ Anomalous students have longer streaks — makes sense!")
    else:
        print(f"  ❌ Anomalous students have shorter streaks — data issue!")
        passed = False

    # ── Sample ────────────────────────────────────────────────────
    print(f"\n  Sample Flagged Anomalies (top 5 by absence streak):")
    top5 = df_att[df_att['is_anomaly'] == 1].sort_values(
        'longest_absence_streak', ascending=False).head(5)
    print(f"  {'Student ID':<12} {'Att Rate':<12} {'Streak':<10} {'Abs 30d'}")
    print("  " + "-" * 44)
    for _, r in top5.iterrows():
        print(f"  {r['student_id']:<12} {r['attendance_rate']:.1%}      "
              f"{int(r['longest_absence_streak']):<10} {int(r['absence_in_last_30_days'])}")

    return passed


def check_fee_features(df_fee: pd.DataFrame) -> bool:
    """Verify fee feature CSV is clean and intuitive."""
    print("\n" + "=" * 60)
    print("  [2/2] Fee Feature Verification")
    print("=" * 60)

    passed = True

    # ── Shape ─────────────────────────────────────────────────────
    print(f"\n  Shape          : {df_fee.shape[0]} rows × {df_fee.shape[1]} cols")
    expected_cols = {'student_id', 'days_since_last_payment', 'previous_term_status',
                     'total_outstanding', 'income_encoded', 'transport_user',
                     'sibling_count', 'label'}
    missing = expected_cols - set(df_fee.columns)
    if missing:
        print(f"  ❌ Missing columns: {missing}")
        passed = False
    else:
        print(f"  ✅ All required columns present")

    # ── NaN check ─────────────────────────────────────────────────
    nan_counts = df_fee.isnull().sum()
    if nan_counts.any():
        print(f"  ❌ NaN values found:\n{nan_counts[nan_counts > 0]}")
        passed = False
    else:
        print(f"  ✅ No NaN values")

    # ── Value ranges ──────────────────────────────────────────────
    for col, expected in [
        ('income_encoded',       (0, 2)),
        ('transport_user',       (0, 1)),
        ('sibling_count',        (0, 3)),
        ('previous_term_status', (0, 2)),
        ('label',                (0, 2)),
    ]:
        mn, mx = df_fee[col].min(), df_fee[col].max()
        if mn < expected[0] or mx > expected[1]:
            print(f"  ❌ {col} out of range {expected}: min={mn}, max={mx}")
            passed = False
        else:
            print(f"  ✅ {col} range OK: [{mn}, {mx}]")

    # ── Label distribution ────────────────────────────────────────
    label_dist = df_fee['label'].value_counts().sort_index()
    total = len(df_fee)
    print(f"\n  Label Distribution:")
    labels = {0: 'On Time', 1: 'Late', 2: 'Default'}
    for k, v in label_dist.items():
        print(f"    {labels.get(k, k)} ({k}): {v:>4}  ({v/total*100:.1f}%)")

    # ── Intuition check: defaulters should have higher days_late ──
    grp = df_fee.groupby('label')['days_since_last_payment'].mean()
    print(f"\n  Intuition Check — Avg Days Late by Label:")
    for k, v in grp.items():
        print(f"    {labels.get(k,k)}: {v:.1f} days")
    if grp.get(2, 0) > grp.get(1, 0) > grp.get(0, 0):
        print(f"  ✅ Days late increases correctly: On Time < Late < Default")
    else:
        print(f"  ⚠️  Days late ordering not strictly increasing — check data")

    # ── Intuition check: defaulters should have higher outstanding ──
    grp2 = df_fee.groupby('label')['total_outstanding'].mean()
    print(f"\n  Intuition Check — Avg Outstanding by Label:")
    for k, v in grp2.items():
        print(f"    {labels.get(k,k)}: ₹{v:,.0f}")
    if grp2.get(2, 0) > grp2.get(0, 0):
        print(f"  ✅ Defaulters have higher outstanding — makes sense!")
    else:
        print(f"  ❌ Defaulters do NOT have higher outstanding — data issue!")
        passed = False

    # ── Intuition check: low income → more defaults ──────────────
    income_default = df_fee[df_fee['label'] == 2]['income_encoded'].mean()
    income_ontime  = df_fee[df_fee['label'] == 0]['income_encoded'].mean()
    # income_encoded: High=0, Medium=1, Low=2 — higher value = lower income
    print(f"\n  Intuition Check — Income vs Default:")
    print(f"    Avg income_encoded (On Time): {income_ontime:.2f}  (0=High, 2=Low)")
    print(f"    Avg income_encoded (Default): {income_default:.2f}  (0=High, 2=Low)")
    if income_default > income_ontime:
        print(f"  ✅ Defaulters tend to have lower income — makes sense!")
    else:
        print(f"  ⚠️  Income encoding not strongly predictive (may be small sample)")

    return passed


def run_verification():
    print("=" * 60)
    print("  KALNET AI-4 — Data Quality Verification (Rohith Koppu)")
    print("=" * 60)

    att_path = 'data/attendance_features.csv'
    fee_path = 'data/fee_features.csv'

    if not os.path.exists(att_path) or not os.path.exists(fee_path):
        print("\n❌ Feature CSVs not found.")
        print("   Run: python mohammed_kaif/feature_engineering.py")
        sys.exit(1)

    df_att = pd.read_csv(att_path)
    df_fee = pd.read_csv(fee_path)

    att_ok = check_attendance_features(df_att)
    fee_ok = check_fee_features(df_fee)

    print("\n" + "=" * 60)
    if att_ok and fee_ok:
        print("  ✅ ALL CHECKS PASSED — Features are ready for model training")
        print("  → Hand off to Om (attendance) and Are (fee) to train models")
    else:
        print("  ❌ SOME CHECKS FAILED — Review issues above before training")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
