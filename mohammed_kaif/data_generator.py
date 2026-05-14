import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_data():
    print("=" * 60)
    print("  KALNET AI-4 — Data Generator (Mohammed Kaif)")
    print("=" * 60)
    
    num_students = 500
    num_days = 200
    
    student_ids = [f"STU_{i:03d}" for i in range(1, num_students + 1)]
    
    # Assign classes for display
    classes = ["6A","6B","7A","7B","8A","8B","9A","9B","10A","10B"]
    student_classes = {sid: classes[i % len(classes)] for i, sid in enumerate(student_ids)}
    
    # --- ATTENDANCE DATA GENERATION ---
    print("\n[1/3] Generating attendance data...")
    attendance_records = []
    student_labels = []
    
    # 86% normal, 14% anomalous
    num_anomalous = int(num_students * 0.14)
    anomalous_students = set(np.random.choice(student_ids, num_anomalous, replace=False))
    
    start_date = datetime(2023, 8, 1)
    
    for stu_id in student_ids:
        is_anomalous = stu_id in anomalous_students
        student_labels.append({
            'student_id': stu_id,
            'class': student_classes[stu_id],
            'is_anomaly': 1 if is_anomalous else 0
        })
        
        if not is_anomalous:
            base_rate = np.random.uniform(0.80, 0.97)
            attendance = np.random.binomial(1, base_rate, num_days)
        else:
            # Anomalous: normal for first ~150 days, then sudden drop
            base_rate = np.random.uniform(0.80, 0.97)
            attendance_early = np.random.binomial(1, base_rate, 150)
            drop_rate = np.random.uniform(0.20, 0.40)
            attendance_late = np.random.binomial(1, drop_rate, num_days - 150)
            attendance = np.concatenate([attendance_early, attendance_late])
            
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            if current_date.weekday() < 5:  # Skip weekends
                attendance_records.append({
                    'student_id': stu_id,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'is_present': int(attendance[day])
                })

    df_attendance_raw = pd.DataFrame(attendance_records)
    
    # --- FEE DATA GENERATION (PROBABILISTIC) ---
    # Now each student has a TENDENCY but each term outcome is probabilistic
    print("[2/3] Generating fee data (probabilistic per term)...")
    fee_records = []
    
    income_brackets = ['Low', 'Medium', 'High']
    
    for stu_id in student_ids:
        # Static student features
        income = np.random.choice(income_brackets, p=[0.3, 0.5, 0.2])
        transport = np.random.choice([0, 1], p=[0.4, 0.6])
        siblings = np.random.randint(0, 4)
        
        # Base tendency — influenced by income and siblings
        # Low income + more siblings → higher default tendency
        income_risk = {'Low': 0.15, 'Medium': 0.05, 'High': 0.01}[income]
        sibling_risk = siblings * 0.03
        base_default_chance = income_risk + sibling_risk
        
        # Student profile determines probability distributions per term
        rand_val = np.random.random()
        if rand_val < 0.05:
            # Defaulter tendency: high chance of default each term
            p_ontime, p_late, p_default = 0.05, 0.15, 0.80
        elif rand_val < 0.20:
            # Late payer tendency: moderate risk
            p_ontime, p_late, p_default = 0.30, 0.50, 0.20
        else:
            # On-time tendency: still has small risk
            p_ontime = 0.85 - base_default_chance
            p_late = 0.12
            p_default = 0.03 + base_default_chance
        
        # Normalize probabilities
        total = p_ontime + p_late + p_default
        p_ontime /= total
        p_late /= total
        p_default /= total
            
        for term in range(1, 4):
            # Each term is a random draw from the student's probability
            status = np.random.choice([0, 1, 2], p=[p_ontime, p_late, p_default])
            
            if status == 2:
                days_late = np.random.randint(45, 120)
                outstanding = np.random.randint(2000, 8000)
            elif status == 1:
                days_late = np.random.randint(5, 45)
                outstanding = np.random.randint(500, 3000)
            else:
                days_late = 0
                outstanding = 0
                
            fee_records.append({
                'student_id': stu_id,
                'term': term,
                'status': status,
                'days_since_last_payment': days_late,
                'family_income_bracket': income,
                'transport_user': transport,
                'sibling_count': siblings,
                'total_outstanding': outstanding
            })
            
    df_fee_raw = pd.DataFrame(fee_records)
    
    # Save
    print("[3/3] Saving to data/ folder...")
    os.makedirs('data', exist_ok=True)
    
    df_attendance_raw.to_csv('data/attendance_raw.csv', index=False)
    df_fee_raw.to_csv('data/fee_raw.csv', index=False)
    pd.DataFrame(student_labels).to_csv('data/student_labels.csv', index=False)
    
    # Print summary
    fee_dist = df_fee_raw['status'].value_counts().sort_index()
    print(f"\n✅ Attendance: {len(df_attendance_raw):,} records ({num_students} students × ~144 school days)")
    print(f"✅ Fee:        {len(df_fee_raw):,} records ({num_students} students × 3 terms)")
    print(f"   On-time: {fee_dist.get(0,0)} | Late: {fee_dist.get(1,0)} | Default: {fee_dist.get(2,0)}")
    print(f"   Anomalous students: {num_anomalous} / {num_students}")

if __name__ == "__main__":
    generate_data()
