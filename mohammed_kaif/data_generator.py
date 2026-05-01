import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_data():
    print("Generating raw data for 500 students...")
    num_students = 500
    num_days = 200
    
    student_ids = [f"STU_{i:03d}" for i in range(1, num_students + 1)]
    
    # --- ATTENDANCE DATA GENERATION ---
    attendance_records = []
    student_labels = [] # To keep track of who is anomalous for later evaluation
    
    # 86% normal, 14% anomalous
    num_anomalous = int(num_students * 0.14)
    anomalous_students = np.random.choice(student_ids, num_anomalous, replace=False)
    
    start_date = datetime(2023, 8, 1)
    
    for stu_id in student_ids:
        is_anomalous = stu_id in anomalous_students
        student_labels.append({'student_id': stu_id, 'is_anomaly': 1 if is_anomalous else 0})
        
        if not is_anomalous:
            # Normal: 80-97% attendance rate
            base_rate = np.random.uniform(0.80, 0.97)
            # Daily attendance with some randomness
            attendance = np.random.binomial(1, base_rate, num_days)
        else:
            # Anomalous: Sudden drops to 20-40%
            # Let's say for the first 150 days they are normal, then drop
            base_rate = np.random.uniform(0.80, 0.97)
            attendance_early = np.random.binomial(1, base_rate, 150)
            drop_rate = np.random.uniform(0.20, 0.40)
            attendance_late = np.random.binomial(1, drop_rate, num_days - 150)
            attendance = np.concatenate([attendance_early, attendance_late])
            
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            # Skip weekends (approximate)
            if current_date.weekday() < 5:
                attendance_records.append({
                    'student_id': stu_id,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'is_present': attendance[day]
                })

    df_attendance_raw = pd.DataFrame(attendance_records)
    
    # --- FEE DATA GENERATION ---
    # 500 students, 3 terms. 80% on time, 15% late, 5% default.
    fee_records = []
    
    income_brackets = ['Low', 'Medium', 'High']
    
    for stu_id in student_ids:
        # Static student features
        income = np.random.choice(income_brackets, p=[0.3, 0.5, 0.2])
        transport = np.random.choice([0, 1], p=[0.4, 0.6])
        siblings = np.random.randint(0, 4)
        
        # Determine if this student is a "defaulter" type (5% chance)
        # This makes the data more realistic - certain students are more likely to default
        rand_val = np.random.random()
        if rand_val < 0.05:
            profile = 'defaulter'
        elif rand_val < 0.20:
            profile = 'late_payer'
        else:
            profile = 'on_time'
            
        for term in range(1, 4):
            if profile == 'defaulter':
                status = 2 # Default
                days_late = np.random.randint(60, 120)
            elif profile == 'late_payer':
                status = 1 # Late
                days_late = np.random.randint(1, 30)
            else:
                status = 0 # On time
                days_late = 0
                
            fee_records.append({
                'student_id': stu_id,
                'term': term,
                'status': status,
                'days_since_last_payment': days_late,
                'family_income_bracket': income,
                'transport_user': transport,
                'sibling_count': siblings,
                'total_outstanding': np.random.randint(0, 5000) if status > 0 else 0
            })
            
    df_fee_raw = pd.DataFrame(fee_records)
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    
    df_attendance_raw.to_csv('data/attendance_raw.csv', index=False)
    df_fee_raw.to_csv('data/fee_raw.csv', index=False)
    pd.DataFrame(student_labels).to_csv('data/student_labels.csv', index=False)
    
    print(f"Generated {len(df_attendance_raw)} attendance records and {len(df_fee_raw)} fee records.")
    return df_attendance_raw, df_fee_raw

if __name__ == "__main__":
    generate_data()
