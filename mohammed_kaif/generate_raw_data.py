import pandas as pd
import numpy as np
import os

def generate_attendance_data(num_students=500, num_days=200):
    print("Generating raw attendance data...")
    np.random.seed(42)
    
    # 86% normal (430 students), 14% anomalous (70 students)
    normal_students = 430
    anomalous_students = 70
    
    records = []
    
    for student_id in range(1, num_students + 1):
        is_anomalous = 1 if student_id > normal_students else 0
        
        if is_anomalous:
            # Anomalous: Good attendance first 100 days (80-97%), then sudden drop to 20-40%
            prob_first_half = np.random.uniform(0.80, 0.97)
            prob_second_half = np.random.uniform(0.20, 0.40)
            
            for day in range(1, num_days + 1):
                prob = prob_first_half if day <= 100 else prob_second_half
                is_present = np.random.choice([1, 0], p=[prob, 1 - prob])
                records.append({
                    "student_id": student_id,
                    "day": day,
                    "is_present": is_present,
                    "is_anomalous": is_anomalous
                })
        else:
            # Normal: Consistent 80-97%
            prob = np.random.uniform(0.80, 0.97)
            # Add some realistic variance (occasional 3-day absence streak)
            has_streak = np.random.random() < 0.1
            streak_start = np.random.randint(1, num_days - 3) if has_streak else -1
            
            for day in range(1, num_days + 1):
                if streak_start <= day <= streak_start + 2:
                    is_present = 0
                else:
                    is_present = np.random.choice([1, 0], p=[prob, 1 - prob])
                    
                records.append({
                    "student_id": student_id,
                    "day": day,
                    "is_present": is_present,
                    "is_anomalous": is_anomalous
                })
                
    df = pd.DataFrame(records)
    # Save inside mohammed_kaif directory
    os.makedirs("mohammed_kaif", exist_ok=True)
    df.to_csv("mohammed_kaif/raw_attendance.csv", index=False)
    print("Saved mohammed_kaif/raw_attendance.csv")

def generate_fee_data(num_students=500):
    print("Generating raw fee data...")
    np.random.seed(42)
    
    records = []
    
    for student_id in range(1, num_students + 1):
        family_income = np.random.choice(["High", "Medium", "Low"], p=[0.3, 0.5, 0.2])
        transport = np.random.choice([1, 0], p=[0.4, 0.6])
        siblings = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05])
        
        # 80% on time, 15% late, 5% default
        # Determine overall payment behavior
        behavior = np.random.choice(["on_time", "late", "default"], p=[0.80, 0.15, 0.05])
        
        for term in range(1, 4):
            # Term fee is 10000
            term_fee = 10000
            
            if behavior == "on_time":
                paid = term_fee
                days_late = 0
            elif behavior == "late":
                paid = term_fee
                days_late = np.random.randint(10, 45)
            else: # default
                if term < 3:
                    # Might pay early terms, default on later
                    paid = term_fee if np.random.random() > 0.5 else 0
                    days_late = 0 if paid == term_fee else np.random.randint(60, 100)
                else:
                    # Defaults on last term
                    paid = 0
                    days_late = np.random.randint(60, 100)
                    
            status = "paid" if paid == term_fee and days_late == 0 else ("late" if paid == term_fee else "default")
            
            records.append({
                "student_id": student_id,
                "term": term,
                "fee_amount": term_fee,
                "amount_paid": paid,
                "days_late": days_late,
                "status": status,
                "family_income_bracket": family_income,
                "transport_user": transport,
                "sibling_count": siblings,
                "will_default": 1 if behavior == "default" else 0
            })
            
    df = pd.DataFrame(records)
    df.to_csv("mohammed_kaif/raw_fee.csv", index=False)
    print("Saved mohammed_kaif/raw_fee.csv")

if __name__ == "__main__":
    generate_attendance_data()
    generate_fee_data()
    print("Data generation complete.")
