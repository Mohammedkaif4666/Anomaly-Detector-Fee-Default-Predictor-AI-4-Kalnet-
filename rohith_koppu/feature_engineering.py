import pandas as pd
import numpy as np
import os

def engineer_attendance_features():
    print("Engineering attendance features...")
    df = pd.read_csv("mohammed_kaif/raw_attendance.csv")
    
    # Calculate attendance_rate
    features = df.groupby('student_id')['is_present'].mean().reset_index(name='attendance_rate')
    
    # Get labels
    labels = df.groupby('student_id')['is_anomalous'].first().reset_index()
    features = features.merge(labels, on='student_id')
    
    # Calculate longest_absence_streak
    def get_longest_streak(group):
        is_absent = (group['is_present'] == 0).astype(int)
        # Identify streaks by looking at changes
        streaks = is_absent * (is_absent.groupby((is_absent != is_absent.shift()).cumsum()).cumcount() + 1)
        return streaks.max()
        
    streaks = df.groupby('student_id').apply(get_longest_streak).reset_index(name='longest_absence_streak')
    features = features.merge(streaks, on='student_id')
    
    # Calculate absence_in_last_30_days
    max_day = df['day'].max()
    last_30 = df[df['day'] > max_day - 30]
    absences_30 = last_30.groupby('student_id').apply(lambda x: (x['is_present'] == 0).sum()).reset_index(name='absence_in_last_30_days')
    features = features.merge(absences_30, on='student_id')
    
    # Calculate day_of_week_variance
    # Assign days of week 0-4 (Mon-Fri)
    df['dow'] = (df['day'] - 1) % 5
    dow_attendance = df.groupby(['student_id', 'dow'])['is_present'].mean().reset_index()
    dow_var = dow_attendance.groupby('student_id')['is_present'].var().fillna(0).reset_index(name='day_of_week_variance')
    features = features.merge(dow_var, on='student_id')
    
    os.makedirs("rohith_koppu", exist_ok=True)
    features.to_csv("rohith_koppu/attendance_features.csv", index=False)
    print("Saved rohith_koppu/attendance_features.csv")

def engineer_fee_features():
    print("Engineering fee features...")
    df = pd.read_csv("mohammed_kaif/raw_fee.csv")
    
    records = []
    # Process per student to get the features at the end of term 2 / start of term 3
    # The label is `will_default`
    for student_id, group in df.groupby("student_id"):
        # We assume the prediction happens before term 3 or at the end of term 2
        # Let's aggregate past behavior
        past_terms = group[group['term'] <= 2]
        
        # total_outstanding
        total_outstanding = (past_terms['fee_amount'] - past_terms['amount_paid']).sum()
        
        # days_since_last_payment
        # we can use the max days_late from term 2 as a proxy, or total days late
        if not past_terms.empty:
            days_since_last_payment = past_terms['days_late'].iloc[-1]
            # previous_term_status 0=on_time, 1=late, 2=default
            prev_status = past_terms['status'].iloc[-1]
            status_map = {"paid": 0, "late": 1, "default": 2}
            previous_term_status = status_map.get(prev_status, 0)
        else:
            days_since_last_payment = 0
            previous_term_status = 0
            
        income_encoded = {"High": 0, "Medium": 1, "Low": 2}[group['family_income_bracket'].iloc[0]]
        
        records.append({
            "student_id": student_id,
            "days_since_last_payment": days_since_last_payment,
            "previous_term_status": previous_term_status,
            "total_outstanding": total_outstanding,
            "income_encoded": income_encoded,
            "transport_user": group['transport_user'].iloc[0],
            "sibling_count": group['sibling_count'].iloc[0],
            "will_default": group['will_default'].iloc[0]
        })
        
    features = pd.DataFrame(records)
    features.to_csv("rohith_koppu/fee_features.csv", index=False)
    print("Saved rohith_koppu/fee_features.csv")

if __name__ == "__main__":
    engineer_attendance_features()
    engineer_fee_features()
    print("Feature engineering complete.")
