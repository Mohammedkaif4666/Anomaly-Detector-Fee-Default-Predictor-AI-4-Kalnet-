import pandas as pd
import numpy as np
import os

def engineer_features():
    print("Engineering features...")
    
    # Load raw data
    if not os.path.exists('data/attendance_raw.csv') or not os.path.exists('data/fee_raw.csv'):
        print("Raw data not found. Run data_generator.py first.")
        return

    df_attendance = pd.read_csv('data/attendance_raw.csv')
    df_fee = pd.read_csv('data/fee_raw.csv')
    df_labels = pd.read_csv('data/student_labels.csv')
    
    # --- ATTENDANCE FEATURES ---
    print("Processing attendance features...")
    
    def get_attendance_features(group):
        is_present = group['is_present'].values
        
        # 1. Attendance rate
        rate = is_present.mean()
        
        # 2. Longest absence streak
        # Convert to string and split by '1' to find sequences of '0'
        absences = "".join(is_present.astype(str))
        streaks = [len(s) for s in absences.split('1')]
        longest_streak = max(streaks) if streaks else 0
        
        # 3. Absence in last 30 days
        last_30 = is_present[-30:]
        absence_last_30 = len(last_30) - sum(last_30)
        
        # 4. Day of week variance
        group['date'] = pd.to_datetime(group['date'])
        group['day_of_week'] = group['date'].dt.dayofweek
        dow_attendance = group.groupby('day_of_week')['is_present'].mean()
        dow_variance = dow_attendance.var()
        
        return pd.Series({
            'attendance_rate': rate,
            'longest_absence_streak': longest_streak,
            'absence_in_last_30_days': absence_last_30,
            'day_of_week_variance': dow_variance
        })

    attendance_features = df_attendance.groupby('student_id').apply(get_attendance_features).reset_index()
    
    # Merge with labels for training
    attendance_features = attendance_features.merge(df_labels, on='student_id')
    
    # --- FEE FEATURES ---
    print("Processing fee features...")
    
    # income_encoded H=0 M=1 L=2
    income_map = {'High': 0, 'Medium': 1, 'Low': 2}
    
    # We want features per student, maybe focusing on the latest term's risk
    # But the requirement says "Engineer per-student fee features"
    
    # For fee prediction, we often predict the NEXT term's status based on current/past
    # Let's pivot the fee data to get previous status
    
    fee_features_list = []
    for stu_id, group in df_fee.groupby('student_id'):
        group = group.sort_values('term')
        
        # Use Term 3 as the "target" status, and Terms 1-2 for features
        # Or just use the overall profile features
        
        latest_record = group.iloc[-1]
        prev_records = group.iloc[:-1]
        
        avg_days_late = prev_records['days_since_last_payment'].mean()
        prev_status = prev_records['status'].max() # 0, 1, or 2
        
        fee_features_list.append({
            'student_id': stu_id,
            'days_since_last_payment': latest_record['days_since_last_payment'],
            'previous_term_status': prev_status,
            'total_outstanding': latest_record['total_outstanding'],
            'income_encoded': income_map[latest_record['family_income_bracket']],
            'transport_user': latest_record['transport_user'],
            'sibling_count': latest_record['sibling_count'],
            'label': latest_record['status'] # We'll predict this status (0/1/2)
        })
        
    fee_features = pd.DataFrame(fee_features_list)
    
    # Save processed features
    attendance_features.to_csv('data/attendance_features.csv', index=False)
    fee_features.to_csv('data/fee_features.csv', index=False)
    
    print("Features saved to data/attendance_features.csv and data/fee_features.csv")

if __name__ == "__main__":
    engineer_features()
