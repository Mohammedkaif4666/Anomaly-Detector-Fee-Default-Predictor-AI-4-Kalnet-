import pandas as pd
import numpy as np

def run_data_quality_checks():
    print("Running data quality checks...")
    
    # Check attendance features
    att_df = pd.read_csv("rohith_koppu/attendance_features.csv")
    
    assert att_df.isnull().sum().sum() == 0, "NaN values found in attendance_features.csv"
    assert att_df['attendance_rate'].between(0, 1).all(), "attendance_rate out of range [0, 1]"
    assert att_df['longest_absence_streak'].between(0, 200).all(), "longest_absence_streak out of bounds"
    
    att_labels = att_df['is_anomalous'].value_counts(normalize=True)
    print(f"Attendance Label Distribution:\n{att_labels}")
    
    # Check fee features
    fee_df = pd.read_csv("rohith_koppu/fee_features.csv")
    
    assert fee_df.isnull().sum().sum() == 0, "NaN values found in fee_features.csv"
    assert fee_df['previous_term_status'].isin([0, 1, 2]).all(), "previous_term_status invalid"
    assert fee_df['income_encoded'].isin([0, 1, 2]).all(), "income_encoded invalid"
    
    fee_labels = fee_df['will_default'].value_counts(normalize=True)
    print(f"Fee Label Distribution:\n{fee_labels}")
    
    print("All data quality checks passed successfully!")

if __name__ == "__main__":
    run_data_quality_checks()
