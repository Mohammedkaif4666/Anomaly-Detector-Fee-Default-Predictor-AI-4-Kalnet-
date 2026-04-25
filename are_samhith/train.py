import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score
from sklearn.utils.class_weight import compute_sample_weight

FEATURE_COLS = [
    'days_since_last_payment',
    'previous_term_status',
    'total_outstanding',
    'income_encoded',
    'transport_user',
    'sibling_count'
]

def train_model():
    print("Training Fee Default Predictor Model...")
    
    df = pd.read_csv("../rohith_koppu/fee_features.csv")
    X = df[FEATURE_COLS]
    y = df['will_default']
    
    # 80/20 split with stratify
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    
    # Compute sample weights to handle the 5% class imbalance and boost recall
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate
    # predict_proba for class 1
    probs = model.predict_proba(X_test)[:, 1]
    
    # We will use a custom threshold to hit the recall target
    # Lowering the threshold to 0.3 to catch more defaulters
    threshold = 0.3
    y_pred = (probs >= threshold).astype(int)
    
    recall = recall_score(y_test, y_pred)
    print(f"Model Recall (Class 1): {recall:.2f}")
    
    if recall >= 0.70:
        print("Success: Recall target met!")
    else:
        print("Warning: Recall target NOT met. Adjusting threshold might be needed.")
        
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nFeature Importances:")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(importances)
    
    # Save the model
    artifacts = {
        'model': model,
        'features': FEATURE_COLS
    }
    
    os.makedirs(os.path.dirname("fee_model.pkl") or ".", exist_ok=True)
    joblib.dump(artifacts, "fee_model.pkl")
    print("\nSaved fee_model.pkl")

def predict_default_risk(df):
    artifacts = joblib.load("fee_model.pkl")
    model = artifacts['model']
    features = artifacts['features']
    
    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    
    results = pd.DataFrame({
        'student_id': df['student_id'],
        'default_probability': np.round(probs, 2)
    })
    
    def get_category(prob):
        if prob > 0.6:
            return "High Risk"
        elif prob > 0.3:
            return "Medium Risk"
        return "Low Risk"
        
    results['risk_category'] = results['default_probability'].apply(get_category)
    return results

if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != 'are_samhith':
        try:
            os.chdir('are_samhith')
        except FileNotFoundError:
            pass
            
    train_model()
