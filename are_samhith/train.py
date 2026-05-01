import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report

def train_fee_model():
    print("Training Fee Default Predictor...")
    
    # Load features
    if not os.path.exists('data/fee_features.csv'):
        print("Features not found. Run feature_engineering.py first.")
        return

    df = pd.read_csv('data/fee_features.csv')
    X = df.drop(['student_id', 'label'], axis=1)
    y = df['label']
    
    # 80/20 split with stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Gradient Boosting
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Calculate recall for 'Default' class (label 2)
    # Note: status 0=On time, 1=Late, 2=Default
    recall_default = recall_score(y_test, y_pred, labels=[2], average='macro')
    
    print(f"Recall for Default Class: {recall_default:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Feature Importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importances:")
    print(importances)
    
    # Save model
    os.makedirs('models/fee_predictor', exist_ok=True)
    joblib.dump(model, 'models/fee_predictor/model.pkl')
    print("Model saved to models/fee_predictor/model.pkl")

def predict_default_risk(df):
    """
    Function to be used in production/API
    Returns {student_id, default_probability, risk_category}
    """
    model = joblib.load('models/fee_predictor/model.pkl')
    
    X = df.drop(['student_id'], axis=1, errors='ignore')
    
    # Get probabilities for all classes
    probs = model.predict_proba(X)
    # Probability of class 2 (Default)
    default_probs = probs[:, 2]
    
    results = []
    for i, stu_id in enumerate(df['student_id']):
        prob = default_probs[i]
        category = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
        results.append({
            'student_id': stu_id,
            'default_probability': round(float(prob), 2),
            'risk_category': category
        })
    return results

if __name__ == "__main__":
    train_fee_model()
