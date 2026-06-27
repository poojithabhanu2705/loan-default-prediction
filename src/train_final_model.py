"""
Train and save the best loan default prediction model.

This script trains a Logistic Regression model with class weights
and saves it along with the preprocessing pipeline for the Streamlit app.
"""

import warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def main():
    root = Path(__file__).resolve().parent.parent / 'outputs'
    
    # Load preprocessed data
    X_train = pd.read_csv(root / 'X_train.csv')
    X_test = pd.read_csv(root / 'X_test.csv')
    y_train = pd.read_csv(root / 'y_train.csv')['target']
    y_test = pd.read_csv(root / 'y_test.csv')['target']
    
    print("Training Logistic Regression with class_weight='balanced'...")
    model = LogisticRegression(
        solver='saga',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"Model Performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    # Save model
    model_path = root / 'loan_default_model.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save feature names
    feature_names_path = root / 'feature_names.joblib'
    joblib.dump(X_train.columns.tolist(), feature_names_path)
    print(f"Feature names saved to {feature_names_path}")


if __name__ == '__main__':
    main()
