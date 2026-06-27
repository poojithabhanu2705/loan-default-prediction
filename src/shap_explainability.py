"""
SHAP Explainability Analysis for Loan Default Prediction

This script uses SHAP (SHapley Additive exPlanations) to interpret the best-performing model:
- Trains a Logistic Regression model with class weights
- Generates SHAP values to explain predictions
- Creates a summary plot showing feature importance
- Identifies top factors influencing loan default risk
"""

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import shap
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def load_data(root: Path):
    """Load preprocessed training and test data."""
    X_train = pd.read_csv(root / 'X_train.csv')
    X_test = pd.read_csv(root / 'X_test.csv')
    y_train = pd.read_csv(root / 'y_train.csv')['target']
    y_test = pd.read_csv(root / 'y_test.csv')['target']
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train the best-performing model: Logistic Regression with class weights."""
    print("Training Logistic Regression with class_weight='balanced'...")
    model = LogisticRegression(
        solver='saga',
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    return acc, roc_auc


def generate_shap_values(model, X_train, X_test, n_background=100):
    """Generate SHAP values using KernelExplainer for interpretability."""
    print(f"\nGenerating SHAP values using {n_background} background samples...")
    
    # Use a sample of training data as background
    background = X_train.sample(n=min(n_background, len(X_train)), random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # Generate SHAP values for test set
    shap_values = explainer.shap_values(X_test)
    print("SHAP values generated!")
    
    return explainer, shap_values, background


def create_summary_plot(explainer, shap_values, X_test, output_dir: Path):
    """Create SHAP summary plot."""
    print("\nGenerating SHAP summary plot...")
    
    # For binary classification, take the positive class SHAP values
    if isinstance(shap_values, list):
        sv = shap_values[1]  # Positive class
    else:
        sv = shap_values
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(sv, X_test, plot_type='bar', show=False)
    plt.title('SHAP Summary Plot - Feature Importance for Loan Default Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_dir / 'shap_summary_plot.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary plot to {path}")
    
    return sv


def create_feature_importance_chart(shap_values, X_test, output_dir: Path):
    """Create a detailed feature importance chart from SHAP values."""
    print("\nGenerating feature importance chart...")
    
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean |SHAP|': pd.Series(shap_values.mean(axis=0), index=X_test.columns)
    }).sort_values('Mean |SHAP|', ascending=True).tail(15)
    
    plt.figure(figsize=(10, 8))
    plt.barh(mean_abs_shap['Feature'], mean_abs_shap['Mean |SHAP|'], color='steelblue')
    plt.xlabel('Mean |SHAP value|', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top 15 Features by SHAP Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_dir / 'feature_importance_chart.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance chart to {path}")
    
    return mean_abs_shap


def identify_top_factors(shap_values, X_test, n_top=10):
    """Identify and report the top factors influencing loan default risk."""
    print(f"\n{'='*80}")
    print("TOP FACTORS INFLUENCING LOAN DEFAULT RISK")
    print('='*80)
    
    # Calculate mean absolute SHAP values
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean |SHAP|': pd.Series(shap_values.mean(axis=0), index=X_test.columns)
    }).sort_values('Mean |SHAP|', ascending=False)
    
    print(f"\nTop {n_top} Features by SHAP Importance:\n")
    for idx, (i, row) in enumerate(feature_importance.head(n_top).iterrows(), 1):
        print(f"{idx:2d}. {row['Feature']:40s} | SHAP Impact: {row['Mean |SHAP|']:.6f}")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print('='*80)
    print("""
The SHAP values indicate how much each feature contributes to pushing the model's 
prediction away from the base value (towards default or away from default).

Higher SHAP mean values indicate stronger influence on model predictions. These 
features are the most important for understanding loan default risk in this dataset.

Key insights:
- Features associated with loan terms (interest rate, installment) are typically high impact
- Borrower credit profile features (FICO scores, delinquencies) are critical
- Financial health indicators (debt-to-income, annual income) drive default predictions
- Historical payment behavior and account diversity matter
    """)
    print('='*80)


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / 'outputs'
    output_dir = data_dir
    
    # Verify directories
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist!")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_dir)
    print(f"Loaded data: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Generate SHAP values
    explainer, shap_values_list, background = generate_shap_values(model, X_train, X_test, n_background=100)
    
    # Create visualizations
    shap_values = create_summary_plot(explainer, shap_values_list, X_test, output_dir)
    feature_importance = create_feature_importance_chart(shap_values, X_test, output_dir)
    
    # Identify and report top factors
    identify_top_factors(shap_values, X_test, n_top=15)
    
    # Save feature importance to CSV
    csv_path = output_dir / 'shap_feature_importance.csv'
    feature_importance.to_csv(csv_path, index=False)
    print(f"\nSaved feature importance table to {csv_path}")
    
    print("\n✓ SHAP analysis complete!")


if __name__ == '__main__':
    main()
