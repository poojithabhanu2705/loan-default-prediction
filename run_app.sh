#!/bin/bash
# Script to run the Loan Default Prediction Streamlit Application

# Make sure we're in the project directory
cd "$(dirname "$0")" || exit

# Activate virtual environment
source venv/bin/activate

# Train and save the model if it doesn't exist
echo "Checking for trained model..."
if [ ! -f "outputs/loan_default_model.joblib" ]; then
    echo "Training final model..."
    python3 << 'EOF'
import warnings
warnings.filterwarnings('ignore')
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

root = Path('outputs')

X_train = pd.read_csv(root / 'X_train.csv')
X_test = pd.read_csv(root / 'X_test.csv')
y_train = pd.read_csv(root / 'y_train.csv')['target']
y_test = pd.read_csv(root / 'y_test.csv')['target']

print("Training Logistic Regression with class_weight='balanced'...")
model = LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Model Performance: Accuracy={acc:.4f}, ROC AUC={roc_auc:.4f}")

model_path = root / 'loan_default_model.joblib'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

feature_names_path = root / 'feature_names.joblib'
joblib.dump(X_train.columns.tolist(), feature_names_path)
print(f"Feature names saved")
EOF
else
    echo "Model found!"
fi

echo ""
echo "Starting Streamlit application..."
echo "Open your browser to: http://localhost:8501"
echo ""
streamlit run app/streamlit_app.py
