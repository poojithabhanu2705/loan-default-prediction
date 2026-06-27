"""
Compare baseline, class-weighted, and SMOTE-balanced models on loan default data.

This script loads preprocessed train/test sets and trains two model families:
- Logistic Regression
- Random Forest

It evaluates each approach on the original test set using:
- Accuracy
- Precision, recall, F1 for the positive class
- ROC AUC
- Confusion matrix

The goal is to compare performance before and after balancing and select the best method.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split


def load_data(root: Path):
    X_train = pd.read_csv(root / 'X_train.csv')
    X_test = pd.read_csv(root / 'X_test.csv')
    y_train = pd.read_csv(root / 'y_train.csv')['target']
    y_test = pd.read_csv(root / 'y_test.csv')['target']
    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(report)

    return {
        'name': name,
        'accuracy': acc,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'report': report,
    }


def train_compare(X_train, X_test, y_train, y_test):
    results = []

    classifiers = {
        'Logistic Regression': LogisticRegression(
            solver='saga', max_iter=1000, random_state=42, n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    for model_name, base_model in classifiers.items():
        # Baseline
        base_model.fit(X_train, y_train)
        results.append(evaluate_model(f"{model_name} - Baseline", base_model, X_test, y_test))

        # Class weights
        weighted_model = None
        if model_name == 'Random Forest':
            weighted_model = RandomForestClassifier(
                n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
            )
        else:
            weighted_model = LogisticRegression(
                solver='saga', max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1
            )
        weighted_model.fit(X_train, y_train)
        results.append(evaluate_model(f"{model_name} - Class Weight", weighted_model, X_test, y_test))

        # SMOTE
        smote_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', base_model.__class__(**{
                'random_state': 42,
                **({'n_estimators': 200} if model_name == 'Random Forest' else {'solver': 'saga', 'max_iter': 1000})
            }))
        ])
        smote_pipeline.fit(X_train, y_train)
        results.append(evaluate_model(f"{model_name} - SMOTE", smote_pipeline, X_test, y_test))

    return results


def summarize_results(results):
    df = pd.DataFrame([
        {'model': r['name'], 'accuracy': r['accuracy'], 'roc_auc': r['roc_auc']}
        for r in results
    ])
    print('\n=== Summary ===')
    print(df.sort_values(['roc_auc', 'accuracy'], ascending=False).to_string(index=False))
    return df


def main():
    root = Path(__file__).resolve().parent.parent / 'outputs'
    X_train, X_test, y_train, y_test = load_data(root)

    print(f"Loaded X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Loaded y_train: {y_train.shape}, y_test: {y_test.shape}")

    results = train_compare(X_train, X_test, y_train, y_test)
    summarize_results(results)


if __name__ == '__main__':
    main()
