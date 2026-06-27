"""
Preprocessing pipeline for the Lending Club loan default dataset.

This script performs the following steps:
- Load the cleaned loan dataset
- Drop high-cardinality or uninformative text/date fields
- Impute missing values for numeric and categorical features
- One-hot encode categorical features
- Scale numeric features using StandardScaler
- Split into stratified train/test sets
- Save X_train, X_test, y_train, and y_test to CSV files
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_names(column_transformer: ColumnTransformer):
    """Recover transformed feature names from a fitted ColumnTransformer."""
    if hasattr(column_transformer, 'get_feature_names_out'):
        return list(column_transformer.get_feature_names_out())

    feature_names = []
    for name, transformer, cols in column_transformer.transformers_:
        if name == 'remainder':
            continue

        if isinstance(transformer, Pipeline):
            transformer = transformer.steps[-1][1]

        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)
    return feature_names


def build_preprocessing_pipeline(numeric_cols, categorical_cols):
    """Build a sklearn preprocessing pipeline for numeric and categorical features."""
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_cols),
        ('categorical', categorical_pipeline, categorical_cols),
    ], remainder='drop')

    return preprocessor


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the cleaned loan dataset from CSV."""
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")
    return df


def select_features(df: pd.DataFrame):
    """Select feature columns for model training and preprocessing."""
    drop_columns = [
        'id', 'member_id', 'url', 'desc', 'title', 'zip_code',
        'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d',
        'last_credit_pull_d', 'sec_app_earliest_cr_line', 'hardship_start_date',
        'hardship_end_date', 'payment_plan_start_date', 'debt_settlement_flag_date',
        'settlement_date', 'loan_status', 'hardship_reason',
    ]

    feature_df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore')

    numeric_cols = feature_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'target']

    # Drop numeric columns that contain no observed values in the full dataset.
    all_null_numeric = [c for c in numeric_cols if feature_df[c].isna().all()]
    numeric_cols = [c for c in numeric_cols if c not in all_null_numeric]
    if all_null_numeric:
        print(f"Dropping {len(all_null_numeric)} numeric columns with no observed values: {all_null_numeric}")

    categorical_cols = feature_df.select_dtypes(include=['object', 'string']).columns.tolist()

    # Keep only relevant categorical columns and drop high-cardinality or uninformative fields.
    keep_categories = [
        'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
        'verification_status', 'purpose', 'addr_state', 'initial_list_status',
        'application_type', 'verification_status_joint', 'hardship_flag',
        'hardship_status', 'disbursement_method', 'debt_settlement_flag',
        'settlement_status',
    ]
    categorical_cols = [c for c in categorical_cols if c in keep_categories]

    print(f"Selected {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features.")
    print(f"Numeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")

    return feature_df, numeric_cols, categorical_cols


def save_dataframe(df: pd.DataFrame, file_path: Path):
    """Save a DataFrame to CSV with no index."""
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path} ({df.shape[0]:,} rows × {df.shape[1]:,} columns)")


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / 'cleaned_loans.csv'
    output_dir = project_root / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_file)
    feature_df, numeric_cols, categorical_cols = select_features(df)

    X = feature_df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print(f"Split data: {len(X_train):,} train rows, {len(X_test):,} test rows.")

    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)
    print(f"Transformed feature count: {len(feature_names)}")

    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    save_dataframe(X_train_df, output_dir / 'X_train.csv')
    save_dataframe(X_test_df, output_dir / 'X_test.csv')
    save_dataframe(pd.DataFrame({'target': y_train.reset_index(drop=True)}), output_dir / 'y_train.csv')
    save_dataframe(pd.DataFrame({'target': y_test.reset_index(drop=True)}), output_dir / 'y_test.csv')

    joblib.dump(preprocessor, output_dir / 'preprocessing_pipeline.joblib')
    print(f"Saved preprocessing pipeline to {output_dir / 'preprocessing_pipeline.joblib'}")

    print('\nPreprocessing complete!')


if __name__ == '__main__':
    main()
