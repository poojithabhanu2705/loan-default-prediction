"""
Lending Club Loan Dataset Analysis and Cleaning Script

This script performs the following tasks:
1. Load the accepted loans dataset using pandas
2. Display dataset shape, columns, missing values statistics, and loan_status distribution
3. Filter data to keep only 'Fully Paid', 'Charged Off', and 'Default' statuses
4. Create a binary target column (0 = Fully Paid, 1 = Charged Off or Default)
5. Generate a comprehensive data summary report
6. Save the cleaned dataset as 'cleaned_loans.csv'
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_dataset(file_path):
    """
    Load the Lending Club dataset from a CSV or compressed CSV file.
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✓ Dataset loaded successfully!")
    return df


def display_initial_statistics(df):
    """
    Display initial dataset statistics: shape, columns, missing values, and loan_status distribution.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
    """
    print("\n" + "="*80)
    print("INITIAL DATASET OVERVIEW")
    print("="*80)
    
    # Dataset shape
    print(f"\n1. Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Columns
    print(f"\n2. Column Names ({len(df.columns)} total):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Missing values
    print(f"\n3. Missing Values Analysis:")
    missing_info = df.isnull().sum()
    missing_pct = (missing_info / len(df)) * 100
    
    if missing_info.sum() == 0:
        print("   ✓ No missing values found!")
    else:
        missing_df = pd.DataFrame({
            'Column': missing_info.index,
            'Missing_Count': missing_info.values,
            'Percentage': missing_pct.values
        }).sort_values('Missing_Count', ascending=False)
        
        items_to_display = missing_df[missing_df['Missing_Count'] > 0]
        if len(items_to_display) > 0:
            print(items_to_display.to_string(index=False))
        else:
            print("   ✓ No significant missing values!")
    
    # Loan status distribution
    print(f"\n4. Loan Status Distribution:")
    if 'loan_status' in df.columns:
        status_counts = df['loan_status'].value_counts()
        status_pct = (status_counts / len(df)) * 100
        
        status_df = pd.DataFrame({
            'Status': status_counts.index,
            'Count': status_counts.values,
            'Percentage': status_pct.values
        })
        print(status_df.to_string(index=False))
    else:
        print("   ⚠ 'loan_status' column not found in dataset!")


def filter_loan_statuses(df):
    """
    Filter the dataset to keep only 'Fully Paid', 'Charged Off', and 'Default' statuses.
    
    Args:
        df (pd.DataFrame): The original dataset
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    target_statuses = ['Fully Paid', 'Charged Off', 'Default']
    
    print(f"\n" + "="*80)
    print("FILTERING LOAN STATUSES")
    print("="*80)
    print(f"\nTarget statuses: {target_statuses}")
    print(f"Original dataset size: {len(df)} rows")
    
    # Filter the dataframe
    df_filtered = df[df['loan_status'].isin(target_statuses)].copy()
    
    print(f"Filtered dataset size: {len(df_filtered)} rows")
    print(f"Rows removed: {len(df) - len(df_filtered)} ({((len(df) - len(df_filtered)) / len(df) * 100):.2f}%)")
    
    # Display filtered status distribution
    print("\nFiltered Loan Status Distribution:")
    status_counts = df_filtered['loan_status'].value_counts()
    status_pct = (status_counts / len(df_filtered)) * 100
    
    for status in target_statuses:
        count = status_counts.get(status, 0)
        pct = status_pct.get(status, 0)
        print(f"   {status:20s}: {count:7d} ({pct:6.2f}%)")
    
    return df_filtered


def create_target_column(df):
    """
    Create a binary target column for loan default prediction.
    
    Mapping:
    - Fully Paid = 0 (Good loan)
    - Charged Off or Default = 1 (Bad loan - Default)
    
    Args:
        df (pd.DataFrame): The dataset with loan_status column
        
    Returns:
        pd.DataFrame: Dataset with new 'target' column
    """
    print(f"\n" + "="*80)
    print("CREATING TARGET COLUMN")
    print("="*80)
    
    # Create target mapping
    target_mapping = {
        'Fully Paid': 0,
        'Charged Off': 1,
        'Default': 1
    }
    
    df['target'] = df['loan_status'].map(target_mapping)
    
    print("\nTarget Column Mapping:")
    print("   Fully Paid         → 0 (Good loan)")
    print("   Charged Off/Default → 1 (Bad loan - Default)")
    
    # Target distribution
    print("\nTarget Column Distribution:")
    target_counts = df['target'].value_counts().sort_index()
    target_pct = (target_counts / len(df)) * 100
    
    print(f"   Target 0 (Good loans - Fully Paid):     {target_counts[0]:7d} ({target_pct[0]:6.2f}%)")
    print(f"   Target 1 (Bad loans - Default):          {target_counts[1]:7d} ({target_pct[1]:6.2f}%)")
    print(f"   Class imbalance ratio: {target_pct[1]/target_pct[0]:.3f}")
    
    return df


def generate_data_summary_report(df, output_path):
    """
    Generate a comprehensive data summary report and save it to a file.
    
    Args:
        df (pd.DataFrame): The cleaned dataset
        output_path (str): Path to save the summary report
    """
    print(f"\n" + "="*80)
    print("GENERATING DATA SUMMARY REPORT")
    print("="*80)
    
    report = []
    report.append("="*80)
    report.append("LENDING CLUB LOAN DATASET - DATA SUMMARY REPORT")
    report.append("="*80)
    
    # Dataset overview
    report.append("\n1. DATASET OVERVIEW")
    report.append("-" * 80)
    report.append(f"Total records: {len(df):,}")
    report.append(f"Total features: {len(df.columns)}")
    report.append(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    report.append("\n2. DATA TYPES")
    report.append("-" * 80)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        report.append(f"   {dtype}: {count} columns")
    
    # Missing values
    report.append("\n3. MISSING VALUES")
    report.append("-" * 80)
    missing_counts = df.isnull().sum()
    if missing_counts.sum() == 0:
        report.append("   ✓ No missing values in the cleaned dataset!")
    else:
        for col in missing_counts[missing_counts > 0].index:
            report.append(f"   {col}: {missing_counts[col]} ({missing_counts[col]/len(df)*100:.2f}%)")
    
    # Loan status and target distribution
    report.append("\n4. LOAN STATUS DISTRIBUTION")
    report.append("-" * 80)
    status_dist = df['loan_status'].value_counts()
    for status in status_dist.index:
        report.append(f"   {status}: {status_dist[status]:,} ({status_dist[status]/len(df)*100:.2f}%)")
    
    # Target distribution
    report.append("\n5. TARGET VARIABLE DISTRIBUTION")
    report.append("-" * 80)
    target_dist = df['target'].value_counts().sort_index()
    class_names = {0: "Fully Paid (Good)", 1: "Default (Bad)"}
    for target_val in target_dist.index:
        report.append(f"   {target_val} - {class_names[target_val]:30s}: {target_dist[target_val]:,} ({target_dist[target_val]/len(df)*100:.2f}%)")
    
    # Numerical features summary
    report.append("\n6. NUMERICAL FEATURES SUMMARY")
    report.append("-" * 80)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    report.append(f"Number of numerical features: {len(numerical_cols)}")
    if len(numerical_cols) > 0:
        stats_df = df[numerical_cols].describe().T
        report.append("\n" + stats_df.to_string())
    
    # Categorical features summary
    report.append("\n7. CATEGORICAL FEATURES SUMMARY")
    report.append("-" * 80)
    categorical_cols = df.select_dtypes(include=['object']).columns
    report.append(f"Number of categorical features: {len(categorical_cols)}")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        report.append(f"   {col}: {unique_count} unique values")
    
    report.append("\n" + "="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    # Write report to file
    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Report saved to: {output_path}")
    print("\n" + report_text)


def save_cleaned_dataset(df, output_path):
    """
    Save the cleaned dataset to a CSV file.
    
    Args:
        df (pd.DataFrame): The cleaned dataset
        output_path (str): Path to save the cleaned dataset
    """
    print(f"\n" + "="*80)
    print("SAVING CLEANED DATASET")
    print("="*80)
    
    df.to_csv(output_path, index=False)
    file_size = os.path.getsize(output_path) / 1024**2
    print(f"✓ Cleaned dataset saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Records: {len(df):,}")
    print(f"  Features: {len(df.columns)}")
    print(f"  File size: {file_size:.2f} MB")


def main():
    """
    Main execution function that orchestrates the entire workflow.
    """
    print("\n" + "="*80)
    print("LENDING CLUB LOAN DATASET ANALYSIS AND CLEANING")
    print("="*80)
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "data" / "accepted_2007_to_2018Q4.csv.gz"
    output_csv = project_root / "cleaned_loans.csv"
    output_report = project_root / "data_summary_report.txt"
    
    # Verify the data file exists
    if not data_file.exists():
        print(f"\n✗ Error: Data file not found at {data_file}")
        print("  Please ensure the dataset is in the correct location.")
        return
    
    # Step 1: Load the dataset
    df = load_dataset(str(data_file))
    
    # Step 2: Display initial statistics
    display_initial_statistics(df)
    
    # Step 3: Filter loan statuses
    df = filter_loan_statuses(df)
    
    # Step 4: Create target column
    df = create_target_column(df)
    
    # Step 5: Generate data summary report
    generate_data_summary_report(df, str(output_report))
    
    # Step 6: Save cleaned dataset
    save_cleaned_dataset(df, str(output_csv))
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput files created:")
    print(f"  1. Cleaned dataset: cleaned_loans.csv")
    print(f"  2. Summary report: data_summary_report.txt")
    print("\n")


if __name__ == "__main__":
    main()
