"""
Exploratory Data Analysis for the cleaned Lending Club loan dataset.

This script creates EDA visualizations for:
- target distribution
- interest rate vs default
- annual income vs default
- debt-to-income ratio vs default
- FICO score vs default
- correlation heatmap

All figures are saved to the root outputs/ folder.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def ensure_output_dir(output_dir: Path):
    """Create the output directory if it does not exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")


def load_cleaned_data(file_path: Path) -> pd.DataFrame:
    """Load the cleaned loan dataset."""
    print(f"Loading cleaned dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} records with {len(df.columns):,} columns.")
    return df


def plot_target_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot the binary target distribution."""
    plt.figure(figsize=(7, 5))
    sns.countplot(x='target', data=df, palette='pastel')
    plt.title('Target Distribution')
    plt.xlabel('Target (0 = Fully Paid, 1 = Default)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Fully Paid (0)', 'Default (1)'])
    plt.tight_layout()
    path = output_dir / 'target_distribution.png'
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Saved target distribution plot: {path}")


def plot_int_rate_vs_default(df: pd.DataFrame, output_dir: Path):
    """Plot interest rate distribution by default target."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='target', y='int_rate', data=df, palette='Set2')
    plt.title('Interest Rate by Loan Outcome')
    plt.xlabel('Target (0 = Fully Paid, 1 = Default)')
    plt.ylabel('Interest Rate (%)')
    plt.xticks([0, 1], ['Fully Paid', 'Default'])
    plt.tight_layout()
    path = output_dir / 'interest_rate_vs_default.png'
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Saved interest rate vs default plot: {path}")


def plot_annual_income_vs_default(df: pd.DataFrame, output_dir: Path):
    """Plot annual income distribution by default target."""
    subset = df[df['annual_inc'] > 0].copy()
    subset['annual_inc_log'] = np.log1p(subset['annual_inc'])

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='target', y='annual_inc_log', data=subset, palette='Set3')
    plt.title('Log Annual Income by Loan Outcome')
    plt.xlabel('Target (0 = Fully Paid, 1 = Default)')
    plt.ylabel('Log(Annual Income + 1)')
    plt.xticks([0, 1], ['Fully Paid', 'Default'])
    plt.tight_layout()
    path = output_dir / 'annual_income_vs_default.png'
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Saved annual income vs default plot: {path}")


def plot_dti_vs_default(df: pd.DataFrame, output_dir: Path):
    """Plot debt-to-income ratio distribution by default target."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='target', y='dti', data=df, palette='coolwarm')
    plt.title('Debt-to-Income Ratio by Loan Outcome')
    plt.xlabel('Target (0 = Fully Paid, 1 = Default)')
    plt.ylabel('Debt-to-Income Ratio (%)')
    plt.xticks([0, 1], ['Fully Paid', 'Default'])
    plt.tight_layout()
    path = output_dir / 'dti_vs_default.png'
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Saved DTI vs default plot: {path}")


def plot_fico_vs_default(df: pd.DataFrame, output_dir: Path):
    """Plot FICO score distribution by default target."""
    df = df.copy()
    df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2.0

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='target', y='fico_score', data=df, palette='muted')
    plt.title('Average FICO Score by Loan Outcome')
    plt.xlabel('Target (0 = Fully Paid, 1 = Default)')
    plt.ylabel('Average FICO Score')
    plt.xticks([0, 1], ['Fully Paid', 'Default'])
    plt.tight_layout()
    path = output_dir / 'fico_vs_default.png'
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Saved FICO score vs default plot: {path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot a correlation heatmap for selected numeric features."""
    features = [
        'target', 'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
        'fico_range_low', 'fico_range_high', 'open_acc', 'revol_bal', 'total_acc',
        'total_rec_prncp', 'total_pymnt', 'last_fico_range_low', 'last_fico_range_high'
    ]
    subset = df[features].copy()
    corr = subset.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap for Selected Loan Features')
    plt.tight_layout()
    path = output_dir / 'correlation_heatmap.png'
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Saved correlation heatmap: {path}")


def print_observations(df: pd.DataFrame):
    """Print key observations for each plot."""
    print('\nOBSERVATIONS:')
    counts = df['target'].value_counts(normalize=True)
    print(f"- Target distribution: {counts.get(0, 0):.2%} fully paid, {counts.get(1, 0):.2%} default.")

    int_rate_stats = df.groupby('target')['int_rate'].median()
    print(f"- Interest rate: defaulted loans have higher median interest rate ({int_rate_stats.loc[1]:.2f}%) than fully paid ({int_rate_stats.loc[0]:.2f}%).")

    inc_stats = df.groupby('target')['annual_inc'].median()
    print(f"- Annual income: defaulted loans have lower median income (${inc_stats.loc[1]:,.0f}) than fully paid (${inc_stats.loc[0]:,.0f}).")

    dti_stats = df.groupby('target')['dti'].median()
    print(f"- DTI ratio: defaulted loans show a higher median DTI ({dti_stats.loc[1]:.2f}%) than fully paid ({dti_stats.loc[0]:.2f}%).")

    fico_score = (df['fico_range_low'] + df['fico_range_high']) / 2.0
    fico_stats = fico_score.groupby(df['target']).median()
    print(f"- FICO score: defaulted loans have lower median FICO ({fico_stats.loc[1]:.1f}) than fully paid ({fico_stats.loc[0]:.1f}).")

    corr = df[['target', 'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'fico_range_low', 'fico_range_high', 'open_acc', 'revol_bal', 'total_acc', 'total_rec_prncp', 'total_pymnt', 'last_fico_range_low', 'last_fico_range_high']].corr()
    print(f"- Correlation highlight: target is most positively correlated with interest rate ({corr.loc['target','int_rate']:.2f}) and most negatively correlated with FICO score ({corr.loc['target','fico_range_low']:.2f}).")


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / 'cleaned_loans.csv'
    output_dir = project_root / 'outputs'

    ensure_output_dir(output_dir)
    df = load_cleaned_data(data_file)

    plot_target_distribution(df, output_dir)
    plot_int_rate_vs_default(df, output_dir)
    plot_annual_income_vs_default(df, output_dir)
    plot_dti_vs_default(df, output_dir)
    plot_fico_vs_default(df, output_dir)
    plot_correlation_heatmap(df, output_dir)

    print_observations(df)


if __name__ == '__main__':
    main()
