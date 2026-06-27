# Loan Default Prediction System

A machine learning-based credit risk assessment system that predicts the probability of loan default using borrower financial information and credit attributes. The project applies data preprocessing, feature engineering, class imbalance handling, and ensemble learning techniques to assist in identifying high-risk borrowers.

---

## Overview

Financial institutions rely on credit risk assessment to minimize losses and make informed lending decisions. This project develops an end-to-end loan default prediction pipeline using real-world borrower data and machine learning models.

The system analyzes applicant characteristics such as income, loan amount, debt-to-income ratio, credit history, and interest rates to estimate default risk and classify borrowers into different risk categories.

---

## Features

* Data preprocessing and missing value handling
* Exploratory data analysis and risk pattern identification
* Feature engineering for borrower attributes
* Class imbalance handling using SMOTE
* Multiple machine learning models
* Model evaluation using classification metrics
* SHAP-based model interpretability
* Interactive Streamlit application
* Real-time risk prediction

---

## Tech Stack

### Languages

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn
* SHAP
* Matplotlib
* Seaborn

### Deployment

* Streamlit

---

## Dataset

The project uses the Lending Club loan dataset containing borrower financial information, loan characteristics, and repayment outcomes.

Features include:

* Loan amount
* Interest rate
* Annual income
* Employment length
* Debt-to-income ratio
* Credit score information
* Home ownership status
* Loan purpose

---

## Machine Learning Pipeline

1. Data Cleaning and Missing Value Treatment
2. Exploratory Data Analysis
3. Feature Engineering
4. Categorical Encoding
5. Train-Test Split
6. Class Imbalance Handling using SMOTE
7. Model Training
8. Performance Evaluation
9. Model Explainability using SHAP
10. Deployment using Streamlit

---

## Models Implemented

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

---

## Project Structure

```text
loan-default-prediction/
│
├── data/
├── notebooks/
├── src/
├── models/
├── app/
├── outputs/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

```bash
git clone <repository-url>
cd loan-default-prediction

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app/streamlit_app.py
```

---

## Results

* Improved minority class detection using SMOTE.
* Achieved strong classification performance using ensemble models.
* Identified key risk factors affecting loan defaults.
* Generated explainable predictions using SHAP.

---

## Future Improvements

* Hyperparameter tuning using Optuna.
* Model tracking with MLflow.
* REST API deployment using FastAPI.
* Docker containerization.
* Cloud deployment using AWS or Render.

