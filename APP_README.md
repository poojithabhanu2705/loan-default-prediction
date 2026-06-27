# 🏦 Loan Default Prediction Streamlit Application

A clean, interactive web application for predicting loan default risk based on borrower and loan characteristics.

## 📋 Features

- **Interactive Input Form** - Easily input loan and borrower information
- **Real-time Predictions** - Get instant default probability predictions
- **Risk Categorization** - Automatic classification into Low, Medium, or High risk
- **Visual Feedback** - Color-coded risk indicators and probability bars
- **Detailed Interpretation** - Actionable insights for each prediction
- **Responsive UI** - Clean, professional design with custom styling

## 🎯 Input Parameters

The application accepts the following inputs:

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Loan Amount** | $500 - $40,000 | The requested loan amount |
| **Annual Income** | $10,000 - $1,100,000 | Borrower's annual income |
| **Interest Rate** | 5.0% - 31.0% | Annual percentage rate (APR) |
| **DTI Ratio** | 0% - 100% | Debt-to-Income ratio |
| **FICO Score** | 300 - 850 | Borrower's credit score |

## 📊 Output

The application provides:

1. **Default Probability** - The percentage likelihood of loan default
2. **Risk Category** - Classification (Low: <30%, Medium: 30-70%, High: >70%)
3. **Risk Interpretation** - Actionable recommendations based on risk level
4. **Input Summary** - Recap of all input parameters

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment with required packages installed
- Preprocessed training and test data in `outputs/` folder
- Trained preprocessing pipeline (`preprocessing_pipeline.joblib`)

### Running the Application

**Option 1: Using the provided script (Linux/Mac)**

```bash
chmod +x run_app.sh
./run_app.sh
```

**Option 2: Manual startup**

```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit
streamlit run app/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 🔄 Model & Processing Pipeline

The application uses:

- **Model**: Logistic Regression with class weights for imbalanced data handling
- **Preprocessing**: Standardized numerical features and one-hot encoded categorical features
- **Input**: 5 key loan/borrower characteristics
- **Output**: Default probability (0-1) and risk category

## 🎨 UI Components

- **Header** - Application title and description
- **Input Section** - Two-column layout for easy data entry
- **Prediction Results** - Visual risk indicator and probability display
- **Interpretation** - Context-specific guidance based on risk level
- **Summary** - Quick recap of all input values

## 📝 Risk Categories

| Category | Probability Range | Indicator | Recommendation |
|----------|-------------------|-----------|-----------------|
| **Low Risk** | < 30% | 🟢 | Approve with standard terms |
| **Medium Risk** | 30% - 70% | 🟡 | Review carefully, consider risk mitigation |
| **High Risk** | > 70% | 🔴 | Decline or require strong collateral |

## 🛠️ Model Performance

The underlying model achieves:

- **Accuracy**: ~80% on test set
- **ROC AUC**: ~0.85 on test set
- **Balanced for**: Imbalanced dataset with class weights

## 🔧 Customization

To customize the application:

1. **Modify input ranges**: Edit the `st.number_input()` parameters in the input section
2. **Change risk thresholds**: Update the `get_risk_category()` function
3. **Adjust styling**: Modify the custom CSS in the page config section
4. **Update feature engineering**: Adjust the feature transformation in `create_sample_features()`

## 📦 Dependencies

- `streamlit` - Web application framework
- `scikit-learn` - Machine learning models
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `joblib` - Model serialization

## 💡 Troubleshooting

**Issue**: "Model not found" error
- **Solution**: Ensure `preprocessing_pipeline.joblib` exists in the `outputs/` folder
- Run `python src/preprocess_loan_data.py` to generate the preprocessing pipeline

**Issue**: "Error making prediction"
- **Solution**: Check that input values are within the expected ranges
- Verify all preprocessed features are available

**Issue**: Port already in use
- **Solution**: Use `streamlit run app/streamlit_app.py --server.port 8502` to use a different port

## 📚 Project Structure

```
loan-default-prediction/
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── src/
│   ├── loan_data_analysis.py     # Data analysis scripts
│   ├── loan_data_eda.py          # Exploratory analysis
│   ├── preprocess_loan_data.py   # Preprocessing pipeline
│   └── train_final_model.py      # Model training
├── outputs/
│   ├── X_train.csv               # Training features
│   ├── X_test.csv                # Test features
│   ├── preprocessing_pipeline.joblib
│   └── loan_default_model.joblib
├── run_app.sh                    # Application startup script
└── README.md
```

## 📞 Support

For issues with the application, check:
1. Model and pipeline files exist in `outputs/` folder
2. All input values are within valid ranges
3. Virtual environment is properly activated
4. Streamlit is up to date: `pip install --upgrade streamlit`

---

**Built with**: Streamlit | **Model**: Scikit-learn | **Data**: Lending Club
