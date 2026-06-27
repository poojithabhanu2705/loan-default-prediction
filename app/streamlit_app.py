"""
Streamlit Application for Loan Default Prediction

A clean, interactive web app for predicting loan default risk based on
borrower and loan characteristics.
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 10px 0;
        }
        .low-risk {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .medium-risk {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .high-risk {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_pipeline():
    """Load the trained model and preprocessing pipeline."""
    root = Path(__file__).parent.parent / 'outputs'
    
    # Try to load existing model
    model_path = root / 'loan_default_model.joblib'
    pipeline_path = root / 'preprocessing_pipeline.joblib'
    feature_names_path = root / 'feature_names.joblib'
    
    if model_path.exists() and pipeline_path.exists():
        model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)
        feature_names = joblib.load(feature_names_path)
        return model, pipeline, feature_names, True
    
    # If model doesn't exist, train it
    if (root / 'X_train.csv').exists():
        import pandas as pd
        from sklearn.metrics import roc_auc_score
        
        X_train = pd.read_csv(root / 'X_train.csv')
        X_test = pd.read_csv(root / 'X_test.csv')
        y_train = pd.read_csv(root / 'y_train.csv')['target']
        y_test = pd.read_csv(root / 'y_test.csv')['target']
        
        model = LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        pipeline = joblib.load(pipeline_path)
        feature_names = X_train.columns.tolist()
        
        # Save model
        joblib.dump(model, model_path)
        joblib.dump(feature_names, feature_names_path)
        
        return model, pipeline, feature_names, True
    
    return None, None, None, False


def get_risk_category(probability):
    """Categorize default risk based on probability."""
    if probability < 0.30:
        return "🟢 LOW RISK", "low-risk"
    elif probability < 0.70:
        return "🟡 MEDIUM RISK", "medium-risk"
    else:
        return "🔴 HIGH RISK", "high-risk"


def create_sample_features(loan_amnt, annual_inc, int_rate, dti, fico_score):
    """Create a feature array from user inputs for prediction."""
    # This is a simplified version that uses the raw inputs
    # In a real scenario, this would need to match the preprocessing pipeline exactly
    features = np.array([[
        loan_amnt, annual_inc, int_rate, dti, fico_score,
        # Placeholder for other features (typically these would be 0 or averages)
        # This is simplified for the streamlit demo
    ]])
    return features


def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #667eea;'>🏦 Loan Default Predictor</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 16px; color: #666;'>Predict the risk of loan default based on borrower and loan characteristics</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and pipeline
    model, pipeline, feature_names, model_exists = load_model_and_pipeline()
    
    if not model_exists:
        st.error("⚠️ Model not found. Please ensure the preprocessing pipeline and training data are available.")
        return
    
    # Create input section
    st.subheader("📋 Loan Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        loan_amount = st.number_input(
            "💰 Loan Amount ($)",
            min_value=500,
            max_value=40000,
            value=12000,
            step=500,
            help="The amount of the loan request"
        )
        
        fico_score = st.number_input(
            "📊 FICO Score",
            min_value=300,
            max_value=850,
            value=700,
            step=10,
            help="Borrower's credit score"
        )
        
        dti_ratio = st.slider(
            "📈 Debt-to-Income Ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=18.0,
            step=0.5,
            help="Total monthly debt payments / monthly income"
        )
    
    with col2:
        annual_income = st.number_input(
            "💵 Annual Income ($)",
            min_value=10000,
            max_value=1100000,
            value=65000,
            step=5000,
            help="Annual income of the borrower"
        )
        
        interest_rate = st.slider(
            "📊 Interest Rate (%)",
            min_value=5.0,
            max_value=31.0,
            value=13.5,
            step=0.1,
            help="Annual percentage rate (APR) for the loan"
        )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Default Risk", use_container_width=True):
        # Create a properly formatted input for the preprocessed model
        # Since the model expects preprocessed features, we need to use the pipeline
        
        # Create a minimal dataframe with the raw features
        # Note: This is simplified. In production, all 243 features would be needed
        raw_input = pd.DataFrame({
            'loan_amnt': [loan_amount],
            'annual_inc': [annual_income],
            'int_rate': [interest_rate],
            'dti': [dti_ratio],
            'fico_range_low': [fico_score],
            'fico_range_high': [fico_score + 4],
        })
        
        try:
            # Transform using preprocessing pipeline
            features_transformed = pipeline.transform(raw_input)
            
            # Get prediction probability
            probability = model.predict_proba(features_transformed)[0][1]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Risk category
            risk_text, risk_class = get_risk_category(probability)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"<div class='{risk_class}'>{risk_text}</div>", unsafe_allow_html=True)
            
            # Probability display
            st.markdown("### Default Probability")
            col1, col2, col3 = st.columns(3)
            with col2:
                st.metric(
                    label="",
                    value=f"{probability*100:.1f}%",
                    delta=f"Risk Score: {probability:.4f}",
                    delta_color="inverse"
                )
            
            # Probability bar
            st.progress(probability, text=f"Default Probability: {probability*100:.2f}%")
            
            # Interpretation
            st.markdown("---")
            st.subheader("💡 Interpretation")
            
            if probability < 0.30:
                st.success("""
                    ✅ **Low Risk Loan**
                    
                    This loan has a low probability of default. The borrower presents favorable 
                    credit characteristics and financial metrics. Consider approval with standard terms.
                """)
            elif probability < 0.70:
                st.warning("""
                    ⚠️ **Medium Risk Loan**
                    
                    This loan has a moderate probability of default. Consider:
                    - Conducting additional due diligence
                    - Applying higher interest rates
                    - Requiring additional collateral or co-signer
                """)
            else:
                st.error("""
                    ❌ **High Risk Loan**
                    
                    This loan has a high probability of default. Recommend:
                    - Declining the application, or
                    - Requiring substantial risk mitigation measures
                    - Significantly higher interest rates
                    - Strong collateral requirements
                """)
            
            # Input summary
            st.markdown("---")
            st.subheader("📝 Input Summary")
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write(f"**Loan Amount:** ${loan_amount:,}")
                st.write(f"**Interest Rate:** {interest_rate:.2f}%")
                st.write(f"**FICO Score:** {fico_score}")
            
            with summary_col2:
                st.write(f"**Annual Income:** ${annual_income:,}")
                st.write(f"**DTI Ratio:** {dti_ratio:.2f}%")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please ensure all required features are properly formatted and within expected ranges.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #999; font-size: 12px;'>
            <p>🏦 Loan Default Prediction System | Built with Streamlit</p>
            <p>Model: Logistic Regression with Class Weights | Risk Categories: Low: <30%, Medium: 30-70%, High: >70%</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
