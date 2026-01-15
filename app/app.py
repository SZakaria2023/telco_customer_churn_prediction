"""
Streamlit App for Telco Customer Churn Prediction
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from app.preprocessing import preprocess_input
except ImportError:
    # Fallback if running from app directory
    from preprocessing import preprocess_input

# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_model_and_artifacts():
    """Load model and preprocessing artifacts"""
    try:
        # Get the directory of this file
        app_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(app_dir, 'best_model.pkl')
        encoder_path = os.path.join(app_dir, 'target_encoder.pkl')
        feature_names_path = os.path.join(app_dir, 'feature_names.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found at {encoder_path}")
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature names file not found at {feature_names_path}")
        
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        feature_names = joblib.load(feature_names_path)
        
        return model, encoder, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load model and artifacts
try:
    model, encoder, feature_names = load_model_and_artifacts()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Title
st.markdown('<h1 class="main-header">📊 Telco Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Customer Information")

# Create input form
with st.sidebar.form("customer_form"):
    # Demographics
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    # Account Information
    st.subheader("Account Information")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    
    # Services
    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    
    if phone_service == "Yes":
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    else:
        multiple_lines = "No phone service"
    
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    # Online Services (only if internet service is not "No")
    if internet_service != "No":
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    else:
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
    
    # Charges
    st.subheader("Charges")
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 118.0, 50.0, step=0.01)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges), step=0.01)
    
    # Submit button
    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

# Main content area
if submitted:
    try:
        # Create input dictionary
        input_dict = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Preprocess input
        input_encoded = preprocess_input(input_dict, feature_names)
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0]
        
        churn_prob = probability[1] * 100
        no_churn_prob = probability[0] * 100
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ **CHURN** - Customer is likely to churn")
            else:
                st.success(f"✅ **NO CHURN** - Customer is likely to stay")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader("Probability")
            st.metric("Churn Probability", f"{churn_prob:.2f}%")
            st.metric("No Churn Probability", f"{no_churn_prob:.2f}%")
            
            # Progress bar
            st.progress(churn_prob / 100)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # SHAP Explanation
        st.markdown("---")
        st.subheader("Model Explanation (SHAP)")
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_encoded)
            
            # Get feature importance for this prediction
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_values[0] if isinstance(shap_values, list) else shap_values[0]
            }).sort_values('SHAP Value', key=abs, ascending=False).head(10)
            
            # Display top contributing features
            st.write("**Top 10 Features Contributing to This Prediction:**")
            st.dataframe(feature_importance_df, use_container_width=True)
            
            # Create SHAP waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0] if isinstance(shap_values, list) else shap_values[0],
                    base_values=explainer.expected_value,
                    data=input_encoded.iloc[0].values,
                    feature_names=feature_names
                ),
                show=False
            )
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.warning(f"SHAP visualization could not be generated: {str(e)}")
            st.info("This might occur if the model type doesn't support SHAP TreeExplainer.")
        
        # Feature Engineering Info
        st.markdown("---")
        with st.expander("View Input Features"):
            st.json(input_dict)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.exception(e)

else:
    # Show instructions when form is not submitted
    st.info("👈 Please fill in the customer information in the sidebar and click 'Predict Churn' to get a prediction.")
    
    # Display some statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", "7,032 customers")
    with col2:
        st.metric("Churn Rate", "~26.6%")
    with col3:
        st.metric("Features", "20 features")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Telco Customer Churn Prediction App</p>
        <p>Built with Streamlit | Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
