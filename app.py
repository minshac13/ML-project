import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="CardioCare Predictor", layout="centered")

# 2. Load the Model and Imputer
# Ensure these files are in the same folder as this script
@st.cache_resource
def load_assets():
    model = joblib.load('cardio_model.pkl')
    imputer = joblib.load('cardio_imputer.pkl')
    return model, imputer

try:
    model, imputer = load_assets()
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'cardio_model.pkl' and 'cardio_imputer.pkl' are in the directory.")

# 3. Header Section
st.title("❤️ Cardiovascular Risk Assessment")
st.write("Enter the patient details below to predict the likelihood of cardiovascular disease.")
st.divider()

# 4. User Input Layout
col1, col2 = st.columns(2)

with col1:
    age_years = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)

with col2:
    systolic = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
    diastolic = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], 
                               format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])

# 5. Prediction Logic
if st.button("Generate Assessment", type="primary"):
    # Convert age back to days as the model was trained on days
    age_days = age_years * 365.25
    
    # Create the input dataframe with EXACT column names from training
    # Note: Using the 6 features we selected: age, height, weight, ap_hi, ap_lo, cholesterol
    user_data = pd.DataFrame([[age_days, height, weight, systolic, diastolic, cholesterol]], 
                              columns=['age_years','height', 'weight','systolic_bp', 'diastolic_bp','cholesterol'])
    
    # Apply the Imputer (in case any inputs were somehow missed)
    user_data_imputed = imputer.transform(user_data)
    
    # Make Prediction
    prediction = model.predict(user_data_imputed)
    probability = model.predict_proba(user_data_imputed)[0][1] # Probability of Class 1

    st.divider()
    
    # 6. Display Results
    if prediction[0] == 1:
        st.error(f"### Result: High Risk Detected")
        st.write(f"The model identifies patterns consistent with cardiovascular disease (Confidence: {probability:.2%})")
        st.warning("Advise: Please consult with a healthcare professional for a detailed clinical examination.")
    else:
        st.success(f"### Result: Low Risk Detected")
        st.write(f"The model does not identify significant risk factors at this time (Confidence: {1-probability:.2%})")
        st.info("Advise: Maintain a healthy diet and regular exercise to keep your risk levels low.")

# 7. Sidebar Info
st.sidebar.info("This tool uses a Random Forest Classifier trained on clinical data to assess risk based on 6 key physiological indicators.")