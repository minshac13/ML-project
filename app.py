import streamlit as st
import requests
import pandas as pd
import os

# Define backend base URL (using container service name or localhost based on your network)
if "BACKEND_URL" in st.secrets:
    BACKEND_URL = st.secrets["BACKEND_URL"]
else:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

BACKEND_URL = BACKEND_URL.rstrip("/")
API_URL = f"{BACKEND_URL}/predict"
HISTORY_URL = f"{BACKEND_URL}/history"

st.set_page_config(page_title="CardioCare Predictor", layout="wide")

# 1. Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["New Assessment", "History Dashboard"])

# 2. Page: New Assessment
if page == "New Assessment":
    st.title("❤️ Cardiovascular Risk Assessment")
    st.write("Enter the patient details below to predict the likelihood of cardiovascular disease.")
    
    st.divider()

    # 4. User Input Layout
    col1, col2 = st.columns(2)

    with col1:
        age_years = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
        is_smoker = st.selectbox("Smoking Status", options=["Non-Smoker", "Active Smoker"])

    with col2:
        systolic = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
        diastolic = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
        cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], 
                                format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
        is_active = st.selectbox("Physical Activity Level", options=["Regularly Active (>=150 mins/week)", "Sedentary / Low Activity"])

    # 5. Prediction Logic via API
    if st.button("Generate Assessment", type="primary"):

        age_years = int(age_years) if age_years else 50
        height = float(height) if height else 165.0
        weight = float(weight) if weight else 70.0
        systolic = int(systolic) if systolic else 120
        diastolic = int(diastolic) if diastolic else 80

        # Create the payload dictionary matching the exact schema FastAPI expects
        payload = {
            "age_years": float(age_years),
            "height": float(height),
            "weight": float(weight),
            "systolic_bp": float(systolic),
            "diastolic_bp": float(diastolic),
            "cholesterol": int(cholesterol),
            "is_smoker": True if is_smoker == "Active Smoker" else False,
            "is_active": True if is_active == "Regularly Active (>=150 mins/week)" else False
        }
        
        try:
            with st.spinner("Analyzing biometrics and generating clinical insights..."):
            # Send an HTTP POST request containing our data to the FastAPI endpoint
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()  # Parse the incoming JSON text into a Python dict
                    
                    # Extract Hybrid Results from Backend
                    ml_prediction = result["ml_prediction"]
                    ml_probability = result["high_risk_probability"]
                    clinical_label = result["clinical_risk_label"]
                    clinical_score = result["clinical_risk_score"]
                    ai_advice = result.get("recommendations", "") # Capture the AI advice
                    
                    st.divider()
                    
                    # 6. Display Results from API Response
                    st.subheader("📊 Diagnostic Assessment Summary")
                
                    # Display both indicators so the user gets context
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(label="Clinical Risk Score (AHA Criteria)", value=f"{clinical_score} pts", delta=clinical_label)
                    with c2:
                        st.metric(label="ML Model Prediction Confidence", value=f"{ml_probability:.1%}", delta="High Risk" if ml_prediction == 1 else "Low Risk", delta_color="inverse")
                    
                    # Final Combined Status Display
                    if clinical_label == "High Risk" or ml_prediction == 1:
                        st.error(f"### Status Flag: Elevated Cardiovascular Risk Detected")
                    else:
                        st.success(f"### Status Flag: Low Cardiovascular Risk")

                    if ai_advice:
                        st.markdown("---")
                        st.subheader("🤖 Personalized Clinical Guidance (AI Generated)")
                        st.markdown(ai_advice)
                else:
                    st.error(f"API Server Error: Received status code {response.status_code}")
                
        except Exception as e:
            st.error(f"Connection Error: {e}")

elif page == "History Dashboard":
    # 1. Global Font Styling via CSS
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
        html, body, [class*="css"], p, h1, h2, h3, h4, span {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }
        div[data-testid="stMetricValue"] {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 700;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.title("📊 Patient Assessment History")
    st.write("View and analyze historical records retrieved from MongoDB Atlas.")

    with st.spinner("Fetching historical data..."):
        try:
            response = requests.get(f"{BACKEND_URL}/history")
            if response.status_code == 200:
                result = response.json()
                
                raw_data = []
                if isinstance(result, dict):
                    if result.get("status") == "success" or "data" in result:
                        raw_data = result.get("data", [])
                elif isinstance(result, list):
                    raw_data = result
                    
                if raw_data:
                    flattened_records = []
                    
                    for record in raw_data:
                        if not isinstance(record, dict):
                            continue
                            
                        flat_row = {}
                        
                        # --- 1. Date / Timestamp Cleanup ---
                        date_str = record.get("timestamp") or record.get("Date Evaluated") or "N/A"
                        if isinstance(date_str, str) and "T" in date_str:
                            date_str = date_str.split(".")[0].replace("T", " ")
                        flat_row["Date"] = str(date_str)
                        
                        # --- 2. Input Features Dictionary Layer Safety Check ---
                        inputs = record.get("input_features", {})
                        if not isinstance(inputs, dict):
                            inputs = {}
                            
                        # --- 3. Core Demographics & Biometrics (With Fallbacks) ---
                        flat_row["Age"] = record.get("age") or inputs.get("age") or inputs.get("age_years") or record.get("Age") or "N/A"
                        flat_row["Height"] = record.get("height") or inputs.get("height") or "N/A"
                        flat_row["Weight"] = record.get("weight") or inputs.get("weight") or "N/A"
                        
                        # --- 4. Blood Pressure Parser ---
                        # Checks lowercase root, uppercase root, and nested layouts
                        bp_str = record.get("blood_pressure") or record.get("Blood Pressure") or inputs.get("blood_pressure") or "N/A"
                        
                        # If a single combined string like "120/80" is found, split it safely
                        if "/" in str(bp_str) and bp_str != "N/A":
                            try:
                                flat_row["Systolic BP"] = bp_str.split("/")[0]
                                flat_row["Diastolic BP"] = bp_str.split("/")[1]
                            except Exception:
                                flat_row["Systolic BP"] = "N/A"
                                flat_row["Diastolic BP"] = "N/A"
                        else:
                            # Fallback if separate integer keys are present instead
                            flat_row["Systolic BP"] = record.get("systolic_bp") or inputs.get("systolic_bp") or inputs.get("ap_hi") or "N/A"
                            flat_row["Diastolic BP"] = record.get("diastolic_bp") or inputs.get("diastolic_bp") or inputs.get("ap_lo") or "N/A"
                        
                        # --- 5. Lifestyle / Behavioral Status Fields ---
                        is_smoker = record.get("is_smoker") if record.get("is_smoker") is not None else inputs.get("is_smoker")
                        if is_smoker is None and "Smoking Status" in record:
                            is_smoker = "Active" in str(record.get("Smoking Status"))
                        flat_row["Smoking Status"] = "Active Smoker" if is_smoker is True else "Non-Smoker"
                        
                        is_active = record.get("is_active") if record.get("is_active") is not None else inputs.get("is_active")
                        if is_active is None and "Activity Level" in record:
                            is_active = "Active" in str(record.get("Activity Level"))
                        flat_row["Activity Level"] = "Active" if is_active is True else "Sedentary"
                        
                        # --- 6. Cholesterol Level Mapping ---
                        raw_chol = str(record.get("cholesterol", inputs.get("cholesterol", inputs.get("cholesterol_level", "1"))))
                        if raw_chol == "1":
                            flat_row["Cholesterol Level"] = "Normal"
                        elif raw_chol == "2":
                            flat_row["Cholesterol Level"] = "Above Normal"
                        elif raw_chol == "3":
                            flat_row["Cholesterol Level"] = "Well Above Normal"
                        else:
                            flat_row["Cholesterol Level"] = "Normal"  # Default fallback
                        
                        # --- 7. Output Metrics & Scores ---
                        res = record.get("final_assessment") or record.get("Assessment") or "N/A"
                        flat_row["Assessment"] = "High Risk" if "High" in str(res) else "Low Risk"
                        
                        score = record.get("clinical_score") or record.get("Clinical Score") or 0
                        # Strip non-numeric text out if points were hardcoded as a string
                        score_clean = str(score).replace(" pts", "")
                        flat_row["Clinical Score"] = f"{score_clean} pts"
                        
                        flattened_records.append(flat_row)
                    
                    if flattened_records:
                        df = pd.DataFrame(flattened_records)
                        
                        desired_order = [
                            "Date", "Age", "Height", "Weight", 
                            "Systolic BP", "Diastolic BP", "Cholesterol Level",
                            "Smoking Status", "Activity Level", "Clinical Score", "Assessment"
                        ]
                        df = df[[col for col in desired_order if col in df.columns]]
                        
                        # --- METRICS SECTION ---
                        st.subheader("💡 Quick Analytics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Assessments Run", len(df))
                        with col2:
                            high_risk_count = df["Assessment"].str.contains("High", na=False).sum()
                            st.metric("High Risk Cases Detected", high_risk_count)
                        
                        st.markdown("---")
                        
                        # --- DATA TABLE ---
                        st.subheader("📋 Interactive Patient Log")
                        st.write("Use the headers to sort or filter specific case entries.")
                        
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Assessment": st.column_config.SelectboxColumn(
                                    "Assessment",
                                    width="medium",
                                    required=True,
                                    options=["High Risk", "Low Risk"],
                                )
                            }
                        )
                    else:
                        st.info("No records could be parsed from the system database history log data.")
                else:
                    st.info("No records found in the database yet.")
            else:
                st.error(f"Failed to connect to backend server API. Status Code: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")