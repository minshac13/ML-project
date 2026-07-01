# import streamlit as st
# import requests

# # 1. Page Configuration
# st.set_page_config(page_title="CardioCare Predictor", layout="centered")

# # 2. Define the exact URL where your FastAPI server is listening
# # This points to the local server we started in the previous steps
# API_URL = "http://backend:8000/predict"

# # 3. Header Section
# st.title("❤️ Cardiovascular Risk Assessment")
# st.write("Enter the patient details below to predict the likelihood of cardiovascular disease.")
# st.divider()

# # 4. User Input Layout
# col1, col2 = st.columns(2)

# with col1:
#     age_years = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
#     height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
#     weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)

# with col2:
#     systolic = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
#     diastolic = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
#     cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], 
#                                format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])

# # 5. Prediction Logic via API
# if st.button("Generate Assessment", type="primary"):
    
#     # Create the payload dictionary matching the exact schema FastAPI expects
#     payload = {
#         "age_years": float(age_years),
#         "height": float(height),
#         "weight": float(weight),
#         "systolic_bp": float(systolic),
#         "diastolic_bp": float(diastolic),
#         "cholesterol": int(cholesterol)
#     }
    
#     try:
#         # Send an HTTP POST request containing our data to the FastAPI endpoint
#         response = requests.post(API_URL, json=payload)
        
#         # Check if the API responded with success (Status Code 200)
#         if response.status_code == 200:
#             result = response.json()  # Parse the incoming JSON text into a Python dict
            
#             prediction = result["prediction"]
#             probability = result["high_risk_probability"]
#             risk_label = result["risk_label"]
            
#             st.divider()
            
#             # 6. Display Results from API Response
#             if prediction == 1:
#                 st.error(f"### Result: {risk_label} Detected")
#                 st.write(f"The model identifies patterns consistent with cardiovascular disease (Confidence: {probability:.2%})")
#                 st.warning("Advise: Please consult with a healthcare professional for a detailed clinical examination.")
#             else:
#                 st.success(f"### Result: {risk_label} Detected")
#                 st.write(f"The model does not identify significant risk factors at this time (Confidence: {1-probability:.2%})")
#                 st.info("Advise: Maintain a healthy diet and regular exercise to keep your risk levels low.")
#         else:
#             st.error(f"API Server Error: Received status code {response.status_code}")
            
#     except requests.exceptions.ConnectionError:
#         st.error("Could not connect to the FastAPI backend. Is your API server running on port 8000?")

# # 7. Sidebar Info
# st.sidebar.info("This interface acts as a clean client frontend, requesting predictive analytics from a dedicated FastAPI microservice backend.")

import streamlit as st
import requests
import pandas as pd

# Define backend base URL (using container service name or localhost based on your network)
BACKEND_URL = "http://backend:8000" # Use "http://localhost:8000" if running outside Docker
API_URL = "http://backend:8000/predict"

st.set_page_config(page_title="CardioCare Predictor", layout="wide")

# 1. Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["New Assessment", "History Dashboard"])

# 2. Page: New Assessment (Your existing form code goes here)
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

    with col2:
        systolic = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
        diastolic = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
        cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], 
                                format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])

    # 5. Prediction Logic via API
    if st.button("Generate Assessment", type="primary"):
        
        # Create the payload dictionary matching the exact schema FastAPI expects
        payload = {
            "age_years": float(age_years),
            "height": float(height),
            "weight": float(weight),
            "systolic_bp": float(systolic),
            "diastolic_bp": float(diastolic),
            "cholesterol": int(cholesterol)
        }
        
        try:
            # Send an HTTP POST request containing our data to the FastAPI endpoint
            response = requests.post(API_URL, json=payload)
            
            # Check if the API responded with success (Status Code 200)
            if response.status_code == 200:
                result = response.json()  # Parse the incoming JSON text into a Python dict
                
                prediction = result["prediction"]
                probability = result["high_risk_probability"]
                risk_label = result["risk_label"]
                
                st.divider()
                
                # 6. Display Results from API Response
                if prediction == 1:
                    st.error(f"### Result: {risk_label} Detected")
                    st.write(f"The model identifies patterns consistent with cardiovascular disease (Confidence: {probability:.2%})")
                    st.warning("Advise: Please consult with a healthcare professional for a detailed clinical examination.")
                else:
                    st.success(f"### Result: {risk_label} Detected")
                    st.write(f"The model does not identify significant risk factors at this time (Confidence: {1-probability:.2%})")
                    st.info("Advise: Maintain a healthy diet and regular exercise to keep your risk levels low.")
            else:
                st.error(f"API Server Error: Received status code {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Is your API server running on port 8000?")


# 3. Page: History Dashboard
# elif page == "History Dashboard":
#     st.title("📊 Patient Assessment History")
#     st.write("View and analyze historical records retrieved from MongoDB Atlas.")
    
#     with st.spinner("Fetching historical data..."):
#         try:
#             response = requests.get(f"{BACKEND_URL}/assessments")
#             if response.status_code == 200:
#                 result = response.json()
                
#                 if result.get("status") == "success" and result.get("data"):
#                     # Convert JSON data straight into a Pandas DataFrame for easy viewing
#                     df = pd.DataFrame(result["data"])
                    
#                     # Optional cleanup: drop the internal DB id column from showing up in the table
#                     if "id" in df.columns:
#                         df = df.drop(columns=["id"])
                    
#                     # Reorder columns visually if you like
#                     st.subheader("📋 All Records")
#                     st.dataframe(df, use_container_width=True)
                    
#                     # Add a simple metric display for fun
#                     st.subheader("💡 Quick Analytics")
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.metric("Total Assessments Run", len(df))
#                     with col2:
#                         # Assuming your prediction result column is named 'prediction' or 'risk_score'
#                         if "prediction" in df.columns:
#                             high_risk_count = (df["prediction"] == "High Risk").sum() # Edit match based on your output
#                             st.metric("High Risk Cases Detected", high_risk_count)
                            
#                 else:
#                     st.info("No records found in the database yet. Go run an assessment!")
#             else:
#                 st.error("Failed to connect to backend server API.")
#         except Exception as e:
#             st.error(f"Error loading dashboard: {e}")

# elif page == "History Dashboard":
#     st.title("📊 Patient Assessment History")
#     st.write("View and analyze historical records retrieved from MongoDB Atlas.")
    
#     with st.spinner("Fetching historical data..."):
#         try:
#             response = requests.get(f"{BACKEND_URL}/assessments")
#             if response.status_code == 200:
#                 result = response.json()
                
#                 if result.get("status") == "success" and result.get("data"):
#                     # 1. Load data into DataFrame
#                     df = pd.DataFrame(result["data"])
                    
#                     # 2. Cleanup MongoDB internal columns if they exist
#                     columns_to_drop = ["id", "_id", "status"]
#                     df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
                    
#                     # 3. Rename columns to look professional
#                     rename_dict = {
#                         "age": "Age (Years)",
#                         "gender": "Gender",
#                         "height": "Height (cm)",
#                         "weight": "Weight (kg)",
#                         "ap_hi": "Systolic BP",
#                         "ap_lo": "Diastolic BP",
#                         "cholesterol": "Cholesterol",
#                         "gluc": "Glucose Level",
#                         "smoke": "Smoker",
#                         "alco": "Alcohol Consumer",
#                         "active": "Physical Activity",
#                         "prediction": "Risk Assessment",
#                         "probability": "Risk Probability (%)",
#                         "timestamp": "Date Evaluated"
#                     }
#                     df = df.rename(columns=rename_dict)
                    
#                     # 4. Map binary/numeric categories to clean text labels if needed
#                     # (Adjust these mappings based on how your backend stores them)
#                     if "Risk Probability (%)" in df.columns:
#                         # Convert to clean percentage formatting if it's a decimal/float
#                         df["Risk Probability (%)"] = df["Risk Probability (%)"].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)

#                     # --- METRICS SECTION ---
#                     st.subheader("💡 Quick Analytics")
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.metric("Total Assessments Run", len(df))
#                     with col2:
#                         # Count high-risk cases dynamically
#                         risk_col = "Risk Assessment" if "Risk Assessment" in df.columns else "prediction"
#                         if risk_col in df.columns:
#                             # Match against whatever string your model outputs (e.g., "High Risk" or 1)
#                             high_risk_mask = df[risk_col].astype(str).str.contains("High|1", case=False, na=False)
#                             high_risk_count = high_risk_mask.sum()
#                             st.metric("High Risk Cases Detected", high_risk_count, delta=f"{high_risk_count/len(df)*100:.1f}% of total", delta_color="inverse")
#                     with col3:
#                         # Average age calculation
#                         age_col = "Age (Years)" if "Age (Years)" in df.columns else "age"
#                         if age_col in df.columns:
#                             avg_age = pd.to_numeric(df[age_col], errors='coerce').mean()
#                             st.metric("Average Patient Age", f"{avg_age:.1f} yrs" if not pd.isna(avg_age) else "N/A")
                    
#                     st.markdown("---")
                    
#                     # --- BEAUTIFIED DATA TABLE ---
#                     st.subheader("📋 Interactive Patient Log")
#                     st.write("Use the column headers to sort, search, or filter specific case details.")
                    
#                     # Display interactive dataframe with customized container width
#                     st.dataframe(
#                         df, 
#                         use_container_width=True,
#                         hide_index=True # Hides the row numbers (0, 1, 2...) for a cleaner look
#                     )
                    
#                 else:
#                     st.info("No records found in the database yet. Go run an assessment!")
#             else:
#                 st.error("Failed to connect to backend server API.")
#         except Exception as e:
#             st.error(f"Error loading dashboard: {e}")

# elif page == "History Dashboard":
#     # Custom CSS to apply a clean font style across this page
#     st.markdown(
#         """
#         <style>
#         @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
#         html, body, [class*="css"], p, h1, h2, h3, h4 {
#             font-family: 'Inter', sans-serif !important;
#         }
#         </style>
#         """, 
#         unsafe_allow_html=True
#     )

#     st.title("📊 Patient Assessment History")
#     st.write("View and analyze historical records retrieved from MongoDB Atlas.")
    
#     with st.spinner("Fetching historical data..."):
#         try:
#             response = requests.get(f"{BACKEND_URL}/assessments")
#             if response.status_code == 200:
#                 result = response.json()
                
#                 if result.get("status") == "success" and result.get("data"):
#                     raw_data = result["data"]
#                     flattened_records = []
                    
#                     # 1. Unpack the nested JSON format from MongoDB rows
#                     for record in raw_data:
#                         flat_row = {}
                        
#                         # Extract the Date
#                         flat_row["Date"] = record.get("Date Evaluated") or record.get("timestamp", "N/A")
#                         # Clean up timestamp strings to show just the date and time cleanly
#                         if "T" in flat_row["Date"]:
#                             flat_row["Date"] = flat_row["Date"].split(".")[0].replace("T", " ")
                        
#                         # Safely grab dictionary pieces from "input_features"
#                         inputs = record.get("input_features", {})
#                         if isinstance(inputs, dict):
#                             flat_row["Age"] = inputs.get("age_years") or inputs.get("age") or "N/A"
#                             flat_row["Height"] = inputs.get("height") or "N/A"
#                             flat_row["Weight"] = inputs.get("weight") or "N/A"
#                             flat_row["Systolic BP"] = inputs.get("systolic_bp") or inputs.get("ap_hi") or "N/A"
#                             flat_row["Diastolic BP"] = inputs.get("diastolic_bp") or inputs.get("ap_lo") or "N/A"
#                             flat_row["Cholesterol Level"] = inputs.get("cholesterol_level") or inputs.get("cholesterol") or "N/A"
                        
#                         # Safely grab assessment pieces from "inference_results"
#                         inference = record.get("inference_results", {})
#                         if isinstance(inference, dict):
#                             # Tries to find your risk labels or defaults gracefully
#                             flat_row["Assessment"] = inference.get("risk_label") or inference.get("prediction") or "N/A"
#                             # If it's a numeric 1 or 0, convert to readable words
#                             if flat_row["Assessment"] == 1 or str(flat_row["Assessment"]) == "1":
#                                 flat_row["Assessment"] = "⚠️ Hjjigh Risk"
#                             elif flat_row["Assessment"] == 0 or str(flat_row["Assessment"]) == "0":
#                                 flat_row["Assessment"] = "✅ Low Risk"
                        
#                         flattened_records.append(flat_row)
                    
#                     # 2. Convert our clean, un-nested records into a DataFrame
#                     df = pd.DataFrame(flattened_records)
                    
#                     # Enforce the column sorting order you asked for
#                     desired_order = ["Date", "Age", "Height", "Weight", "Systolic BP", "Diastolic BP", "Cholesterol Level", "Assessment"]
#                     df = df[[col for col in desired_order if col in df.columns]]
                    
#                     # --- METRICS SECTION ---
#                     st.subheader("💡 Quick Analytics")
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.metric("Total Assessments Run", len(df))
#                     with col2:
#                         if "Assessment" in df.columns:
#                             high_risk_count = df["Assessment"].astype(str).str.contains("High|1", case=False, na=False).sum()
#                             st.metric("High Risk Cases Detected", high_risk_count)
                    
#                     st.markdown("---")
                    
#                     # --- BEAUTIFIED DATA TABLE ---
#                     st.subheader("📋 Interactive Patient Log")
                    
#                     # 3. Render dataframe with background colors applied to the "Assessment" column
#                     st.dataframe(
#                         df,
#                         use_container_width=True,
#                         hide_index=True,
#                         column_config={
#                             "Assessment": st.column_config.TextColumn(
#                                 "Assessment",
#                                 help="Calculated risk status from machine learning model",
#                                 # This subtly highlights text formatting natively or adds color structures
#                             )
#                         }
#                     )
                    
#                 else:
#                     st.info("No records found in the database yet. Go run an assessment!")
#             else:
#                 st.error("Failed to connect to backend server API.")
#         except Exception as e:
#             st.error(f"Error loading dashboard: {e}")

elif page == "History Dashboard":
    # 1. Global Font Styling via CSS (for headers, metrics, and text)
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
            response = requests.get(f"{BACKEND_URL}/assessments")
            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") == "success" and result.get("data"):
                    raw_data = result["data"]
                    flattened_records = []
                    
                    for record in raw_data:
                        flat_row = {}
                        
                        # Date Cleanup
                        date_str = record.get("Date Evaluated") or record.get("timestamp", "N/A")
                        if "T" in date_str:
                            date_str = date_str.split(".")[0].replace("T", " ")
                        flat_row["Date"] = date_str
                        
                        # Extract inputs safely
                        inputs = record.get("input_features", {})
                        if isinstance(inputs, dict):
                            flat_row["Age"] = inputs.get("age_years") or inputs.get("age") or "N/A"
                            flat_row["Height"] = inputs.get("height") or "N/A"
                            flat_row["Weight"] = inputs.get("weight") or "N/A"
                            flat_row["Systolic BP"] = inputs.get("systolic_bp") or inputs.get("ap_hi") or "N/A"
                            flat_row["Diastolic BP"] = inputs.get("diastolic_bp") or inputs.get("ap_lo") or "N/A"
                            
                            # Clean up the raw numerical code for Cholesterol
                            raw_chol = str(inputs.get("cholesterol", inputs.get("cholesterol_level", "1")))
                            if raw_chol == "1":
                                flat_row["Cholesterol Level"] = "Normal"
                            elif raw_chol == "2":
                                flat_row["Cholesterol Level"] = "Above Normal"
                            elif raw_chol == "3":
                                flat_row["Cholesterol Level"] = "Well Above Normal"
                            else:
                                flat_row["Cholesterol Level"] = "Unknown"
                        
                        # Extract inference results safely
                        inference = record.get("inference_results", {})
                        if isinstance(inference, dict):
                            res = inference.get("risk_label") or inference.get("prediction") or "N/A"
                            # Map values cleanly to visual badges
                            if str(res) in ["1", "High Risk"]:
                                flat_row["Assessment"] = "High Risk"
                            else:
                                flat_row["Assessment"] = "Low Risk"
                        
                        flattened_records.append(flat_row)
                    
                    df = pd.DataFrame(flattened_records)
                    
                    # Ensure exact column order requested
                    desired_order = ["Date", "Age", "Height", "Weight", "Systolic BP", "Diastolic BP", "Cholesterol Level", "Assessment"]
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
                    
                    # --- BEAUTIFIED DATA TABLE ---
                    st.subheader("📋 Interactive Patient Log")
                    st.write("Use the headers to sort or filter specific case entries.")
                    
                    # Style and highlight the assessment column natively
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
                    st.info("No records found in the database yet.")
            else:
                st.error("Failed to connect to backend server API.")
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")