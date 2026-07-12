import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timezone
import google.generativeai as genai

# 1. Initialize Gemini with active model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash") 

app = FastAPI(title="CardioCare Hybrid Predictor Engine")

# 2. Database connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["cardiocare_db"]
predictions_collection = db["patient_predictions"]

# 3. Load ML model files
model = joblib.load('cardio_model.pkl')

# Updated Pydantic Model to accept smoking and activity
class PatientData(BaseModel):
    age_years: int
    height: float
    weight: float
    systolic_bp: int
    diastolic_bp: int
    cholesterol: int
    is_smoker: bool
    is_active: bool

# Deterministic Clinical Scoring Engine Function
def calculate_clinical_risk(age: int, height: float, weight: float, sys: int, dia: int, chol: int, smoker: bool, active: bool):
    risk_points = 0
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    
    # Age factor
    if age >= 60: risk_points += 4
    elif age >= 45: risk_points += 2
    
    # Blood Pressure (AHA Guideline Thresholds)
    if sys >= 140 or dia >= 90: risk_points += 3       # Stage 2 Hypertension
    elif 130 <= sys < 140 or 80 <= dia < 90: risk_points += 1 # Stage 1 Hypertension
    
    # Metabolic strain (BMI)
    if bmi >= 30: risk_points += 2                     # Obesity marker
    
    # Behavioral Risk factors (Direct Physiological impact)
    if smoker: risk_points += 3                        # Arterial damage multiplier
    if not active: risk_points += 1                    # Sedentary modifier
    
    # Cholesterol level points
    if chol == 3: risk_points += 3                     # Well above normal
    elif chol == 2: risk_points += 1                   # Above normal
    
    risk_label = "High Risk" if risk_points >= 6 else "Low Risk"
    return risk_points, risk_label

@app.post("/predict")
def predict_cardio_risk(patient: PatientData):
    try:
        # --- 1. RUN MACHINE LEARNING INFERENCE ---
        # The ML model expects exactly the 6 columns it was trained on
        input_data = pd.DataFrame([{
            'age_years': patient.age_years,
            'height': patient.height,
            'weight': patient.weight,
            'systolic_bp': patient.systolic_bp,
            'diastolic_bp': patient.diastolic_bp,
            'cholesterol': patient.cholesterol
        }])
        
        ml_prediction = int(model.predict(input_data)[0])
        ml_probability = float(model.predict_proba(input_data)[0][1])
        
        # --- 2. RUN CLINICAL SCORING ENGINE ---
        clinical_score, clinical_label = calculate_clinical_risk(
            patient.age_years, patient.height, patient.weight,
            patient.systolic_bp, patient.diastolic_bp, patient.cholesterol,
            patient.is_smoker, patient.is_active
        )
        
        # --- 3. COMBINE INSIGHTS FOR THE PROMPT ---
        chol_labels = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
        ai_advice = generate_ai_recommendations(
            age=patient.age_years,
            sys=patient.systolic_bp,
            dia=patient.diastolic_bp,
            chol=chol_labels.get(patient.cholesterol, "Unknown"),
            smoker=patient.is_smoker,
            active=patient.is_active,
            ml_status="High Risk" if ml_prediction == 1 else "Low Risk",
            clinical_status=clinical_label
        )
        
        # --- 4. PERSIST DOCUMENT TO NOSQL (MongoDB) ---
        log_document = {
            "timestamp": datetime.now(timezone.utc), # Avoids deprecated utcnow()
            "age": patient.age_years,
            "height": patient.height,
            "weight": patient.weight,
            "bmi": round(patient.weight / ((patient.height / 100) ** 2), 2),
            "blood_pressure": f"{patient.systolic_bp}/{patient.diastolic_bp}",
            "is_smoker": patient.is_smoker,
            "is_active": patient.is_active,
            "ml_high_risk_prob": ml_probability,
            "clinical_score": clinical_score,
            "final_assessment": "High Risk" if (clinical_label == "High Risk" or ml_prediction == 1) else "Low Risk"
        }
        predictions_collection.insert_one(log_document)
        
        return {
            "ml_prediction": ml_prediction,
            "high_risk_probability": ml_probability,
            "clinical_risk_score": clinical_score,
            "clinical_risk_label": clinical_label,
            "recommendations": ai_advice
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/history")
def get_assessment_history():
    try:
        # Fetch all records from the collection, newest first
        cursor = predictions_collection.find().sort("timestamp", -1)
        
        raw_records = []
        for record in cursor:
            if not isinstance(record, dict):
                continue
            
            # Convert MongoDB ObjectId to standard string so JSON can send it
            record["_id"] = str(record["_id"])
            
            # Convert datetime objects to string ISO format safely
            if isinstance(record.get("timestamp"), datetime):
                record["timestamp"] = record["timestamp"].isoformat()
                
            raw_records.append(record)
            
        return raw_records
        
    except Exception as e:
        print(f"DATABASE DASHBOARD ERROR: {str(e)}")
        return {"status": "error", "message": str(e)}

def generate_ai_recommendations(age: int, sys: int, dia: int, chol: str, smoker: bool, active: bool, ml_status: str, clinical_status: str) -> str:
    # Build prompt that forces explicit awareness of smoking and activity context
    prompt = f"""
    You are an expert AI clinical assistant specializing in preventive cardiology. 
    Analyze this patient's comprehensive profile and provide structured, actionable lifestyle recommendations.

    PATIENT PROFILE:
    - Age: {age} years
    - Blood Pressure: {sys}/{dia} mmHg
    - Cholesterol Status: {chol}
    - Tobacco Use: {"Active Smoker" if smoker else "Non-Smoker"}
    - Physical Activity: {"Regularly Active" if active else "Sedentary / Low Activity"}
    - Machine Learning Assessment: {ml_status}
    - Clinical Criteria Assessment: {clinical_status}

    CRITICAL INSTRUCTIONS:
    1. Identify specific risk areas based strictly on AHA guidelines. If the patient is an active smoker, prioritize explicit smoking cessation advice. If sedentary, outline custom physical guidelines.
    2. Provide 3 highly practical, bulleted changes the patient can easily initiate (nutrition targets, safe cardio ranges, etc.).
    3. Keep the tone professional, empathetic, and encouraging.
    4. Provide the output in clean Markdown format with explicit bold headings.
    5. Conclude with a clear medical disclaimer statement.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Recommendations temporarily unavailable. (Error Details: {str(e)})"