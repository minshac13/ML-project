# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# import numpy as np

# # 1. Initialize the FastAPI app
# app = FastAPI(title="CardioCare Predictor Engine")

# # 2. Load your exact assets (Model and Imputer)
# model = joblib.load('cardio_model.pkl')
# imputer = joblib.load('cardio_imputer.pkl')

# # 3. Define the strict Data Schema representing your 6 features
# class PatientInput(BaseModel):
#     age_years: float
#     height: float
#     weight: float
#     systolic_bp: float
#     diastolic_bp: float
#     cholesterol: int  # Accepts 1, 2, or 3

# # 4. Create the prediction endpoint
# @app.post("/predict")
# def predict_cardio_risk(patient: PatientInput):
#     """
#     Accepts patient features, preprocesses age to days, applies the 
#     imputer, and runs inference using the CardioCare model.
#     """
#     # Preprocessing: Convert age to days just like in your training logic
#     age_days = patient.age_years * 365.25
    
#     # Create the input DataFrame with your exact training column names
#     user_data = pd.DataFrame(
#         [[age_days, patient.height, patient.weight, patient.systolic_bp, patient.diastolic_bp, patient.cholesterol]], 
#         columns=['age_years', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 'cholesterol']
#     )
    
#     # Process the data through your imputer pipeline
#     user_data_imputed = imputer.transform(user_data)
    
#     # Run Inference
#     prediction = int(model.predict(user_data_imputed)[0])
#     probability = float(model.predict_proba(user_data_imputed)[0][1]) # Probability of High Risk
    
#     # Return structured JSON output back to the client
#     return {
#         "status": "success",
#         "prediction": prediction,
#         "high_risk_probability": round(probability, 4),
#         "risk_label": "High Risk" if prediction == 1 else "Low Risk"
#     }

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId # Import this to handle MongoDB IDs safely

app = FastAPI(title="CardioCare Predictor Engine")

# --- MONGODB CONFIGURATION ---
# Paste your actual connection string from Atlas below! 
# Swap out <password> with your real database user password.
MONGO_URI = "mongodb+srv://minshac13_db_user:db_123_password@cluster0.oap4ftf.mongodb.net/?appName=Cluster0"

# Initialize the MongoClient to hold an open connection pool to the cloud
client = MongoClient(MONGO_URI)

# Select a database and a collection. MongoDB will automatically create these if they don't exist yet!
db = client["cardiocare_db"]
predictions_collection = db["patient_predictions"]
# ------------------------------

model = joblib.load('cardio_model.pkl')
imputer = joblib.load('cardio_imputer.pkl')

class PatientInput(BaseModel):
    age_years: float
    height: float
    weight: float
    systolic_bp: float
    diastolic_bp: float
    cholesterol: int

@app.post("/predict")
def predict_cardio_risk(patient: PatientInput):
    # Preprocessing
    age_days = patient.age_years * 365.25
    user_data = pd.DataFrame(
        [[age_days, patient.height, patient.weight, patient.systolic_bp, patient.diastolic_bp, patient.cholesterol]], 
        columns=['age_years', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 'cholesterol']
    )
    
    # Process through imputer and formatting dataframe for sklearn to clear warnings
    user_data_imputed = imputer.transform(user_data)
    user_data_imputed_df = pd.DataFrame(user_data_imputed, columns=user_data.columns)
    
    # Inference
    prediction = int(model.predict(user_data_imputed_df)[0])
    probability = float(model.predict_proba(user_data_imputed_df)[0][1])
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    
    # --- NO_SQL DATABASE LOGGING ---
    # We construct a clean, nested document (Python dictionary) representing the interaction
    log_document = {
        "timestamp": datetime.utcnow(),
        "input_features": {
            "age_years": patient.age_years,
            "height": patient.height,
            "weight": patient.weight,
            "systolic_bp": patient.systolic_bp,
            "diastolic_bp": patient.diastolic_bp,
            "cholesterol": patient.cholesterol
        },
        "inference_results": {
            "prediction": prediction,
            "high_risk_probability": round(probability, 4),
            "risk_label": risk_label
        }
    }
    
    # Push the document directly to our cloud database collection
    predictions_collection.insert_one(log_document)
    # -------------------------------

    return {
        "status": "success",
        "prediction": prediction,
        "high_risk_probability": round(probability, 4),
        "risk_label": risk_label
    }


# @app.get("/assessments")
# async def get_assessments():
#     try:
#         # Fetch all documents from your collection, sorting by newest first
#         cursor = predictions_collection.find().sort("_id", -1) 
        
#         assessments = []
#         for doc in await cursor.to_list(length=100): # Adjust length limit as needed
#             # MongoDB's default '_id' is an ObjectId, which FastAPI can't serialize to JSON.
#             # We convert it to a string, or just remove it before sending.
#             doc["id"] = str(doc["_id"])
#             del doc["_id"]
#             assessments.append(doc)
            
#         return {"status": "success", "data": assessments}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
    
# @app.get("/assessments")
# async def get_assessments():
#     try:
#         # 1. Fetch data from collection
#         cursor = predictions_collection.find().sort("_id", -1)
        
#         # 2. Extract documents into a list (Crucial if using Motor)
#         # If your app is async, use: await cursor.to_list(length=100)
#         # If your app is synchronous (standard pymongo), use: list(cursor)
#         documents = await cursor.to_list(length=100) 
        
#         assessments = []
#         for doc in documents:
#             # 3. Safely convert MongoDB's ObjectId to a string
#             doc["id"] = str(doc["_id"])
#             del doc["_id"]
#             assessments.append(doc)
            
#         print(f"Backend fetched {len(assessments)} documents.") # Debug line in your Docker logs
#         return {"status": "success", "data": assessments}
        
#     except Exception as e:
#         print(f"Database error: {str(e)}")
#         return {"status": "error", "message": str(e)}
    
@app.get("/assessments")
def get_assessments(): # 1. Removed 'async' from the function def since it's synchronous PyMongo
    try:
        # 2. Fetch the documents from your collection
        cursor = predictions_collection.find().sort("_id", -1)
        
        # 3. Convert the cursor straight into a Python list without 'await'
        documents = list(cursor) 
        
        assessments = []
        for doc in documents:
            # 4. Safely convert MongoDB's ObjectId to a string
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            assessments.append(doc)
            
        print(f"Backend fetched {len(assessments)} documents successfully!") 
        return {"status": "success", "data": assessments}
        
    except Exception as e:
        print(f"Database error: {str(e)}")
        return {"status": "error", "message": str(e)}