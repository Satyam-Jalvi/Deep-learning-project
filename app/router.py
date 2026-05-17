# app/router.py

from fastapi import APIRouter
import joblib
import pandas as pd

from src.config.heart_config import CONFIG as HEART_CONFIG
from src.config.breast_config import CONFIG as BREAST_CONFIG
from src.config.diabetes_config import CONFIG as DIABETES_CONFIG
from src.config.dermatology_config import CONFIG as DERMATOLOGY_CONFIG

router = APIRouter()

# -------------------------
# LOAD MODELS (ON START)
# -------------------------
models = {
    "heart": joblib.load("models/heart_model.pkl"),
    "breast": joblib.load("models/breast_model.pkl"),
    "diabetes": joblib.load("models/diabetes_model.pkl"),
    "dermatology": joblib.load("models/dermatology_model.pkl")
}

configs = {
    "heart": HEART_CONFIG,
    "breast": BREAST_CONFIG,
    "diabetes": DIABETES_CONFIG,
    "dermatology": DERMATOLOGY_CONFIG
}


# -------------------------
# ROOT
# -------------------------
@router.get("/")
def home():
    return {"message": "Medical Diagnosis API is running"}


# -------------------------
# GET AVAILABLE DISEASES
# -------------------------
@router.get("/diseases")
def get_diseases():
    return {"available": list(models.keys())}


# -------------------------
# PREDICTION
# -------------------------
@router.post("/predict/{disease}")
def predict(disease: str, input_data: dict):

    if disease not in models:
        return {"error": "Invalid disease"}

    model = models[disease]

    # convert to dataframe
    df = pd.DataFrame([input_data])

    # prediction
    pred = model.predict(df)[0]

    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df)[0].max())

    # FIX HERE
    if isinstance(pred, str):
        pred_out = 1 if pred.lower() in ["yes", "m"] else 0
    else:
        pred_out = int(pred)

    label_map = {

        "heart": {
            0: "No Heart Disease",
            1: "Heart Disease"
        },

        "breast": {
            0: "Benign",
            1: "Malignant"
        },

        "diabetes": {
            0: "Not Diabetic",
            1: "Diabetic"
        },

        "dermatology": {

            1: "Psoriasis",

            2: "Seborrheic Dermatitis",

            3: "Lichen Planus",

            4: "Pityriasis Rosea",

            5: "Chronic Dermatitis",

            6: "Pityriasis Rubra Pilaris"
        }
    }

    return {
        "disease": disease,
        "prediction": pred_out,
        "label": label_map[disease][pred_out],
        "confidence": prob
    }