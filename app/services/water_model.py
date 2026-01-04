"""
Water Requirement ML Model Service - IMPORT SAFE
"""
import logging
import numpy as np
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# -------------------------------------------------
# MODEL PATH
# -------------------------------------------------
MODEL_PATH = Path("app/models/water/water_model.pkl")

# -------------------------------------------------
# SAFE MODEL LOAD (NEVER RAISE AT IMPORT)
# -------------------------------------------------
model = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Water ML model loaded successfully")
    else:
        logger.warning("⚠️ Water ML model file not found")
except Exception as e:
    logger.error(f"❌ Failed to load water ML model: {e}")
    model = None


# -------------------------------------------------
# FEATURE PREPARATION
# -------------------------------------------------
def prepare_features(request) -> np.ndarray:
    """
    Convert WaterRequest → ML feature array
    MUST match training order
    """
    return np.array([[
        request.temperature,
        request.humidity,
        request.rainfall,
        request.soil_moisture,
        request.crop_coefficient,
        request.growth_stage_index
    ]])


# -------------------------------------------------
# SAFE PREDICTION FUNCTION
# -------------------------------------------------
def predict_water_requirement(X: np.ndarray):
    """
    Predict water requirement safely.
    NEVER raises if model missing.
    """
    if model is None:
        return {"error": "model_not_loaded"}

    try:
        return model.predict(X)
    except Exception as e:
        logger.error(f"Water ML inference failed: {e}")
        return {"error": "prediction_failed"}
