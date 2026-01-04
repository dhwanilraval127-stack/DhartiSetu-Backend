# app/config/settings.py
"""
DhartiSetu Configuration (Production Ready)
Compatible with Hugging Face + Koyeb + Vercel
"""

from pydantic_settings import BaseSettings
from typing import List, Dict


class Settings(BaseSettings):
    # --------------------------------------------------
    # APP INFO
    # --------------------------------------------------
    APP_NAME: str = "DhartiSetu"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # --------------------------------------------------
    # API SETTINGS
    # --------------------------------------------------
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "*"
    ]

    # --------------------------------------------------
    # IMAGE UPLOAD SETTINGS
    # --------------------------------------------------
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_IMAGE_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]

    # --------------------------------------------------
    # 🔑 MODEL PATHS (HUGGING FACE REPO RELATIVE PATHS)
    # --------------------------------------------------
    # These paths MUST match exactly what you uploaded to:
    # https://huggingface.co/crimson1232/dhartisetu-ml-models
    # --------------------------------------------------
    MODEL_PATHS: Dict[str, Dict[str, str]] = {
        "aqi": {
            "model": "aqi/aqi_xgb.pkl",
            "encoder": "aqi/city_encoder.pkl"
        },
        "co2": {
            "model": "co2/co2_model.pkl",
            "scaler": "co2/co2_scaler.pkl"
        },
        "crop": {
            "model": "crop/crop_xgboost.pkl",
            "scaler": "crop/crop_scaler.pkl",
            "encoder": "crop/crop_label_encoder.pkl"
        },
        "flood": {
            "model": "flood/flood_rf.pkl",
            "encoders": "flood/flood_encoders.pkl"
        },
        "ndvi": {
            "model": "ndvi/ndvi_rf.pkl"
        },
        "plant_disease": {
            "model": "plant_disease/plant_disease.h5"
        },
        "price": {
            "model": "price/crop_price_xgb.pkl",
            "encoders": "price/price_encoders.pkl"
        },
        "profit": {
            "model": "profit/crop_profit_model.pkl",
            "encoders": "profit/profit_encoders.pkl"
        },
        "rainfall": {
            "model": "rainfall/rainfall_rf.pkl",
            "encoder": "rainfall/subdivision_encoder.pkl"
        },
        "soil_cnn": {
            "model": "soil/soil_model.pkl"
        },
        "soil_health": {
            "model": "soil_health/soil_health_xgb.pkl",
            "scaler": "soil_health/scaler.pkl",
            "encoder": "soil_health/label_encoder.pkl"
        },
        "storm": {
            "model": "storm/storm_rf.pkl",
            "encoders": "storm/storm_encoders.pkl"
        },
        "water": {
            "model": "water/water_rf.pkl"
        },
        "yield": {
            "model": "yield/yield_xgb.pkl",
            "encoders": "yield/label_encoders.pkl"
        }
    }

    class Config:
        env_file = ".env"
        case_sensitive = True


# --------------------------------------------------
# SINGLETON SETTINGS INSTANCE
# --------------------------------------------------
settings = Settings()
