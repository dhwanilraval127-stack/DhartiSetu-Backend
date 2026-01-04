"""
Crop Yield Prediction Router - FIXED & COMPLETE
"""
from fastapi import APIRouter
import numpy as np
import logging

from app.models.schemas import YieldRequest, PredictionResponse
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)

# -------------------------
# CONSTANT DATA
# -------------------------

CROPS = [
    "Rice",
    "Wheat",
    "Maize",
    "Cotton",
    "Sugarcane",
    "Groundnut",
    "Soybean",
    "Potato",
    "Onion",
    "Tomato"
]

SEASONS = [
    {"en": "Kharif", "hi": "खरीफ", "value": "Kharif", "months": "June - October"},
    {"en": "Rabi", "hi": "रबी", "value": "Rabi", "months": "October - March"},
    {"en": "Zaid", "hi": "जायद", "value": "Zaid", "months": "March - June"},
    {"en": "Whole Year", "hi": "पूर्ण वर्ष", "value": "Whole Year", "months": "All Year"}
]

# -------------------------
# GET CROPS (🔥 REQUIRED)
# -------------------------

@router.get("/crops")
async def get_crop_list():
    return {
        "crops": CROPS
    }

# -------------------------
# GET SEASONS
# -------------------------

@router.get("/seasons")
async def get_seasons():
    return {
        "seasons": SEASONS
    }

# -------------------------
# PREDICT YIELD
# -------------------------

@router.post("/predict", response_model=PredictionResponse)
async def predict_yield(request: YieldRequest):
    try:
        yield_prediction = estimate_yield_fallback(request)

        model_data = model_loader.get_all("yield")
        if model_data and "model" in model_data:
            try:
                model = model_data["model"]
                features = np.array([[request.area_hectares]])
                yield_prediction = float(model.predict(features)[0])
            except Exception as e:
                logger.warning(f"Model failed, fallback used: {e}")

        yield_prediction = max(500, yield_prediction)
        total_production = yield_prediction * request.area_hectares

        if yield_prediction > 3000:
            category = "high"
            label = "High Yield"
        elif yield_prediction > 1500:
            category = "moderate"
            label = "Moderate Yield"
        else:
            category = "low"
            label = "Low Yield"

        return PredictionResponse(
            success=True,
            prediction={
                "yield_per_hectare_kg": round(yield_prediction, 2),
                "total_production_kg": round(total_production, 2),
                "yield_per_hectare_quintal": round(yield_prediction / 100, 2),
                "total_production_quintal": round(total_production / 100, 2),
                "yield_category": category,
                "yield_label": label,
                "details": {
                    "crop": request.crop,
                    "state": request.state,
                    "district": request.district,
                    "season": request.season,
                    "area_hectares": request.area_hectares
                }
            },
            confidence=0.79,
            explanation=explainer.get_explanation("yield", "default", request.language),
            language=request.language
        )

    except Exception as e:
        logger.error(f"Yield prediction error: {e}")
        raise

# -------------------------
# FALLBACK LOGIC
# -------------------------

def estimate_yield_fallback(request: YieldRequest) -> float:
    avg_yields = {
        "rice": 2500,
        "wheat": 3000,
        "maize": 2800,
        "cotton": 400,
        "sugarcane": 70000,
        "groundnut": 1500,
        "soybean": 1000,
        "potato": 20000,
        "onion": 15000,
        "tomato": 20000
    }

    base = avg_yields.get(request.crop.lower(), 2000)
    return base
