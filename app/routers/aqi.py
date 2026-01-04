"""
AQI Prediction Router - PRODUCTION SAFE
"""
from fastapi import APIRouter
import numpy as np
import logging

from app.models.schemas import AQIRequest, PredictionResponse
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)


def get_lang(lang):
    return lang.value if hasattr(lang, "value") else lang


def get_aqi_category(aqi_value: float) -> str:
    if aqi_value <= 50:
        return "good"
    elif aqi_value <= 100:
        return "moderate"
    elif aqi_value <= 150:
        return "unhealthy_sensitive"
    elif aqi_value <= 200:
        return "unhealthy"
    elif aqi_value <= 300:
        return "very_unhealthy"
    return "hazardous"


def get_aqi_label(category: str, language) -> str:
    labels = {
        "good": {"en": "Good", "hi": "अच्छा"},
        "moderate": {"en": "Moderate", "hi": "मध्यम"},
        "unhealthy_sensitive": {"en": "Unhealthy for Sensitive Groups", "hi": "संवेदनशील समूहों के लिए अस्वस्थ"},
        "unhealthy": {"en": "Unhealthy", "hi": "अस्वस्थ"},
        "very_unhealthy": {"en": "Very Unhealthy", "hi": "बहुत अस्वस्थ"},
        "hazardous": {"en": "Hazardous", "hi": "खतरनाक"},
    }
    lang = get_lang(language)
    return labels.get(category, labels["moderate"]).get(lang, "Moderate")


@router.post("/predict", response_model=PredictionResponse)
async def predict_aqi(request: AQIRequest):
    try:
        model_data = model_loader.get_all("aqi")

        if model_data and "model" in model_data:
            model = model_data["model"]
            encoder = model_data.get("encoder")

            features = np.array([[
                request.pm25,
                request.pm10,
                request.no2,
                request.so2,
                request.co,
                request.o3,
                request.temperature,
                request.humidity,
                request.wind_speed,
                request.pressure,
                request.visibility,
                0.0
            ]], dtype=float)

            if encoder and hasattr(request, "city"):
                try:
                    features[0][-1] = encoder.transform([request.city])[0]
                except Exception:
                    pass

            aqi_value = float(model.predict(features)[0])
        else:
            aqi_value = calculate_aqi_fallback(request)

    except Exception as e:
        logger.error("AQI ML failed, using fallback", exc_info=True)
        aqi_value = calculate_aqi_fallback(request)

    category = get_aqi_category(aqi_value)
    explanation = explainer.get_explanation("aqi", category, request.language)

    return PredictionResponse(
        success=True,
        prediction={
            "aqi_value": round(aqi_value, 1),
            "category": get_aqi_label(category, request.language),
            "category_code": category,
            "city": request.city,
        },
        confidence=0.8,
        explanation=explanation,
        language=request.language,
    )
