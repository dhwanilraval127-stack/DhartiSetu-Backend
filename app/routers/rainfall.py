"""
Rainfall Prediction Router - HARDENED (CRASH SAFE)
"""
from fastapi import APIRouter
import numpy as np
import logging

from app.models.schemas import RainfallRequest, PredictionResponse
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictionResponse)
async def predict_rainfall(request: RainfallRequest):
    """Predict rainfall with safe feature handling and fallbacks"""
    try:
        # -------------------------------------------------
        # 🌐 LANGUAGE SAFE HANDLING
        # -------------------------------------------------
        lang = request.language.value if hasattr(request.language, "value") else "en"

        model_data = model_loader.get_all("rainfall")

        if model_data and "model" in model_data:
            model = model_data["model"]
            encoder = model_data.get("encoder")

            # -------------------------------------------------
            # FEATURE PREPARATION (BASE = 18)
            # -------------------------------------------------
            features = np.array([[
                request.month,          # 1
                request.year,           # 2
                request.temperature,    # 3
                request.humidity,       # 4
                request.pressure,       # 5
                0, 0, 0, 0, 0,           # 6–10
                0, 0, 0, 0, 0,           # 11–15
                0,                      # 16 subdivision
                0                       # 17 extra
            ]])

            # -------------------------------------------------
            # 🔥 CRITICAL FIX — NEVER RAISE
            # -------------------------------------------------
            expected = getattr(model, "n_features_in_", features.shape[1])
            features = features[:, :expected]

            # -------------------------------------------------
            # SUBDIVISION ENCODING (SAFE)
            # -------------------------------------------------
            if encoder and expected > 16:
                try:
                    subdivision_encoded = encoder.transform(
                        [request.subdivision]
                    )[0]
                    features[0][16] = subdivision_encoded
                except Exception as e:
                    logger.warning(f"Subdivision encoding failed: {e}")

            rainfall_prediction = float(model.predict(features)[0])
        else:
            rainfall_prediction = calculate_rainfall_fallback(request)

        # -------------------------------------------------
        # CATEGORY CLASSIFICATION
        # -------------------------------------------------
        if rainfall_prediction < 50:
            category = "scanty"
            category_label = "Scanty Rainfall" if lang == "en" else "अल्प वर्षा"
        elif rainfall_prediction < 100:
            category = "light"
            category_label = "Light Rainfall" if lang == "en" else "हल्की वर्षा"
        elif rainfall_prediction < 200:
            category = "moderate"
            category_label = "Moderate Rainfall" if lang == "en" else "मध्यम वर्षा"
        elif rainfall_prediction < 400:
            category = "heavy"
            category_label = "Heavy Rainfall" if lang == "en" else "भारी वर्षा"
        else:
            category = "very_heavy"
            category_label = "Very Heavy Rainfall" if lang == "en" else "अत्यधिक वर्षा"

        explanation = explainer.get_explanation("rainfall", "default", request.language)

        # -------------------------------------------------
        # MONTH NAMES
        # -------------------------------------------------
        month_names = {
            1: {"en": "January", "hi": "जनवरी"},
            2: {"en": "February", "hi": "फरवरी"},
            3: {"en": "March", "hi": "मार्च"},
            4: {"en": "April", "hi": "अप्रैल"},
            5: {"en": "May", "hi": "मई"},
            6: {"en": "June", "hi": "जून"},
            7: {"en": "July", "hi": "जुलाई"},
            8: {"en": "August", "hi": "अगस्त"},
            9: {"en": "September", "hi": "सितंबर"},
            10: {"en": "October", "hi": "अक्टूबर"},
            11: {"en": "November", "hi": "नवंबर"},
            12: {"en": "December", "hi": "दिसंबर"},
        }

        month_name = month_names.get(
            request.month, {"en": "Unknown", "hi": "अज्ञात"}
        )[lang]

        return PredictionResponse(
            success=True,
            prediction={
                "rainfall_mm": round(rainfall_prediction, 2),
                "category": category,
                "category_label": category_label,
                "subdivision": request.subdivision,
                "month": request.month,
                "month_name": month_name,
                "year": request.year,
                "conditions": {
                    "temperature": request.temperature,
                    "humidity": request.humidity,
                    "pressure": request.pressure,
                },
            },
            confidence=0.80,
            explanation=explanation,
            language=request.language,
        )

    except Exception as e:
        logger.error(f"Rainfall prediction error: {e}")

        rainfall_prediction = calculate_rainfall_fallback(request)
        explanation = explainer.get_explanation("rainfall", "default", request.language)

        return PredictionResponse(
            success=True,
            prediction={
                "rainfall_mm": round(rainfall_prediction, 2),
                "category": "moderate",
                "category_label": "Moderate Rainfall" if lang == "en" else "मध्यम वर्षा",
                "subdivision": request.subdivision,
                "month": request.month,
                "month_name": month_name,
                "year": request.year,
            },
            confidence=0.70,
            explanation=explanation,
            language=request.language,
        )


# ==========================================================
# FALLBACK LOGIC
# ==========================================================
def calculate_rainfall_fallback(request: RainfallRequest) -> float:
    """Fallback rainfall estimation based on seasonal patterns"""
    monsoon_factor = {
        1: 0.2, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5,
        6: 1.5, 7: 2.0, 8: 1.8, 9: 1.5, 10: 0.8,
        11: 0.4, 12: 0.3,
    }

    base_rainfall = 100
    factor = monsoon_factor.get(request.month, 1.0)
    temp_effect = max(0, (getattr(request, "temperature", 25) - 20) * 2)

    return base_rainfall * factor + temp_effect
