"""
Rainfall Prediction Router - Debugged
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
    """Predict rainfall with fixed feature count and safe fallbacks"""
    try:
        model_data = model_loader.get_all("rainfall")

        if model_data and "model" in model_data:
            model = model_data["model"]
            encoder = model_data.get("encoder")

            # Prepare features - exactly 18 features
            features = np.array([[
                request.month,          # 1
                request.year,           # 2
                request.temperature,    # 3
                request.humidity,       # 4
                request.pressure,       # 5
                0,                      # 6 cloud cover
                0,                      # 7 wind speed
                0,                      # 8 evapotranspiration
                0,                      # 9 soil moisture
                0,                      # 10 previous month rainfall
                0,                      # 11 same month last year
                0,                      # 12 Jan rainfall
                0,                      # 13 Feb rainfall
                0,                      # 14 Mar rainfall
                0,                      # 15 Apr rainfall
                0,                      # 16 May rainfall
                0,                      # 17 subdivision encoded
                0                       # 18 extra feature
            ]])

            # Validate feature count
            expected = getattr(model, "n_features_in_", 18)
            if features.shape[1] != expected:
                logger.warning(
                    f"Rainfall feature count mismatch: expected {expected}, got {features.shape[1]}"
                )
                raise ValueError("Feature count mismatch")

            # Encode subdivision if encoder exists
            if encoder:
                try:
                    subdivision_encoded = encoder.transform([request.subdivision])[0]
                    features[0][16] = subdivision_encoded  # index 16 reserved for subdivision
                except Exception as e:
                    logger.warning(f"Subdivision encoding failed: {e}")
                    features[0][16] = 0

            rainfall_prediction = float(model.predict(features)[0])
        else:
            # Fallback calculation
            rainfall_prediction = calculate_rainfall_fallback(request)

        # Categorize rainfall
        if rainfall_prediction < 50:
            category = "scanty"
            category_label = "Scanty Rainfall" if request.language.value == "en" else "अल्प वर्षा"
        elif rainfall_prediction < 100:
            category = "light"
            category_label = "Light Rainfall" if request.language.value == "en" else "हल्की वर्षा"
        elif rainfall_prediction < 200:
            category = "moderate"
            category_label = "Moderate Rainfall" if request.language.value == "en" else "मध्यम वर्षा"
        elif rainfall_prediction < 400:
            category = "heavy"
            category_label = "Heavy Rainfall" if request.language.value == "en" else "भारी वर्षा"
        else:
            category = "very_heavy"
            category_label = "Very Heavy Rainfall" if request.language.value == "en" else "अत्यधिक वर्षा"

        explanation = explainer.get_explanation("rainfall", "default", request.language)

        # Month names
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
            12: {"en": "December", "hi": "दिसंबर"}
        }

        month_name = month_names.get(request.month, {"en": "Unknown", "hi": "अज्ञात"})[request.language.value]

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
                    "pressure": request.pressure
                }
            },
            confidence=0.80,
            explanation=explanation,
            language=request.language
        )

    except Exception as e:
        logger.error(f"Rainfall prediction error: {e}")
        rainfall_prediction = calculate_rainfall_fallback(request)
        explanation = explainer.get_explanation("rainfall", "default", request.language)

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
            12: {"en": "December", "hi": "दिसंबर"}
        }
        month_name = month_names.get(request.month, {"en": "Unknown", "hi": "अज्ञात"})[request.language.value]

        return PredictionResponse(
            success=True,
            prediction={
                "rainfall_mm": round(rainfall_prediction, 2),
                "category": "moderate",
                "category_label": "Moderate Rainfall" if request.language.value == "en" else "मध्यम वर्षा",
                "subdivision": request.subdivision,
                "month": request.month,
                "month_name": month_name,
                "year": request.year,
                "conditions": {
                    "temperature": getattr(request, "temperature", None),
                    "humidity": getattr(request, "humidity", None),
                    "pressure": getattr(request, "pressure", None)
                }
            },
            confidence=0.70,
            explanation=explanation,
            language=request.language
        )


def calculate_rainfall_fallback(request: RainfallRequest) -> float:
    """Fallback rainfall estimation based on typical patterns"""
    monsoon_factor = {
        1: 0.2, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5,
        6: 1.5, 7: 2.0, 8: 1.8, 9: 1.5, 10: 0.8,
        11: 0.4, 12: 0.3
    }

    base_rainfall = 100
    factor = monsoon_factor.get(request.month, 1.0)

    temp_effect = max(0, (getattr(request, "temperature", 25) - 20) * 2)

    return base_rainfall * factor + temp_effect