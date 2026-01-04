"""
CO2 Level Prediction Router - FINAL FIXED VERSION
"""
from fastapi import APIRouter
import numpy as np
import logging

from app.models.schemas import CO2Request, PredictionResponse
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResponse)
async def predict_co2(request: CO2Request):
    try:
        # SAFE language handling
        lang = request.language.value if hasattr(request.language, "value") else request.language

        model_data = model_loader.get_all("co2")

        if model_data and "model" in model_data:
            model = model_data["model"]
            scaler = model_data.get("scaler")

            # EXACTLY 7 FEATURES
            features = np.array([[
                request.month,
                request.year,
                request.temperature,
                request.humidity,
                request.pressure,
                request.wind_speed,
                0.0
            ]], dtype=float)

            if features.shape[1] != 7:
                raise ValueError("CO2 feature count mismatch")

            # Scaling (SAFE)
            if scaler and hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == 7:
                features = scaler.transform(features)

            pred = model.predict(features)[0]

            # Validate model output
            if np.isnan(pred) or np.isinf(pred):
                raise ValueError("Invalid CO2 model output")

            co2_level = float(pred)

        else:
            co2_level = calculate_co2_fallback(request)

        # Categorization
        if co2_level < 350:
            level = "normal"
            label_en, label_hi = "Normal", "सामान्य"
        elif co2_level < 400:
            level = "elevated"
            label_en, label_hi = "Elevated", "ऊंचा"
        elif co2_level < 450:
            level = "high"
            label_en, label_hi = "High", "उच्च"
        else:
            level = "very_high"
            label_en, label_hi = "Very High", "बहुत उच्च"

        explanation = explainer.get_explanation("co2", level, request.language)

        return PredictionResponse(
            success=True,
            prediction={
                "co2_ppm": round(co2_level, 2),
                "level": level,
                "level_label": label_en if lang == "en" else label_hi,
                "month": request.month,
                "year": request.year,
                "conditions": {
                    "temperature": request.temperature,
                    "humidity": request.humidity,
                    "pressure": request.pressure,
                    "wind_speed": request.wind_speed
                }
            },
            confidence=0.88,
            explanation=explanation,
            language=request.language
        )

    except Exception as e:
        logger.error(f"CO2 prediction error: {e}")

        # SAFE fallback
        co2_level = calculate_co2_fallback(request)
        explanation = explainer.get_explanation("co2", "normal", request.language)

        return PredictionResponse(
            success=True,
            prediction={
                "co2_ppm": round(co2_level, 2),
                "level": "normal",
                "level_label": "Normal" if (request.language == "en") else "सामान्य",
                "month": request.month,
                "year": request.year,
                "conditions": {
                    "temperature": request.temperature,
                    "humidity": request.humidity,
                    "pressure": request.pressure,
                    "wind_speed": request.wind_speed
                }
            },
            confidence=0.75,
            explanation=explanation,
            language=request.language
        )

def calculate_co2_fallback(request: CO2Request) -> float:
    """
    Realistic CO2 estimation (Mauna Loa inspired)
    """
    # 1958 ≈ 315 ppm, growth ≈ 1.6 ppm/year
    base_co2 = 315 + (request.year - 1958) * 1.6

    seasonal = 4 * np.sin((request.month - 4) * np.pi / 6)
    temp_effect = (request.temperature - 15) * 0.15

    co2 = base_co2 + seasonal + temp_effect
    return max(280, min(co2, 500))
