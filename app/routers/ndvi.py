"""
NDVI (Normalized Difference Vegetation Index) Router - FIXED
"""
from fastapi import APIRouter, HTTPException
import numpy as np
import logging

from app.models.schemas import NDVIRequest, PredictionResponse
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)

# =========================================================
# NDVI ANALYSIS
# =========================================================
@router.post("/analyze", response_model=PredictionResponse)
async def analyze_ndvi(request: NDVIRequest):
    try:
        # -------------------------------------------------
        # LANGUAGE SAFE HANDLING
        # -------------------------------------------------
        lang = request.language.value if hasattr(request.language, "value") else "en"

        # -------------------------------------------------
        # NDVI CALCULATION
        # -------------------------------------------------
        if request.nir_band + request.red_band == 0:
            ndvi = 0.0
        else:
            ndvi = (request.nir_band - request.red_band) / (
                request.nir_band + request.red_band
            )

        # -------------------------------------------------
        # MODEL / FALLBACK
        # -------------------------------------------------
        model_data = model_loader.get_all("ndvi")

        if model_data and "model" in model_data:
            try:
                model = model_data["model"]
                features = np.array([[ndvi, request.temperature, request.rainfall]])
                health_score = float(model.predict(features)[0])
            except Exception as e:
                logger.warning(f"NDVI model failed, using fallback: {e}")
                health_score = calculate_health_score_fallback(ndvi, request)
        else:
            health_score = calculate_health_score_fallback(ndvi, request)

        # -------------------------------------------------
        # VEGETATION STATUS
        # -------------------------------------------------
        if ndvi >= 0.6:
            status = "high"
            status_label = (
                "Dense Healthy Vegetation" if lang == "en" else "घनी स्वस्थ वनस्पति"
            )
        elif ndvi >= 0.3:
            status = "moderate"
            status_label = (
                "Moderate Vegetation" if lang == "en" else "मध्यम वनस्पति"
            )
        elif ndvi >= 0.1:
            status = "low"
            status_label = (
                "Sparse Vegetation" if lang == "en" else "विरल वनस्पति"
            )
        else:
            status = "very_low"
            status_label = (
                "Bare Soil / No Vegetation"
                if lang == "en"
                else "नंगी मिट्टी / कोई वनस्पति नहीं"
            )

        explanation = explainer.get_explanation("ndvi", status, request.language)

        return PredictionResponse(
            success=True,
            prediction={
                "ndvi_value": round(ndvi, 4),
                "health_score": round(health_score, 2),
                "status": status,
                "status_label": status_label,
                "interpretation": get_ndvi_interpretation(ndvi, lang),
            },
            confidence=0.90,
            explanation=explanation,
            language=request.language,
        )

    except Exception as e:
        logger.error(f"NDVI analysis error: {e}")
        raise HTTPException(status_code=500, detail="NDVI analysis error")


# =========================================================
# FALLBACK HEALTH SCORE
# =========================================================
def calculate_health_score_fallback(ndvi: float, request: NDVIRequest) -> float:
    """Fallback health score calculation"""
    base_score = (ndvi + 1) / 2 * 100  # -1..1 → 0..100

    # Temperature adjustment
    if 20 <= request.temperature <= 30:
        base_score *= 1.1
    elif request.temperature < 10 or request.temperature > 40:
        base_score *= 0.8

    # Rainfall adjustment
    if 50 <= request.rainfall <= 200:
        base_score *= 1.05
    elif request.rainfall < 20:
        base_score *= 0.85

    return min(100, max(0, base_score))


# =========================================================
# NDVI INTERPRETATION
# =========================================================
def get_ndvi_interpretation(ndvi: float, lang: str) -> str:
    """Get NDVI interpretation"""
    if lang == "en":
        if ndvi >= 0.6:
            return "Excellent crop health. Plants are thriving with good chlorophyll content."
        elif ndvi >= 0.3:
            return "Good vegetation. Crops are growing normally but monitor for stress."
        elif ndvi >= 0.1:
            return "Sparse vegetation detected. May indicate early stage crops or stress."
        else:
            return "Very low vegetation. Could be bare soil, harvested field, or severe stress."
    else:
        if ndvi >= 0.6:
            return "उत्कृष्ट फसल स्वास्थ्य। पौधे अच्छे क्लोरोफिल के साथ पनप रहे हैं।"
        elif ndvi >= 0.3:
            return "अच्छी वनस्पति। फसलें सामान्य रूप से बढ़ रही हैं लेकिन तनाव के लिए निगरानी करें।"
        elif ndvi >= 0.1:
            return "विरल वनस्पति का पता चला। प्रारंभिक चरण की फसलें या तनाव हो सकता है।"
        else:
            return "बहुत कम वनस्पति। नंगी मिट्टी, कटी हुई फसल, या गंभीर तनाव हो सकता है।"
