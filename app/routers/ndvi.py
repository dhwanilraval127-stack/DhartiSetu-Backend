"""
NDVI (Normalized Difference Vegetation Index) Router
"""
from fastapi import APIRouter, HTTPException
import numpy as np
import logging

from app.models.schemas import NDVIRequest, PredictionResponse
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=PredictionResponse)
async def analyze_ndvi(request: NDVIRequest):
    """Analyze vegetation health using NDVI"""
    try:
        # Calculate NDVI
        if request.nir_band + request.red_band == 0:
            ndvi = 0
        else:
            ndvi = (request.nir_band - request.red_band) / (request.nir_band + request.red_band)
        
        model_data = model_loader.get_all("ndvi")
        
        if model_data and "model" in model_data:
            model = model_data["model"]
            
            # Prepare features for additional analysis
            features = np.array([[
                ndvi, request.temperature, request.rainfall
            ]])
            
            # Predict vegetation health score
            health_score = float(model.predict(features)[0])
        else:
            # Fallback calculation
            health_score = calculate_health_score_fallback(ndvi, request)
        
        # Determine vegetation status
        if ndvi >= 0.6:
            status = "high"
            status_label = "Dense Healthy Vegetation" if request.language.value == "en" else "घनी स्वस्थ वनस्पति"
        elif ndvi >= 0.3:
            status = "moderate"
            status_label = "Moderate Vegetation" if request.language.value == "en" else "मध्यम वनस्पति"
        elif ndvi >= 0.1:
            status = "low"
            status_label = "Sparse Vegetation" if request.language.value == "en" else "विरल वनस्पति"
        else:
            status = "very_low"
            status_label = "Bare Soil / No Vegetation" if request.language.value == "en" else "नंगी मिट्टी / कोई वनस्पति नहीं"
        
        explanation = explainer.get_explanation("ndvi", status, request.language)
        
        return PredictionResponse(
            success=True,
            prediction={
                "ndvi_value": round(ndvi, 4),
                "health_score": round(health_score, 2),
                "status": status,
                "status_label": status_label,
                "interpretation": get_ndvi_interpretation(ndvi, request.language)
            },
            confidence=0.90,
            explanation=explanation,
            language=request.language
        )
        
    except Exception as e:
        logger.error(f"NDVI analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_health_score_fallback(ndvi: float, request: NDVIRequest) -> float:
    """Fallback health score calculation"""
    base_score = (ndvi + 1) / 2 * 100  # Convert -1 to 1 range to 0-100
    
    # Adjust for temperature
    if 20 <= request.temperature <= 30:
        base_score *= 1.1
    elif request.temperature < 10 or request.temperature > 40:
        base_score *= 0.8
    
    # Adjust for rainfall
    if 50 <= request.rainfall <= 200:
        base_score *= 1.05
    elif request.rainfall < 20:
        base_score *= 0.85
    
    return min(100, max(0, base_score))

def get_ndvi_interpretation(ndvi: float, language) -> str:
    """Get NDVI interpretation"""
    if language.value == "en":
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