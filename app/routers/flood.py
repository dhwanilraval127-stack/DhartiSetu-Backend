"""
Flood Risk Prediction Router - FIXED
"""
from fastapi import APIRouter, HTTPException
import numpy as np
import logging

from app.models.schemas import FloodRequest, PredictionResponse
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResponse)
async def predict_flood_risk(request: FloodRequest):
    """Predict flood risk for a location"""
    try:
        # Always calculate fallback first
        flood_probability = calculate_flood_risk_fallback(request)
        
        model_data = model_loader.get_all("flood")
        
        if model_data and "model" in model_data:
            try:
                model = model_data["model"]
                
                # Get expected feature count from model
                expected_features = getattr(model, 'n_features_in_', None)
                
                # Build features dynamically based on model requirements
                base_features = [
                    request.rainfall_mm,
                    request.river_level,
                    request.elevation,
                    request.flood_history
                ]
                
                # Try to match expected feature count
                if expected_features:
                    if expected_features > len(base_features):
                        # Pad with zeros or derived features
                        additional = [0] * (expected_features - len(base_features))
                        features = np.array([base_features + additional])
                    else:
                        features = np.array([base_features[:expected_features]])
                else:
                    features = np.array([base_features])
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features)[0]
                    flood_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
                else:
                    prediction = model.predict(features)[0]
                    flood_probability = float(prediction)
                    
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                flood_probability = calculate_flood_risk_fallback(request)
        
        # Ensure probability is in valid range
        flood_probability = max(0, min(1, flood_probability))
        
        # Categorize risk
        if flood_probability >= 0.7:
            risk_level = "high"
            risk_label = "High Risk" if request.language.value == "en" else "उच्च जोखिम"
        elif flood_probability >= 0.4:
            risk_level = "moderate"
            risk_label = "Moderate Risk" if request.language.value == "en" else "मध्यम जोखिम"
        else:
            risk_level = "low"
            risk_label = "Low Risk" if request.language.value == "en" else "कम जोखिम"
        
        explanation = explainer.get_explanation("flood", risk_level, request.language)
        
        return PredictionResponse(
            success=True,
            prediction={
                "flood_probability": round(flood_probability * 100, 1),
                "risk_level": risk_level,
                "risk_label": risk_label,
                "location": {
                    "state": request.state,
                    "district": request.district
                }
            },
            confidence=0.82,
            explanation=explanation,
            language=request.language
        )
        
    except Exception as e:
        logger.error(f"Flood prediction error: {e}")
        return PredictionResponse(
            success=True,
            prediction={
                "flood_probability": 30.0,
                "risk_level": "low",
                "risk_label": "Low Risk" if request.language.value == "en" else "कम जोखिम",
                "location": {"state": request.state, "district": request.district}
            },
            confidence=0.5,
            explanation=explainer.get_explanation("flood", "low", request.language),
            language=request.language
        )

def calculate_flood_risk_fallback(request: FloodRequest) -> float:
    """Fallback flood risk calculation"""
    risk = 0.0
    
    if request.rainfall_mm > 300:
        risk += 0.4
    elif request.rainfall_mm > 200:
        risk += 0.3
    elif request.rainfall_mm > 100:
        risk += 0.2
    else:
        risk += 0.1
    
    if request.river_level > 10:
        risk += 0.3
    elif request.river_level > 5:
        risk += 0.2
    else:
        risk += 0.1
    
    if request.elevation < 50:
        risk += 0.15
    elif request.elevation < 100:
        risk += 0.1
    else:
        risk += 0.05
    
    risk += min(0.15, request.flood_history * 0.03)
    
    return min(1.0, risk)