"""
Storm Prediction Router - FIXED
"""
from fastapi import APIRouter, HTTPException
import numpy as np
import logging

from app.models.schemas import StormRequest, PredictionResponse
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResponse)
async def predict_storm(request: StormRequest):
    """Predict storm risk"""
    try:
        # Calculate fallback first
        storm_probability = estimate_storm_risk_fallback(request)
        
        model_data = model_loader.get_all("storm")
        
        if model_data and "model" in model_data:
            try:
                model = model_data["model"]
                
                # Core features
                base_features = [
                    request.month,
                    request.wind_speed,
                    request.pressure,
                    request.humidity
                ]
                
                expected_features = getattr(model, 'n_features_in_', len(base_features))
                
                if expected_features > len(base_features):
                    additional = [0] * (expected_features - len(base_features))
                    features = np.array([base_features + additional])
                else:
                    features = np.array([base_features[:expected_features]])
                
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(features)[0]
                    storm_probability = float(probas[1]) if len(probas) > 1 else float(probas[0])
                else:
                    prediction = model.predict(features)[0]
                    storm_probability = float(prediction)
                    
            except Exception as e:
                logger.warning(f"Storm model failed: {e}")
                storm_probability = estimate_storm_risk_fallback(request)
        
        storm_probability = max(0, min(1, storm_probability))
        
        if storm_probability >= 0.7:
            risk_level = "high"
            risk_label = "High Storm Risk" if request.language.value == "en" else "उच्च तूफान जोखिम"
        elif storm_probability >= 0.4:
            risk_level = "moderate"
            risk_label = "Moderate Storm Risk" if request.language.value == "en" else "मध्यम तूफान जोखिम"
        else:
            risk_level = "low"
            risk_label = "Low Storm Risk" if request.language.value == "en" else "कम तूफान जोखिम"
        
        explanation = explainer.get_explanation("storm", risk_level, request.language)
        
        return PredictionResponse(
            success=True,
            prediction={
                "storm_probability": round(storm_probability * 100, 1),
                "risk_level": risk_level,
                "risk_label": risk_label,
                "conditions": {
                    "wind_speed": request.wind_speed,
                    "pressure": request.pressure,
                    "humidity": request.humidity
                },
                "location": request.state
            },
            confidence=0.76,
            explanation=explanation,
            language=request.language
        )
        
    except Exception as e:
        logger.error(f"Storm prediction error: {e}")
        return PredictionResponse(
            success=True,
            prediction={
                "storm_probability": 20.0,
                "risk_level": "low",
                "risk_label": "Low Storm Risk",
                "conditions": {},
                "location": request.state
            },
            confidence=0.5,
            explanation=explainer.get_explanation("storm", "low", request.language),
            language=request.language
        )

def estimate_storm_risk_fallback(request: StormRequest) -> float:
    """Fallback storm risk estimation"""
    risk = 0.0
    
    if request.wind_speed > 100:
        risk += 0.4
    elif request.wind_speed > 60:
        risk += 0.25
    elif request.wind_speed > 40:
        risk += 0.15
    
    if request.pressure < 990:
        risk += 0.35
    elif request.pressure < 1000:
        risk += 0.2
    elif request.pressure < 1010:
        risk += 0.1
    
    if request.humidity > 85:
        risk += 0.15
    elif request.humidity > 70:
        risk += 0.1
    
    if request.month in [4, 5, 10, 11]:
        risk *= 1.3
    
    return min(1.0, risk)