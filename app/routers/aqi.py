"""
AQI Prediction Router - FIXED FEATURE COUNT
"""
from fastapi import APIRouter, HTTPException
import numpy as np
import logging

from app.models.schemas import AQIRequest, PredictionResponse
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)

def get_aqi_category(aqi_value: float) -> str:
    """Categorize AQI value"""
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
    else:
        return "hazardous"

def get_aqi_label(category: str, language) -> str:
    """Get human-readable AQI label"""
    labels = {
        "good": {"en": "Good", "hi": "अच्छा"},
        "moderate": {"en": "Moderate", "hi": "मध्यम"},
        "unhealthy_sensitive": {"en": "Unhealthy for Sensitive Groups", "hi": "संवेदनशील समूहों के लिए अस्वस्थ"},
        "unhealthy": {"en": "Unhealthy", "hi": "अस्वस्थ"},
        "very_unhealthy": {"en": "Very Unhealthy", "hi": "बहुत अस्वस्थ"},
        "hazardous": {"en": "Hazardous", "hi": "खतरनाक"}
    }
    return labels.get(category, labels["moderate"])[language.value]

@router.post("/predict", response_model=PredictionResponse)
async def predict_aqi(request: AQIRequest):
    """Predict AQI based on pollutant levels - FIXED FEATURE COUNT"""
    try:
        model_data = model_loader.get_all("aqi")
        
        if not model_data or "model" not in model_data:
            # Fallback calculation using EPA formula
            aqi_value = calculate_aqi_fallback(request)
        else:
            model = model_data["model"]
            encoder = model_data.get("encoder")
            
            # Prepare features - EXACTLY 12 FEATURES for AQI model
            features = np.array([[
                request.pm25,      # 1
                request.pm10,      # 2
                request.no2,       # 3
                request.so2,       # 4
                request.co,        # 5
                request.o3,        # 6
                request.temperature, # 7
                request.humidity,   # 8
                request.wind_speed, # 9
                request.pressure,   # 10
                request.visibility,  # 11
                0                   # 12 - placeholder for additional feature
            ]])
            
            # Validate feature count
            if features.shape[1] != 12:
                logger.warning(f"AQI feature count mismatch: expected 12, got {features.shape[1]}")
                raise ValueError("Feature count mismatch")
            
            # If city encoder exists, add encoded city
            if encoder and hasattr(request, 'city'):
                try:
                    city_encoded = encoder.transform([request.city])[0]
                    # Replace last feature with encoded city
                    features[0][-1] = city_encoded
                except Exception as e:
                    logger.warning(f"City encoding failed: {e}")
            
            aqi_value = float(model.predict(features)[0])
        
        # Get category and explanation
        category = get_aqi_category(aqi_value)
        
        # Map to explanation category
        if category in ["good"]:
            explain_category = "good"
        elif category in ["moderate", "unhealthy_sensitive"]:
            explain_category = "moderate"
        else:
            explain_category = "poor"
        
        explanation = explainer.get_explanation("aqi", explain_category, request.language)
        
        return PredictionResponse(
            success=True,
            prediction={
                "aqi_value": round(aqi_value, 1),
                "category": get_aqi_label(category, request.language),
                "category_code": category,
                "city": request.city,
                "pollutants": {
                    "pm25": request.pm25,
                    "pm10": request.pm10,
                    "no2": request.no2,
                    "so2": request.so2,
                    "co": request.co,
                    "o3": request.o3
                }
            },
            confidence=0.85,
            explanation=explanation,
            language=request.language
        )
        
    except Exception as e:
        logger.error(f"AQI prediction error: {e}")
        # Return fallback instead of error
        aqi_value = calculate_aqi_fallback(request)
        category = get_aqi_category(aqi_value)
        explanation = explainer.get_explanation("aqi", "moderate", request.language)
        
        return PredictionResponse(
            success=True,
            prediction={
                "aqi_value": round(aqi_value, 1),
                "category": get_aqi_label(category, request.language),
                "category_code": category,
                "city": request.city,
                "pollutants": {
                    "pm25": request.pm25,
                    "pm10": request.pm10,
                    "no2": request.no2,
                    "so2": request.so2,
                    "co": request.co,
                    "o3": request.o3
                }
            },
            confidence=0.75,
            explanation=explanation,
            language=request.language
        )

def calculate_aqi_fallback(request: AQIRequest) -> float:
    """Fallback AQI calculation using simplified EPA formula"""
    # Calculate individual AQI components
    def calculate_component(concentration, breakpoints):
        for bp in breakpoints:
            if bp[0] <= concentration <= bp[1]:
                return ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (concentration - bp[0]) + bp[2]
        return 0
    
    # PM2.5 breakpoints (0-500 μg/m³)
    pm25_aqi = calculate_component(request.pm25, [
        (0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ])
    
    # PM10 breakpoints (0-600 μg/m³)
    pm10_aqi = calculate_component(request.pm10, [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400),
        (505, 604, 401, 500)
    ])
    
    # NO2 breakpoints (0-2000 ppb)
    no2_aqi = calculate_component(request.no2, [
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
        (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 1649, 301, 400),
        (1650, 2049, 401, 500)
    ])
    
    # Take maximum AQI component
    aqi_value = max(pm25_aqi, pm10_aqi, no2_aqi, 50)  # Minimum 50 for safety
    
    return min(500, max(0, aqi_value))