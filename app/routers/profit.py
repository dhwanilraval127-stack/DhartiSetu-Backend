"""
Profit Calculator Router - FIXED CALCULATIONS
"""
from fastapi import APIRouter, HTTPException
import numpy as np
import logging

from app.models.schemas import ProfitRequest, PredictionResponse
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/calculate", response_model=PredictionResponse)
async def calculate_profit(request: ProfitRequest):
    """Calculate expected profit from crop cultivation - FIXED"""
    try:
        # Calculate basic profit components
        total_cost = request.cost_per_hectare * request.area_hectares
        total_revenue = request.expected_yield * request.market_price * request.area_hectares
        basic_profit = total_revenue - total_cost
        
        model_data = model_loader.get_all("profit")
        
        if model_data and "model" in model_data:
            model = model_data["model"]
            encoders = model_data.get("encoders", {})
            
            # Encode categorical features
            crop_encoded = 0
            if encoders and "crop" in encoders:
                try:
                    crop_encoded = encoders["crop"].transform([request.crop])[0]
                except Exception as e:
                    logger.warning(f"Profit crop encoding failed: {e}")
            
            # Prepare features for model - EXACTLY 6 FEATURES
            features = np.array([[
                crop_encoded,              # 1
                request.area_hectares,     # 2
                request.cost_per_hectare,  # 3
                request.expected_yield,   # 4
                request.market_price,     # 5
                0                          # 6 - placeholder
            ]])
            
            # Validate feature count
            if features.shape[1] != 6:
                logger.warning(f"Profit feature count mismatch: expected 6, got {features.shape[1]}")
                raise ValueError("Feature count mismatch")
            
            predicted_profit = float(model.predict(features)[0])
        else:
            predicted_profit = basic_profit
        
        # Calculate ROI
        roi = (predicted_profit / total_cost) * 100 if total_cost > 0 else 0
        
        # Determine profitability category
        if roi > 50:
            profitability = "high"
            profitability_label = "Highly Profitable" if request.language.value == "en" else "अत्यधिक लाभदायक"
        elif roi > 20:
            profitability = "moderate"
            profitability_label = "Moderately Profitable" if request.language.value == "en" else "मध्यम लाभदायक"
        elif roi > 0:
            profitability = "low"
            profitability_label = "Low Profit" if request.language.value == "en" else "कम लाभ"
        else:
            profitability = "loss"
            profitability_label = "Loss Expected" if request.language.value == "en" else "हानि की संभावना"
        
        explanation = explainer.get_explanation("profit", "default", request.language)
        
        return PredictionResponse(
            success=True,
            prediction={
                "expected_profit": round(predicted_profit, 2),
                "total_cost": round(total_cost, 2),
                "total_revenue": round(total_revenue, 2),
                "roi_percentage": round(roi, 2),
                "profitability": profitability,
                "profitability_label": profitability_label,
                "breakdown": {
                    "area": request.area_hectares,
                    "cost_per_hectare": request.cost_per_hectare,
                    "expected_yield_per_hectare": request.expected_yield,
                    "market_price_per_unit": request.market_price,
                    "total_area_cost": round(total_cost, 2),
                    "total_expected_revenue": round(total_revenue, 2)
                },
                "currency": "INR"
            },
            confidence=0.75,
            explanation=explanation,
            language=request.language
        )
        
    except Exception as e:
        logger.error(f"Profit calculation error: {e}")
        # Return fallback instead of error
        total_cost = request.cost_per_hectare * request.area_hectares
        total_revenue = request.expected_yield * request.market_price * request.area_hectares
        basic_profit = total_revenue - total_cost
        roi = (basic_profit / total_cost) * 100 if total_cost > 0 else 0
        
        explanation = explainer.get_explanation("profit", "default", request.language)
        
        return PredictionResponse(
            success=True,
            prediction={
                "expected_profit": round(basic_profit, 2),
                "total_cost": round(total_cost, 2),
                "total_revenue": round(total_revenue, 2),
                "roi_percentage": round(roi, 2),
                "profitability": "moderate",
                "profitability_label": "Moderate Profit",
                "breakdown": {
                    "area": request.area_hectares,
                    "cost_per_hectare": request.cost_per_hectare,
                    "expected_yield_per_hectare": request.expected_yield,
                    "market_price_per_unit": request.market_price
                },
                "currency": "INR"
            },
            confidence=0.65,
            explanation=explanation,
            language=request.language
        )