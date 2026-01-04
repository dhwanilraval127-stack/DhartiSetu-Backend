"""
Profit Calculator Router - FIXED CALCULATIONS (HARDENED)
"""
from fastapi import APIRouter
import numpy as np
import logging

from app.models.schemas import ProfitRequest, PredictionResponse
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/calculate", response_model=PredictionResponse)
async def calculate_profit(request: ProfitRequest):
    """Calculate expected profit from crop cultivation"""
    try:
        # -------------------------------------------------
        # 🌐 LANGUAGE SAFE HANDLING
        # -------------------------------------------------
        lang = request.language.value if hasattr(request.language, "value") else "en"

        # -------------------------------------------------
        # BASIC CALCULATIONS (DIVISION SAFE)
        # -------------------------------------------------
        area = max(request.area_hectares, 0)
        cost_per_hectare = max(request.cost_per_hectare, 0)
        expected_yield = max(request.expected_yield, 0)
        market_price = max(request.market_price, 0)

        total_cost = cost_per_hectare * area
        total_revenue = expected_yield * market_price * area
        basic_profit = total_revenue - total_cost

        # -------------------------------------------------
        # ML MODEL (OPTIONAL)
        # -------------------------------------------------
        model_data = model_loader.get_all("profit")

        if model_data and "model" in model_data:
            try:
                model = model_data["model"]
                encoders = model_data.get("encoders", {})

                crop_encoded = 0
                if "crop" in encoders:
                    try:
                        crop_encoded = encoders["crop"].transform([request.crop])[0]
                    except Exception as e:
                        logger.warning(f"Profit crop encoding failed: {e}")

                features = np.array([[
                    crop_encoded,
                    area,
                    cost_per_hectare,
                    expected_yield,
                    market_price,
                    0  # placeholder
                ]])

                predicted_profit = float(model.predict(features)[0])

            except Exception as e:
                logger.warning(f"Profit model failed, using fallback: {e}")
                predicted_profit = basic_profit
        else:
            predicted_profit = basic_profit

        # -------------------------------------------------
        # ROI (ZERO SAFE)
        # -------------------------------------------------
        roi = (predicted_profit / total_cost * 100) if total_cost > 0 else 0.0

        # -------------------------------------------------
        # PROFITABILITY CATEGORY
        # -------------------------------------------------
        if roi > 50:
            profitability = "high"
            profitability_label = (
                "Highly Profitable" if lang == "en" else "अत्यधिक लाभदायक"
            )
        elif roi > 20:
            profitability = "moderate"
            profitability_label = (
                "Moderately Profitable" if lang == "en" else "मध्यम लाभदायक"
            )
        elif roi > 0:
            profitability = "low"
            profitability_label = (
                "Low Profit" if lang == "en" else "कम लाभ"
            )
        else:
            profitability = "loss"
            profitability_label = (
                "Loss Expected" if lang == "en" else "हानि की संभावना"
            )

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
                    "area": area,
                    "cost_per_hectare": cost_per_hectare,
                    "expected_yield_per_hectare": expected_yield,
                    "market_price_per_unit": market_price,
                    "total_area_cost": round(total_cost, 2),
                    "total_expected_revenue": round(total_revenue, 2),
                },
                "currency": "INR",
            },
            confidence=0.75,
            explanation=explanation,
            language=request.language,
        )

    except Exception as e:
        logger.error(f"Profit calculation error: {e}")

        total_cost = request.cost_per_hectare * request.area_hectares
        total_revenue = request.expected_yield * request.market_price * request.area_hectares
        basic_profit = total_revenue - total_cost
        roi = (basic_profit / total_cost * 100) if total_cost > 0 else 0.0

        return PredictionResponse(
            success=True,
            prediction={
                "expected_profit": round(basic_profit, 2),
                "total_cost": round(total_cost, 2),
                "total_revenue": round(total_revenue, 2),
                "roi_percentage": round(roi, 2),
                "profitability": "moderate",
                "profitability_label": "Moderate Profit" if lang == "en" else "मध्यम लाभ",
                "currency": "INR",
            },
            confidence=0.65,
            explanation=explainer.get_explanation("profit", "default", request.language),
            language=request.language,
        )
