from fastapi import APIRouter, HTTPException
import logging

from app.models.schemas import WaterRequest, PredictionResponse
from app.services.water_model import (
    prepare_features,
    predict_water_requirement
)
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/calculate", response_model=PredictionResponse)
async def calculate_water_requirement(request: WaterRequest):
    try:
        # Prepare ML features
        X = prepare_features(request)

        # SAFE prediction call
        result = predict_water_requirement(X)

        # Handle missing model case
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(
                status_code=503,
                detail="Water ML model is not available on the server"
            )

        prediction = float(result[0])

        if prediction <= 0:
            raise ValueError("Invalid ML prediction")

        weekly = prediction * 7
        monthly = prediction * 30
        total_liters = prediction * request.area_hectares * 10000

        explanation = explainer.get_explanation(
            "water", "ml", request.language
        )

        return PredictionResponse(
            success=True,
            prediction={
                "daily_water_mm": round(prediction, 2),
                "weekly_water_mm": round(weekly, 2),
                "monthly_water_mm": round(monthly, 2),
                "total_daily_liters": round(total_liters, 0),
                "crop": request.crop,
                "growth_stage": request.growth_stage,
                "soil_type": request.soil_type,
                "area_hectares": request.area_hectares,
                "irrigation_frequency": "ML Based"
            },
            confidence=0.95,
            explanation=explanation,
            language=request.language
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"ML prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="ML prediction failed"
        )
