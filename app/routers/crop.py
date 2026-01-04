from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import numpy as np
import logging

from app.models.schemas import CropRequest, PredictionResponse, Language
from app.models.loader import model_loader
from app.services.explainer import explainer
from app.services.image_processor import image_processor
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# -------------------------------
# Crop names (Hindi)
# -------------------------------
CROP_NAMES_HI = {
    "rice": "चावल",
    "wheat": "गेहूं",
    "maize": "मक्का",
    "chickpea": "चना",
    "cotton": "कपास",
    "sugarcane": "गन्ना",
    "soybean": "सोयाबीन",
    "groundnut": "मूंगफली",
    "potato": "आलू",
    "onion": "प्याज",
    "tomato": "टमाटर"
}

# -------------------------------
# Crop list API (REQUIRED)
# -------------------------------
@router.get("/list")
async def list_crops():
    return {
        "success": True,
        "crops": [
            {"en": k.capitalize(), "hi": v}
            for k, v in CROP_NAMES_HI.items()
        ]
    }

# -------------------------------
# Numeric crop recommendation
# -------------------------------
@router.post("/recommend", response_model=PredictionResponse)
async def recommend_crop(request: CropRequest):
    try:
        model_data = model_loader.get_all("crop")

        features = np.array([[ 
            request.nitrogen,
            request.phosphorus,
            request.potassium,
            request.temperature,
            request.humidity,
            request.ph,
            request.rainfall
        ]])

        crop_name = "Rice"
        confidence = 70.0
        recommendations = []

        if model_data:
            model = model_data.get("model")
            scaler = model_data.get("scaler")
            encoder = model_data.get("encoder")

            if scaler:
                features = scaler.transform(features)

            pred = model.predict(features)[0]
            crop_name = encoder.inverse_transform([pred])[0]

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                confidence = float(max(probs) * 100)

                top_idx = np.argsort(probs)[-3:][::-1]
                top_crops = encoder.inverse_transform(top_idx)

                recommendations = [
                    {
                        "crop": c,
                        "crop_hi": CROP_NAMES_HI.get(c.lower(), c),
                        "probability": round(float(probs[i]) * 100, 1)
                    }
                    for c, i in zip(top_crops, top_idx)
                ]

        explanation = explainer.get_explanation(
            "crop_recommendation", "default", request.language
        )

        return PredictionResponse(
            success=True,
            prediction={
                "recommended_crop": crop_name,
                "recommended_crop_hi": CROP_NAMES_HI.get(crop_name.lower(), crop_name),
                "all_recommendations": recommendations
            },
            confidence=round(confidence, 1),
            explanation=explanation,
            language=request.language
        )

    except Exception:
        logger.error("Crop recommendation failed", exc_info=True)
        raise HTTPException(500, "Crop recommendation failed")

# -------------------------------
# Soil image → crop recommendation
# -------------------------------
@router.post("/recommend-from-soil-image", response_model=PredictionResponse)
async def recommend_crop_from_soil_image(
    file: UploadFile = File(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    language: Language = Form(Language.EN)
):
    try:
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, "Invalid image type")

        image_bytes = await file.read()
        soil_model = model_loader.get_model("soil_cnn", "model")

        if soil_model is None:
            raise HTTPException(503, "Soil model not available")

        img = image_processor.load_and_preprocess(image_bytes, (224, 224))
        preds = soil_model.predict(img, verbose=0)[0]
        soil_type = settings.SOIL_CLASSES[int(np.argmax(preds))]

        params = settings.SOIL_TO_DEFAULT_PARAMS.get(soil_type)
        if not params:
            raise HTTPException(400, "Unsupported soil type")

        crop_request = CropRequest(
            nitrogen=params["N"],
            phosphorus=params["P"],
            potassium=params["K"],
            temperature=temperature,
            humidity=humidity,
            ph=params["pH"],
            rainfall=params["rainfall"],
            language=language
        )

        response = await recommend_crop(crop_request)
        response.prediction["soil_type"] = soil_type
        return response

    except Exception:
        logger.error("Soil image crop recommendation failed", exc_info=True)
        raise HTTPException(500, "Soil image processing failed")
