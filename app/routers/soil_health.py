"""
Soil Health Assessment Router - FINAL (PRODUCTION SAFE)
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import numpy as np
import logging
import io
from PIL import Image

from app.models.schemas import SoilHealthRequest, PredictionResponse, Language
from app.models.loader import model_loader
from app.services.explainer import explainer

router = APIRouter()
logger = logging.getLogger(__name__)

# ======================================================
# 1️⃣ SOIL HEALTH FROM LAB VALUES
# ======================================================
@router.post("/assess", response_model=PredictionResponse)
async def assess_soil_health(request: SoilHealthRequest):
    try:
        health_status, confidence = assess_soil_fallback(request)

        explanation = explainer.get_explanation(
            "soil_health",
            health_status,
            request.language
        )

        return PredictionResponse(
            success=True,
            prediction={
                "health_status": health_status,
                "health_label": get_health_label(health_status, request.language),
                "nutrient_analysis": analyze_nutrients(request, request.language),
                "input_values": request.model_dump(exclude={"language"})
            },
            confidence=round(confidence * 100, 2),
            explanation=explanation,
            language=request.language
        )

    except Exception:
        logger.error("Soil health assess error", exc_info=True)
        raise HTTPException(500, "Soil health assessment failed")


# ======================================================
# 2️⃣ SOIL HEALTH FROM IMAGE (CNN)
# ======================================================
@router.post("/assess-from-image", response_model=PredictionResponse)
async def assess_soil_health_from_image(
    file: UploadFile = File(...),
    language: Language = Form(Language.EN)
):
    try:
        # ---------- Image type validation ----------
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(400, "Invalid image type")

        soil_model = model_loader.get_model("soil_cnn", "model")
        if soil_model is None:
            raise HTTPException(503, "Soil CNN model not loaded")

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = soil_model.predict(img_array)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        soil_classes = ["black", "red", "alluvial", "sandy"]
        soil_type = soil_classes[idx]

        health_status, nutrient_analysis = infer_soil_health_from_type(
            soil_type,
            language
        )

        explanation = explainer.get_explanation(
            "soil_health",
            health_status,
            language
        )

        return PredictionResponse(
            success=True,
            prediction={
                "soil_type": soil_type,
                "soil_type_label": soil_type.title(),
                "health_status": health_status,
                "health_label": get_health_label(health_status, language),
                "nutrient_analysis": nutrient_analysis
            },
            confidence=round(confidence * 100, 2),
            explanation=explanation,
            language=language
        )

    except Exception:
        logger.error("Soil health image error", exc_info=True)
        raise HTTPException(500, "Soil health assessment failed")


# ======================================================
# 🔧 HELPERS
# ======================================================
def get_health_label(status: str, language: Language) -> str:
    lang = language.value if hasattr(language, "value") else language
    labels = {
        "good": {"en": "Good Health", "hi": "अच्छा स्वास्थ्य"},
        "moderate": {"en": "Moderate Health", "hi": "मध्यम स्वास्थ्य"},
        "poor": {"en": "Poor Health", "hi": "खराब स्वास्थ्य"}
    }
    return labels.get(status, labels["moderate"])[lang]


def infer_soil_health_from_type(soil_type: str, language: Language):
    lang = language.value if hasattr(language, "value") else language

    if soil_type == "black":
        return "good", {
            "nitrogen": label("High", "उच्च", lang),
            "phosphorus": label("Moderate", "मध्यम", lang),
            "potassium": label("High", "उच्च", lang),
            "ph": label("Optimal", "इष्टतम", lang)
        }

    if soil_type == "alluvial":
        return "good", {
            "nitrogen": label("Moderate", "मध्यम", lang),
            "phosphorus": label("Moderate", "मध्यम", lang),
            "potassium": label("Moderate", "मध्यम", lang),
            "ph": label("Optimal", "इष्टतम", lang)
        }

    if soil_type == "red":
        return "moderate", {
            "nitrogen": label("Low", "कम", lang),
            "phosphorus": label("Low", "कम", lang),
            "potassium": label("Moderate", "मध्यम", lang),
            "ph": label("Acidic", "अम्लीय", lang)
        }

    if soil_type == "sandy":
        return "poor", {
            "nitrogen": label("Very Low", "बहुत कम", lang),
            "phosphorus": label("Low", "कम", lang),
            "potassium": label("Low", "कम", lang),
            "ph": label("Alkaline", "क्षारीय", lang)
        }

    return "moderate", {}


def label(en: str, hi: str, lang: str):
    return {
        "status": en.lower(),
        "label": hi if lang == "hi" else en
    }


def assess_soil_fallback(request: SoilHealthRequest):
    score = 0

    if 250 <= request.nitrogen <= 500:
        score += 20
    if 25 <= request.phosphorus <= 50:
        score += 20
    if 200 <= request.potassium <= 300:
        score += 20
    if 6.0 <= request.ph <= 7.5:
        score += 20
    if request.organic_carbon >= 0.75:
        score += 20

    if score >= 70:
        return "good", 0.85
    elif score >= 40:
        return "moderate", 0.75
    return "poor", 0.80


def analyze_nutrients(request: SoilHealthRequest, language: Language):
    lang = language.value if hasattr(language, "value") else language

    def check(val, low, high):
        if val < low:
            return label("Low", "कम", lang)
        if val > high:
            return label("High", "अधिक", lang)
        return label("Optimal", "इष्टतम", lang)

    return {
        "nitrogen": check(request.nitrogen, 250, 500),
        "phosphorus": check(request.phosphorus, 25, 50),
        "potassium": check(request.potassium, 200, 300),
        "ph": check(request.ph, 6.0, 7.5),
        "organic_carbon": check(request.organic_carbon, 0.5, 2.0)
    }
