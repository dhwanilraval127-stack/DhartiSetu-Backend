"""
Soil Type Detection Router - FINAL (RENDER SAFE)
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import numpy as np
import logging

from app.models.schemas import Language, ImageUploadResponse
from app.models.loader import model_loader
from app.services.image_processor import image_processor
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# ==================================================
# MUST MATCH MODEL TRAINING ORDER
# ==================================================
SOIL_CLASSES = [
    "Alluvial soil",
    "Black Soil",
    "Cinder Soil",
    "Clayey soils",
    "Laterite soil",
    "Loamy soil",
    "Peat Soil",
    "Sandy loam",
    "Sandy soil",
    "Yellow Soil"
]

SOIL_NAMES_HI = {
    "Alluvial soil": "जलोढ़ मिट्टी",
    "Black Soil": "काली मिट्टी",
    "Cinder Soil": "सिंडर मिट्टी",
    "Clayey soils": "चिकनी मिट्टी",
    "Laterite soil": "लैटेराइट मिट्टी",
    "Loamy soil": "दोमट मिट्टी",
    "Peat Soil": "पीट मिट्टी",
    "Sandy loam": "रेतीली दोमट मिट्टी",
    "Sandy soil": "रेतीली मिट्टी",
    "Yellow Soil": "पीली मिट्टी"
}

# ==================================================
# SOIL → CROP KNOWLEDGE
# ==================================================
SOIL_CROP_RECOMMENDATIONS = {
    "Alluvial soil": {"primary": ["Rice", "Wheat"], "secondary": ["Sugarcane", "Maize"]},
    "Black Soil": {"primary": ["Cotton", "Soybean"], "secondary": ["Groundnut"]},
    "Sandy soil": {"primary": ["Watermelon"], "secondary": ["Millets"]},
}

# ==================================================
# DETECTION ENDPOINT
# ==================================================
@router.post("/detect", response_model=ImageUploadResponse)
async def detect_soil_type(
    file: UploadFile = File(...),
    language: Language = Form(Language.EN)
):
    try:
        lang = language.value if hasattr(language, "value") else language

        # ---------- Validate image ----------
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, "Invalid image type")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(400, "Empty image file")

        is_valid, msg = image_processor.validate_image(image_bytes)
        if not is_valid:
            raise HTTPException(400, msg)

        # ---------- Load model (render-safe) ----------
        model = model_loader.get_model("soil_cnn", "model")
        if not model:
            return ImageUploadResponse(
                success=False,
                prediction={},
                confidence=0.0,
                explanation={"error": "Soil model unavailable"},
                language=language
            )

        # ---------- Preprocess ----------
        img = image_processor.load_and_preprocess(
            image_bytes,
            target_size=(224, 224)
        )

        # ---------- Predict ----------
        preds = model.predict(img, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        soil_en = SOIL_CLASSES[idx]
        soil_hi = SOIL_NAMES_HI.get(soil_en, soil_en)

        crop_data = SOIL_CROP_RECOMMENDATIONS.get(soil_en, {})
        primary = crop_data.get("primary", [])
        secondary = crop_data.get("secondary", [])

        return ImageUploadResponse(
            success=True,
            prediction={
                "soil_en": soil_en,
                "soil_hi": soil_hi,
                "selected": soil_hi if lang == "hi" else soil_en,
                "recommended_crops": {
                    "primary": primary,
                    "secondary": secondary
                }
            },
            confidence=round(confidence * 100, 2),
            explanation={
                "why": f"Soil identified as {soil_en} using CNN visual patterns."
            },
            language=language
        )

    except HTTPException:
        raise
    except Exception:
        logger.error("Soil detection error", exc_info=True)
        raise HTTPException(500, "Soil detection failed")
