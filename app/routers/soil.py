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
# Crop translations
# ==================================================
CROP_NAMES_HI = {
    "Rice": "चावल",
    "Wheat": "गेहूं",
    "Sugarcane": "गन्ना",
    "Maize": "मक्का",
    "Cotton": "कपास",
    "Soybean": "सोयाबीन",
    "Groundnut": "मूंगफली",
    "Millets": "बाजरा",
    "Barley": "जौ",
    "Lentils": "दालें",
    "Gram": "चना",
    "Tea": "चाय",
    "Coffee": "कॉफी",
    "Cashew": "काजू",
    "Rubber": "रबर",
    "Vegetables": "सब्ज़ियाँ",
    "Fruits": "फल",
    "Potato": "आलू",
    "Carrot": "गाजर",
    "Onion": "प्याज़",
    "Tomato": "टमाटर",
    "Peanut": "मूंगफली",
    "Chilli": "मिर्च",
    "Melons": "खरबूजे",
    "Watermelon": "तरबूज",
    "Oilseeds": "तिलहन",
    "Paddy": "धान",
    "Pulses": "दलहन",
    "Sorghum": "ज्वार",
    "Cucumber": "खीरा"
}

# ==================================================
# Soil → Crop knowledge base
# ==================================================
SOIL_CROP_RECOMMENDATIONS = {
    "Alluvial soil": {
        "primary": ["Rice", "Wheat"],
        "secondary": ["Sugarcane", "Maize", "Pulses"]
    },
    "Black Soil": {
        "primary": ["Cotton", "Soybean"],
        "secondary": ["Groundnut", "Sorghum"]
    },
    "Cinder Soil": {
        "primary": ["Millets"],
        "secondary": ["Barley"]
    },
    "Clayey soils": {
        "primary": ["Rice"],
        "secondary": ["Lentils", "Gram"]
    },
    "Laterite soil": {
        "primary": ["Tea", "Coffee"],
        "secondary": ["Cashew", "Rubber"]
    },
    "Loamy soil": {
        "primary": ["Vegetables", "Wheat"],
        "secondary": ["Fruits", "Pulses"]
    },
    "Peat Soil": {
        "primary": ["Potato"],
        "secondary": ["Carrot", "Onion"]
    },
    "Sandy loam": {
        "primary": ["Tomato", "Peanut"],
        "secondary": ["Chilli", "Melons"]
    },
    "Sandy soil": {
        "primary": ["Watermelon"],
        "secondary": ["Millets", "Cucumber"]
    },
    "Yellow Soil": {
        "primary": ["Paddy"],
        "secondary": ["Oilseeds", "Maize"]
    }
}

# ==================================================
# SOIL DETECTION ENDPOINT
# ==================================================
@router.post("/detect", response_model=ImageUploadResponse)
async def detect_soil_type(
    file: UploadFile = File(...),
    language: Language = Form(Language.EN)
):
    try:
        # ---------- Validate file ----------
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, "Invalid image type")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(400, "Empty image file")

        is_valid, msg = image_processor.validate_image(image_bytes)
        if not is_valid:
            raise HTTPException(400, msg)

        # ---------- Load model ----------
        model = model_loader.get_model("soil_cnn", "model")
        if model is None:
            raise HTTPException(503, "Soil model not loaded")

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
        primary_en = crop_data.get("primary", [])
        secondary_en = crop_data.get("secondary", [])

        primary_hi = [CROP_NAMES_HI.get(c, c) for c in primary_en]
        secondary_hi = [CROP_NAMES_HI.get(c, c) for c in secondary_en]

        # ---------- Explanation ----------
        explanation = {
            "why": (
                f"The soil was identified as {soil_en} using visual patterns "
                f"such as texture, color, and granularity learned by the AI model."
            ),
            "factors": [
                "Soil color",
                "Surface texture",
                "Granularity",
                "Light reflection"
            ],
            "prevention": [
                "Avoid over-irrigation",
                "Prevent soil erosion",
                "Maintain organic matter"
            ],
            "next_steps": [
                f"Primary crops: {', '.join(primary_en)}",
                f"Secondary crops: {', '.join(secondary_en)}",
                "Conduct soil nutrient testing before sowing"
            ]
        }

        # ---------- Response ----------
        return ImageUploadResponse(
            success=True,
            prediction={
                "soil_en": soil_en,
                "soil_hi": soil_hi,
                "selected": soil_hi if language == Language.HI else soil_en,
                "recommended_crops": {
                    "primary": {
                        "en": primary_en,
                        "hi": primary_hi
                    },
                    "secondary": {
                        "en": secondary_en,
                        "hi": secondary_hi
                    }
                }
            },
            confidence=round(confidence * 100, 2),
            all_predictions=[
                {
                    "label": SOIL_CLASSES[i],
                    "confidence": round(float(preds[i]) * 100, 2)
                }
                for i in range(len(SOIL_CLASSES))
            ],
            explanation=explanation,
            language=language
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Soil detection error", exc_info=True)
        raise HTTPException(500, str(e))
