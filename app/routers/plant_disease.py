"""
Plant Disease Detection Router
FINAL – ZERO-ERROR VERSION
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import logging
import torch
import io
import numpy as np
from PIL import Image

from app.models.schemas import ImageUploadResponse, Language
from app.models.loader import model_loader
from app.services.explainer import explainer
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

def is_healthy_label(raw_label: str) -> bool:
    return "healthy" in raw_label.lower()

def get_health_condition(is_healthy: bool, confidence: float, language: Language):
    if is_healthy:
        return {
            "title": "Plant is Healthy" if language == Language.EN else "पौधा स्वस्थ है",
            "status": "Healthy",
            "status_hi": "स्वस्थ",
            "severity": "none"
        }

    if confidence < 90:
        return {
            "title": "Early Stage Disease Detected" if language == Language.EN else "बीमारी की शुरुआती अवस्था",
            "status": "Early Stage",
            "status_hi": "शुरुआती अवस्था",
            "severity": "low"
        }

    return {
        "title": "Disease Detected" if language == Language.EN else "बीमारी पाई गई",
        "status": "Diseased",
        "status_hi": "बीमार",
        "severity": "high"
    }

@router.post("/detect", response_model=ImageUploadResponse)
async def detect_plant_disease(
    file: UploadFile = File(...),
    language: Language = Form(Language.EN)
):
    try:
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail="Invalid image type")

        image_bytes = await file.read()

        model = model_loader.get_model("plant_disease", "model")
        processor = model_loader.get_model("plant_disease", "processor")

        if model is None or processor is None:
            raise HTTPException(status_code=503, detail="Plant disease model not loaded")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        confidence = float(probs[idx]) * 100

        raw_label = model.config.id2label[idx]
        is_healthy = is_healthy_label(raw_label)

        health_info = get_health_condition(is_healthy, confidence, language)

        # 🔥 THIS IS THE KEY FIX
        explanation_data = explainer.get_explanation(
            "plant_disease",
            "healthy" if is_healthy else "disease",
            language
        )

        return ImageUploadResponse(
            success=True,
            prediction={
                "title": health_info["title"],
                "status": health_info["status"],
                "status_hi": health_info["status_hi"],
                "severity": health_info["severity"],
                "is_healthy": is_healthy
            },
            confidence=round(confidence, 2),
            explanation=explanation_data.model_dump(),  # ✅ SAFE
            language=language
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Plant disease detection error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
