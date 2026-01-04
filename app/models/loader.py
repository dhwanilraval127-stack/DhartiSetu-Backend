# app/models/loader.py

import os
import joblib
import logging
from typing import Dict, Any, Optional

import tensorflow as tf
from huggingface_hub import hf_hub_download

from app.config import settings

logger = logging.getLogger(__name__)

# 🔗 Hugging Face repo where all ML models are stored
HF_REPO_ID = "crimson1232/dhartisetu-ml-models"

# ✅ Render / Railway / Koyeb SAFE cache directory
HF_CACHE_DIR = os.getenv("HF_HOME", "/tmp/hf_models")


class ModelLoader:
    """
    Centralized ML model loader (Singleton)
    Loads models from Hugging Face and caches them locally.
    """

    _instance = None
    _models: Dict[str, Dict[str, Any]] = {}
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # --------------------------------------------------
    # INTERNAL LOADERS
    # --------------------------------------------------
    def _load_pickle(self, hf_path: str):
        file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_path,
            cache_dir=HF_CACHE_DIR
        )
        return joblib.load(file_path)

    def _load_keras(self, hf_path: str):
        file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_path,
            cache_dir=HF_CACHE_DIR
        )
        return tf.keras.models.load_model(file_path, compile=False)

    # --------------------------------------------------
    # LOAD ALL MODELS (ONCE AT STARTUP)
    # --------------------------------------------------
    def load_all_models(self) -> None:
        if self._loaded:
            logger.info("⚠️ Models already loaded, skipping reload")
            return

        logger.info("🚀 Loading ML models from Hugging Face...")
        logger.info(f"📁 HF cache directory: {HF_CACHE_DIR}")

        for model_name, components in settings.MODEL_PATHS.items():
            self._models[model_name] = {}

            for component, hf_path in components.items():
                try:
                    if hf_path.endswith((".h5", ".keras")):
                        logger.info(f"📦 Loading Keras model: {hf_path}")
                        self._models[model_name][component] = self._load_keras(hf_path)

                    elif hf_path.endswith(".pkl"):
                        logger.info(f"📦 Loading Pickle model: {hf_path}")
                        self._models[model_name][component] = self._load_pickle(hf_path)

                    else:
                        logger.warning(f"⚠️ Unsupported model type: {hf_path}")
                        continue

                    logger.info(f"✅ Loaded {model_name}/{component}")

                except Exception as e:
                    logger.error(
                        f"❌ Failed loading {model_name}/{component}: {e}",
                        exc_info=True
                    )

            if not self._models[model_name]:
                logger.warning(
                    f"⚠️ No components loaded for model group '{model_name}'. "
                    f"Fallback logic will be used."
                )

        if not self._models:
            raise RuntimeError("❌ No ML models were loaded at all. Startup aborted.")

        self._loaded = True
        logger.info("🎉 Model loading process completed")

    # --------------------------------------------------
    # GETTERS (SAFE)
    # --------------------------------------------------
    def get_model(self, model_name: str, component: str = "model") -> Optional[Any]:
        return self._models.get(model_name, {}).get(component)

    def get_all(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self._models.get(model_name)


# ✅ Singleton instance
model_loader = ModelLoader()
