from pathlib import Path
import joblib
import numpy as np

# --------------------------------------------------
# Absolute path to model (SAFE on Windows)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
MODEL_PATH = BASE_DIR / "ml_models" / "water" / "water_rf.pkl"

# --------------------------------------------------
# Load model
# --------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Water model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Feature preparation (MATCH TRAINING ORDER)
# --------------------------------------------------
def prepare_features(request):
    """
    Converts API request into model-ready feature vector
    MUST match training column order
    """

    # IMPORTANT:
    # Replace this order ONLY if your training CSV order is different
    features = np.array([[
        request.temperature,
        request.humidity,
        request.area_hectares,

        # categorical encodings (example – adjust if needed)
        1 if request.crop.lower() == "rice" else 0,
        1 if request.crop.lower() == "wheat" else 0,
        1 if request.crop.lower() == "maize" else 0,

        1 if request.soil_type.lower() == "loamy" else 0,
        1 if request.soil_type.lower() == "sandy" else 0,
        1 if request.soil_type.lower() == "clay" else 0,

        1 if request.growth_stage.lower() == "seedling" else 0,
        1 if request.growth_stage.lower() == "vegetative" else 0,
        1 if request.growth_stage.lower() == "flowering" else 0,
        1 if request.growth_stage.lower() == "maturity" else 0
    ]])

    return features
