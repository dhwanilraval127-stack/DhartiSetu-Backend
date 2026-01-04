from pathlib import Path
import joblib
import numpy as np

# --------------------------------------------------
# Absolute path to model (works locally & on Render)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
MODEL_PATH = BASE_DIR / "ml_models" / "water" / "water_rf.pkl"

# --------------------------------------------------
# Safe model loading (DO NOT CRASH SERVER)
# --------------------------------------------------
model = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[INFO] Water model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load water model: {e}")
else:
    print(f"[WARN] Water model not found at {MODEL_PATH}. API will run without it.")

# --------------------------------------------------
# Feature preparation (MUST match training order)
# --------------------------------------------------
def prepare_features(request):
    """
    Converts API request into model-ready feature vector
    MUST match training column order
    """

    features = np.array([[

        # numerical
        request.temperature,
        request.humidity,
        request.area_hectares,

        # crop (one-hot)
        1 if request.crop.lower() == "rice" else 0,
        1 if request.crop.lower() == "wheat" else 0,
        1 if request.crop.lower() == "maize" else 0,

        # soil type (one-hot)
        1 if request.soil_type.lower() == "loamy" else 0,
        1 if request.soil_type.lower() == "sandy" else 0,
        1 if request.soil_type.lower() == "clay" else 0,

        # growth stage (one-hot)
        1 if request.growth_stage.lower() == "seedling" else 0,
        1 if request.growth_stage.lower() == "vegetative" else 0,
        1 if request.growth_stage.lower() == "flowering" else 0,
        1 if request.growth_stage.lower() == "maturity" else 0
    ]])

    return features

# --------------------------------------------------
# Prediction wrapper (SAFE)
# --------------------------------------------------
def predict_water_requirement(features):
    """
    Returns prediction or clean error if model not loaded
    """
    if model is None:
        return {
            "error": "Water model is not available on this server"
        }

    prediction = model.predict(features)
    return prediction.tolist()
