import tensorflow as tf
from tensorflow import keras

# Correct path (use raw string or forward slashes)
model_path = r"E:/final final code/backend/ml_models/soil_cnn/soil_type_model.h5"

# Load H5 model directly
model = keras.models.load_model(model_path, compile=False)

# Optional: print summary
model.summary()
print("✅ Soil CNN model loaded successfully!")
