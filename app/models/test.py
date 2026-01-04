# test_model_loading.py
import tensorflow as tf
import tensorflow_hub as hub
import traceback

print("=== Testing Model Loading ===")

model_path = "E:/final final code/backend/ml_models/soil_cnn/soil_type_model.h5"

print(f"Testing model: {model_path}")

try:
    print("Attempt 1: Standard loading...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Standard loading successful!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
except ValueError as e:
    if "Unknown layer" in str(e):
        print("⚠ Custom layers detected, trying with custom objects...")
        try:
            custom_objects = {'KerasLayer': hub.KerasLayer}
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print("✅ Custom object loading successful!")
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
        except Exception as e2:
            print(f"❌ Custom loading also failed: {e2}")
            traceback.print_exc()
    else:
        print(f"❌ Standard loading failed with different error: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    traceback.print_exc()

print("=== Test Complete ===")