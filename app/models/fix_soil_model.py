# fix_soil_model.py
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

print("Creating compatible soil model...")

# Create the exact same architecture as your original model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", trainable=False),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 soil classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model created successfully!")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Test the model
dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32) / 255.0
predictions = model.predict(dummy_input, verbose=0)
print(f"Test prediction: {predictions.shape}")

# Save the model
model.save("E:/final final code/backend/ml_models/soil_cnn/soil_type_compatible.h5")
print("✅ Compatible model saved as soil_type_compatible.h5")

# Test loading the saved model
print("Testing saved model...")
test_model = tf.keras.models.load_model(
    "E:/final final code/backend/ml_models/soil_cnn/soil_type_compatible.h5",
    custom_objects={'KerasLayer': hub.KerasLayer}
)
print("✅ Saved model loads successfully!")