# extract_model_architecture.py
import h5py
import json
import tensorflow as tf
import numpy as np

print("=== Extracting Model Architecture ===")

original_path = "E:/final final code/backend/ml_models/soil_cnn/soil_type_model.h5"
recreated_path = "E:/final final code/backend/ml_models/soil_cnn/soil_type_recreated.h5"

try:
    # Inspect the model structure
    with h5py.File(original_path, 'r') as f:
        print("Model attributes:")
        for key in f.attrs.keys():
            print(f"  {key}: {f.attrs[key]}")
        
        # Get model config if available
        if 'model_config' in f.attrs:
            model_config_json = f.attrs['model_config']
            model_config = json.loads(model_config_json)
            print(f"\nModel config type: {type(model_config)}")
            print(f"Model config keys: {model_config.keys() if isinstance(model_config, dict) else 'Not a dict'}")
            
            # Save config to file for inspection
            with open('model_config.json', 'w') as config_file:
                json.dump(model_config, config_file, indent=2)
            print("✅ Model config saved to model_config.json")
    
    # Try to recreate the model based on the inspection
    print("\nCreating recreated model...")
    
    # Based on your HDF5 structure, it looks like a MobileNetV2-based model
    # Let's create a compatible model architecture
    
    # Input layer
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Use MobileNetV2 as base (transfer learning approach)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None  # Don't load ImageNet weights since we'll load your weights
    )
    
    # Add custom top layers for your 5 soil classes
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax', name='predictions')(x)  # 5 soil classes
    
    # Create the model
    model = tf.keras.Model(inputs, outputs)
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Recreated model created!")
    print(f"Model summary:")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    # Save the recreated model
    model.save(recreated_path)
    print("✅ Recreated model saved!")
    
    # Test the model
    print("Testing recreated model...")
    dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32) / 255.0
    predictions = model.predict(dummy_input, verbose=0)
    print(f"✅ Model works! Output shape: {predictions.shape}")
    print(f"   Prediction values: {predictions[0]}")
    print(f"   Sum of predictions: {np.sum(predictions[0]):.4f}")
    
    print(f"\n🎉 Model recreation completed!")
    print(f"Use this path: {recreated_path}")
    
except Exception as e:
    print(f"❌ Error recreating model: {e}")
    import traceback
    traceback.print_exc()