# diagnose_soil_model.py
import os
from pathlib import Path

print("=== Soil Model Diagnosis ===")

# Check your exact model path
model_path = "E:/final final code/backend/ml_models/soil_cnn/soil_type_model.h5"
print(f"Expected model path: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    size = os.path.getsize(model_path)
    print(f"File size: {size} bytes ({size/1024/1024:.2f} MB)")
    
    # List directory contents
    dir_path = Path(model_path).parent
    print(f"\nFiles in {dir_path}:")
    for file in dir_path.iterdir():
        print(f"  - {file.name}")
else:
    print("❌ MODEL FILE NOT FOUND!")
    # Check parent directories
    parent = Path(model_path).parent
    if parent.exists():
        print(f"Files in parent directory:")
        for file in parent.iterdir():
            print(f"  - {file.name}")