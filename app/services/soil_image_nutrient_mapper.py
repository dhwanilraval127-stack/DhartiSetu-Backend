import random

SOIL_NUTRIENT_RANGES = {
    "Black Soil": {
        "nitrogen": (300, 550),
        "phosphorus": (25, 45),
        "potassium": (250, 400),
        "ph": (6.5, 7.8),
        "organic_carbon": (0.7, 1.2),
        "ec": (0.3, 0.8)
    },
    "Red Soil": {
        "nitrogen": (150, 300),
        "phosphorus": (10, 25),
        "potassium": (150, 250),
        "ph": (5.5, 6.5),
        "organic_carbon": (0.3, 0.7),
        "ec": (0.2, 0.6)
    },
    "Sandy Soil": {
        "nitrogen": (80, 180),
        "phosphorus": (5, 15),
        "potassium": (80, 150),
        "ph": (5.0, 6.0),
        "organic_carbon": (0.1, 0.4),
        "ec": (0.1, 0.4)
    },
    "Alluvial Soil": {
        "nitrogen": (200, 400),
        "phosphorus": (20, 40),
        "potassium": (200, 350),
        "ph": (6.0, 7.5),
        "organic_carbon": (0.5, 1.0),
        "ec": (0.3, 0.7)
    }
}

def estimate_nutrients_from_soil_type(soil_type: str) -> dict:
    if soil_type not in SOIL_NUTRIENT_RANGES:
        raise ValueError("Unknown soil type")

    ranges = SOIL_NUTRIENT_RANGES[soil_type]

    return {
        "nitrogen": random.randint(*ranges["nitrogen"]),
        "phosphorus": random.randint(*ranges["phosphorus"]),
        "potassium": random.randint(*ranges["potassium"]),
        "ph": round(random.uniform(*ranges["ph"]), 1),
        "organic_carbon": round(random.uniform(*ranges["organic_carbon"]), 2),
        "ec": round(random.uniform(*ranges["ec"]), 2)
    }
