"""
Helper utilities for DhartiSetu API
"""
import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    if denominator == 0:
        return default
    return numerator / denominator

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max"""
    return max(min_val, min(value, max_val))

def round_to_decimals(value: float, decimals: int = 2) -> float:
    """Round to specified decimal places"""
    return round(value, decimals)

def load_json_file(file_path: Path) -> Optional[Dict]:
    """Load JSON file safely"""
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
    return None

def save_json_file(data: Dict, file_path: Path) -> bool:
    """Save data to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False

def get_indian_states() -> List[str]:
    """Get list of Indian states and union territories"""
    return [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
        "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
        "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
        "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
        "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
    ]

def get_crop_seasons() -> List[Dict[str, str]]:
    """Get crop seasons in India"""
    return [
        {"value": "Kharif", "en": "Kharif", "hi": "खरीफ", "months": "June - October"},
        {"value": "Rabi", "en": "Rabi", "hi": "रबी", "months": "October - March"},
        {"value": "Zaid", "en": "Zaid", "hi": "जायद", "months": "March - June"},
        {"value": "Whole Year", "en": "Whole Year", "hi": "पूर्ण वर्ष", "months": "All Year"}
    ]

def get_soil_types() -> List[Dict[str, str]]:
    """Get soil types"""
    return [
        {"en": "Alluvial", "hi": "जलोढ़"},
        {"en": "Black", "hi": "काली"},
        {"en": "Clay", "hi": "चिकनी"},
        {"en": "Laterite", "hi": "लैटेराइट"},
        {"en": "Loamy", "hi": "दोमट"},
        {"en": "Red", "hi": "लाल"},
        {"en": "Sandy", "hi": "रेतीली"}
    ]

def get_growth_stages() -> List[Dict[str, str]]:
    """Get crop growth stages"""
    return [
        {"value": "seedling", "en": "Seedling", "hi": "अंकुर"},
        {"value": "vegetative", "en": "Vegetative", "hi": "वानस्पतिक"},
        {"value": "flowering", "en": "Flowering", "hi": "फूल"},
        {"value": "maturity", "en": "Maturity", "hi": "परिपक्वता"}
    ]

def categorize_aqi(aqi_value: float) -> Dict[str, Any]:
    """Categorize AQI value"""
    if aqi_value <= 50:
        return {"category": "good", "en": "Good", "hi": "अच्छा", "color": "green"}
    elif aqi_value <= 100:
        return {"category": "moderate", "en": "Moderate", "hi": "मध्यम", "color": "yellow"}
    elif aqi_value <= 150:
        return {"category": "unhealthy_sensitive", "en": "Unhealthy for Sensitive", "hi": "संवेदनशील के लिए अस्वस्थ", "color": "orange"}
    elif aqi_value <= 200:
        return {"category": "unhealthy", "en": "Unhealthy", "hi": "अस्वस्थ", "color": "red"}
    elif aqi_value <= 300:
        return {"category": "very_unhealthy", "en": "Very Unhealthy", "hi": "बहुत अस्वस्थ", "color": "purple"}
    else:
        return {"category": "hazardous", "en": "Hazardous", "hi": "खतरनाक", "color": "maroon"}

def categorize_risk(probability: float) -> Dict[str, Any]:
    """Categorize risk level based on probability"""
    if probability >= 0.7:
        return {"level": "high", "en": "High Risk", "hi": "उच्च जोखिम", "color": "red"}
    elif probability >= 0.4:
        return {"level": "moderate", "en": "Moderate Risk", "hi": "मध्यम जोखिम", "color": "orange"}
    else:
        return {"level": "low", "en": "Low Risk", "hi": "कम जोखिम", "color": "green"}

def format_currency(amount: float, currency: str = "INR") -> str:
    """Format currency amount"""
    if currency == "INR":
        return f"₹{amount:,.2f}"
    return f"{amount:,.2f} {currency}"

def validate_coordinates(latitude: float, longitude: float) -> bool:
    """Validate geographic coordinates"""
    return -90 <= latitude <= 90 and -180 <= longitude <= 180

def is_within_india(latitude: float, longitude: float) -> bool:
    """Check if coordinates are within India's approximate bounds"""
    return 6.5 <= latitude <= 37.5 and 68.0 <= longitude <= 97.5