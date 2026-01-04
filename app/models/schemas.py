"""
Pydantic Schemas for API
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class Language(str, Enum):
    EN = "en"
    HI = "hi"

# Base Response
class BaseResponse(BaseModel):
    success: bool = True
    message: str = ""
    data: Optional[Dict[str, Any]] = None

class ExplanationResponse(BaseModel):
    why: str
    factors: List[str]
    prevention: List[str]
    next_steps: List[str]

class PredictionResponse(BaseModel):
    success: bool = True
    prediction: Any
    confidence: Optional[float] = None
    explanation: ExplanationResponse
    language: Language = Language.EN

# Location
class LocationRequest(BaseModel):
    latitude: float
    longitude: float

class LocationResponse(BaseModel):
    city: str
    district: str
    state: str
    country: str = "India"

# AQI
class AQIRequest(BaseModel):
    city: str
    pm25: float = Field(..., ge=0)
    pm10: float = Field(..., ge=0)
    no2: float = Field(..., ge=0)
    so2: float = Field(..., ge=0)
    co: float = Field(..., ge=0)
    o3: float = Field(..., ge=0)
    language: Language = Language.EN

class CO2Request(BaseModel):
    year: int = Field(..., ge=1900, le=2100)
    month: int = Field(..., ge=1, le=12)

    # ✅ REQUIRED ENVIRONMENTAL FEATURES
    temperature: float = Field(..., ge=-20, le=60, description="Temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Humidity in %")
    pressure: float = Field(..., ge=900, le=1100, description="Atmospheric pressure in hPa")
    wind_speed: float = Field(..., ge=0, le=200, description="Wind speed in km/h")

    language: Language = Language.EN

# Crop Recommendation
class CropRequest(BaseModel):
    nitrogen: float = Field(..., ge=0, le=200)
    phosphorus: float = Field(..., ge=0, le=200)
    potassium: float = Field(..., ge=0, le=200)
    temperature: float = Field(..., ge=-10, le=60)
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0)
    language: Language = Language.EN

# Flood
class FloodRequest(BaseModel):
    state: str
    district: str
    rainfall_mm: float = Field(..., ge=0)
    river_level: float = Field(..., ge=0)
    elevation: float
    flood_history: int = Field(..., ge=0, le=10)
    language: Language = Language.EN

# NDVI
class NDVIRequest(BaseModel):
    red_band: float = Field(..., ge=0, le=1)
    nir_band: float = Field(..., ge=0, le=1)
    temperature: float
    rainfall: float = Field(..., ge=0)
    language: Language = Language.EN

# Price
class PriceRequest(BaseModel):
    crop: str = Field(..., example="wheat")
    state: str = Field(..., example="Gujarat")

    # optional but router safely handles it
    district: Optional[str] = Field("", example="Bhavnagar")

    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2000, le=2100)

    # 🔥 REQUIRED (used in router & ML features)
    production: float = Field(..., ge=0, example=100000)
    demand_index: float = Field(..., ge=0, le=100, example=50)

    language: Language = Language.EN

# Profit
class ProfitRequest(BaseModel):
    crop: str
    area_hectares: float = Field(..., gt=0)
    cost_per_hectare: float = Field(..., ge=0)
    expected_yield: float = Field(..., ge=0)
    market_price: float = Field(..., gt=0)
    language: Language = Language.EN

# Rainfall
class RainfallRequest(BaseModel):
    subdivision: str
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=1900, le=2100)
    language: Language = Language.EN

# Soil Health
class SoilHealthRequest(BaseModel):
    nitrogen: float = Field(..., ge=0)
    phosphorus: float = Field(..., ge=0)
    potassium: float = Field(..., ge=0)
    ph: float = Field(..., ge=0, le=14)
    organic_carbon: float = Field(..., ge=0)
    ec: float = Field(..., ge=0)  # Electrical Conductivity
    language: Language = Language.EN

# Storm
class StormRequest(BaseModel):
    state: str
    month: int = Field(..., ge=1, le=12)
    wind_speed: float = Field(..., ge=0)
    pressure: float = Field(..., gt=0)
    humidity: float = Field(..., ge=0, le=100)
    language: Language = Language.EN

# Water Requirement
class WaterRequest(BaseModel):
    crop: str
    growth_stage: str
    temperature: float
    humidity: float = Field(..., ge=0, le=100)
    soil_type: str
    area_hectares: float = Field(..., gt=0)
    language: Language = Language.EN

# Yield
class YieldRequest(BaseModel):
    crop: str
    state: str
    district: str
    season: str
    area_hectares: float = Field(..., gt=0)
    language: Language = Language.EN

class ImageUploadResponse(BaseModel):
    success: bool = True
    prediction: Dict[str, Any]   # ✅ FIX
    confidence: float
    all_predictions: Optional[List[Dict[str, Any]]] = None
    explanation: ExplanationResponse
    language: Language = Language.EN
