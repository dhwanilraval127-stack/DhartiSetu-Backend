"""
Location Service Router
"""
from fastapi import APIRouter, HTTPException
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import logging

from app.models.schemas import LocationRequest, LocationResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize geocoder
geolocator = Nominatim(user_agent="dhartisetu_app")

@router.post("/reverse", response_model=LocationResponse)
async def reverse_geocode(request: LocationRequest):
    """Convert coordinates to address"""
    try:
        location = geolocator.reverse(
            f"{request.latitude}, {request.longitude}",
            language="en",
            timeout=10
        )
        
        if location is None:
            raise HTTPException(status_code=404, detail="Location not found")
        
        address = location.raw.get('address', {})
        
        # Extract location components
        city = (
            address.get('city') or 
            address.get('town') or 
            address.get('village') or 
            address.get('municipality') or
            address.get('suburb') or
            "Unknown"
        )
        
        district = (
            address.get('county') or 
            address.get('state_district') or 
            address.get('district') or
            city
        )
        
        state = address.get('state', 'Unknown')
        country = address.get('country', 'India')
        
        return LocationResponse(
            city=city,
            district=district,
            state=state,
            country=country
        )
        
    except GeocoderTimedOut:
        raise HTTPException(status_code=408, detail="Location service timeout")
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/states")
async def get_indian_states():
    """Get list of Indian states"""
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
        "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
        "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
        "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
        "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
    ]
    return {"states": states}

@router.get("/subdivisions")
async def get_subdivisions():
    """Get meteorological subdivisions of India"""
    subdivisions = [
        "Andaman & Nicobar Islands", "Arunachal Pradesh", "Assam & Meghalaya",
        "Bihar", "Chhattisgarh", "Coastal Andhra Pradesh", "Coastal Karnataka",
        "East Madhya Pradesh", "East Rajasthan", "East Uttar Pradesh",
        "Gangetic West Bengal", "Gujarat Region", "Haryana Delhi & Chandigarh",
        "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Kerala",
        "Konkan & Goa", "Lakshadweep", "Madhya Maharashtra", "Marathwada",
        "Naga Mani Mizo Tripura", "North Interior Karnataka", "Odisha",
        "Punjab", "Rayalaseema", "Saurashtra & Kutch", "South Interior Karnataka",
        "Sub Himalayan West Bengal & Sikkim", "Tamil Nadu", "Telangana",
        "Uttarakhand", "Vidarbha", "West Madhya Pradesh", "West Rajasthan",
        "West Uttar Pradesh"
    ]
    return {"subdivisions": subdivisions}