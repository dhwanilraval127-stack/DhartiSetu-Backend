"""
Pytest Configuration and Fixtures
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    """Create a test client"""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_crop_data():
    """Sample crop recommendation data"""
    return {
        "nitrogen": 90,
        "phosphorus": 42,
        "potassium": 43,
        "temperature": 25,
        "humidity": 80,
        "ph": 6.5,
        "rainfall": 200,
        "language": "en"
    }

@pytest.fixture
def sample_flood_data():
    """Sample flood prediction data"""
    return {
        "state": "Bihar",
        "district": "Patna",
        "rainfall_mm": 200,
        "river_level": 8,
        "elevation": 50,
        "flood_history": 3,
        "language": "en"
    }

@pytest.fixture
def sample_soil_health_data():
    """Sample soil health data"""
    return {
        "nitrogen": 280,
        "phosphorus": 35,
        "potassium": 250,
        "ph": 6.5,
        "organic_carbon": 0.8,
        "ec": 0.5,
        "language": "en"
    }

@pytest.fixture
def sample_yield_data():
    """Sample yield prediction data"""
    return {
        "crop": "Rice",
        "state": "Punjab",
        "district": "Ludhiana",
        "season": "Kharif",
        "area_hectares": 5,
        "language": "en"
    }