"""
API Endpoint Tests
"""
import pytest
from fastapi.testclient import TestClient

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "DhartiSetu"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

class TestLocationEndpoints:
    """Test location endpoints"""
    
    def test_get_states(self, client):
        """Test get states endpoint"""
        response = client.get("/api/v1/location/states")
        assert response.status_code == 200
        data = response.json()
        assert "states" in data
        assert len(data["states"]) > 0
        assert "Maharashtra" in data["states"]
    
    def test_get_subdivisions(self, client):
        """Test get subdivisions endpoint"""
        response = client.get("/api/v1/location/subdivisions")
        assert response.status_code == 200
        data = response.json()
        assert "subdivisions" in data

class TestCropEndpoints:
    """Test crop recommendation endpoints"""
    
    def test_get_crop_list(self, client):
        """Test get crop list endpoint"""
        response = client.get("/api/v1/crop/list")
        assert response.status_code == 200
        data = response.json()
        assert "crops" in data
        assert len(data["crops"]) > 0
    
    def test_crop_recommendation(self, client, sample_crop_data):
        """Test crop recommendation endpoint"""
        response = client.post("/api/v1/crop/recommend", json=sample_crop_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "prediction" in data
        assert "explanation" in data

class TestFloodEndpoints:
    """Test flood prediction endpoints"""
    
    def test_flood_prediction(self, client, sample_flood_data):
        """Test flood prediction endpoint"""
        response = client.post("/api/v1/flood/predict", json=sample_flood_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "prediction" in data
        assert "flood_probability" in data["prediction"]
        assert "risk_level" in data["prediction"]

class TestSoilHealthEndpoints:
    """Test soil health endpoints"""
    
    def test_soil_health_assessment(self, client, sample_soil_health_data):
        """Test soil health assessment endpoint"""
        response = client.post("/api/v1/soil-health/assess", json=sample_soil_health_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "prediction" in data
        assert "health_status" in data["prediction"]

class TestYieldEndpoints:
    """Test yield prediction endpoints"""
    
    def test_get_seasons(self, client):
        """Test get seasons endpoint"""
        response = client.get("/api/v1/yield/seasons")
        assert response.status_code == 200
        data = response.json()
        assert "seasons" in data
        assert len(data["seasons"]) > 0
    
    def test_yield_prediction(self, client, sample_yield_data):
        """Test yield prediction endpoint"""
        response = client.post("/api/v1/yield/predict", json=sample_yield_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "prediction" in data

class TestWaterEndpoints:
    """Test water requirement endpoints"""
    
    def test_get_growth_stages(self, client):
        """Test get growth stages endpoint"""
        response = client.get("/api/v1/water/growth-stages")
        assert response.status_code == 200
        data = response.json()
        assert "stages" in data