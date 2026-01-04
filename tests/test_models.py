"""
Model Loading and Prediction Tests
"""
import pytest
import numpy as np
from app.models.loader import model_loader
from app.services.explainer import explainer
from app.models.schemas import Language

class TestModelLoader:
    """Test model loading functionality"""
    
    def test_model_loader_singleton(self):
        """Test that model loader is a singleton"""
        loader1 = model_loader
        loader2 = model_loader
        assert loader1 is loader2
    
    def test_load_models(self):
        """Test loading models"""
        model_loader.load_all_models()
        # Models should be loaded (or attempted)
        assert model_loader._loaded == True

class TestExplainer:
    """Test explainer functionality"""
    
    def test_get_explanation_english(self):
        """Test getting explanation in English"""
        explanation = explainer.get_explanation("crop_recommendation", "default", Language.EN)
        assert explanation.why is not None
        assert len(explanation.factors) > 0
        assert len(explanation.prevention) > 0
        assert len(explanation.next_steps) > 0
    
    def test_get_explanation_hindi(self):
        """Test getting explanation in Hindi"""
        explanation = explainer.get_explanation("crop_recommendation", "default", Language.HI)
        assert explanation.why is not None
        # Hindi text should contain Devanagari characters
        assert any('\u0900' <= c <= '\u097F' for c in explanation.why)
    
    def test_flood_explanation(self):
        """Test flood explanation"""
        for risk_level in ["high", "moderate", "low"]:
            explanation = explainer.get_explanation("flood", risk_level, Language.EN)
            assert explanation.why is not None
    
    def test_aqi_explanation(self):
        """Test AQI explanation"""
        for category in ["good", "moderate", "poor"]:
            explanation = explainer.get_explanation("aqi", category, Language.EN)
            assert explanation.why is not None