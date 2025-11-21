"""
API Test Suite

Basic tests for the FastAPI endpoints to ensure
proper functionality and error handling.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from api.main import app

# Create test client
client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self):
        """Test main health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "timestamp" in data
    
    def test_readiness_check(self):
        """Test readiness probe."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
    
    def test_liveness_check(self):
        """Test liveness probe."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert "alive" in data
        assert data["alive"] is True


class TestRootEndpoints:
    """Test root and general endpoints."""
    
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_docs_available(self):
        """Test that docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200


class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_get_example_data(self):
        """Test example data endpoint."""
        response = client.get("/predict/example")
        assert response.status_code == 200
        data = response.json()
        assert "example_low_risk" in data
        assert "example_high_risk" in data
    
    def test_predict_single_valid(self):
        """Test single prediction with valid data."""
        customer_data = {
            "State": "CA",
            "Area_code": 415,
            "International_plan": "No",
            "Voice_mail_plan": "Yes",
            "Account_length": 128,
            "Number_vmail_messages": 25,
            "Total_day_minutes": 265.1,
            "Total_day_calls": 110,
            "Total_day_charge": 45.07,
            "Total_eve_minutes": 197.4,
            "Total_eve_calls": 99,
            "Total_eve_charge": 16.78,
            "Total_night_minutes": 244.7,
            "Total_night_calls": 91,
            "Total_night_charge": 11.01,
            "Total_intl_minutes": 10.0,
            "Total_intl_calls": 3,
            "Total_intl_charge": 2.70,
            "Customer_service_calls": 1
        }
        
        response = client.post("/predict", json=customer_data)
        
        # Should succeed if model is loaded
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "churn_probability" in data
            assert "risk_category" in data
            assert "confidence" in data
            assert "model_version" in data
        # Or return 503 if model not loaded
        elif response.status_code == 503:
            assert "model not loaded" in response.json()["detail"].lower()
    
    def test_predict_single_invalid_field(self):
        """Test single prediction with invalid field."""
        invalid_data = {
            "State": "CA",
            "Area_code": 415,
            # Missing required fields
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_batch_valid(self):
        """Test batch prediction with valid data."""
        batch_data = {
            "customers": [
                {
                    "State": "CA",
                    "Area_code": 415,
                    "International_plan": "No",
                    "Voice_mail_plan": "Yes",
                    "Account_length": 128,
                    "Number_vmail_messages": 25,
                    "Total_day_minutes": 265.1,
                    "Total_day_calls": 110,
                    "Total_day_charge": 45.07,
                    "Total_eve_minutes": 197.4,
                    "Total_eve_calls": 99,
                    "Total_eve_charge": 16.78,
                    "Total_night_minutes": 244.7,
                    "Total_night_calls": 91,
                    "Total_night_charge": 11.01,
                    "Total_intl_minutes": 10.0,
                    "Total_intl_calls": 3,
                    "Total_intl_charge": 2.70,
                    "Customer_service_calls": 1
                }
            ]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_customers" in data
            assert "processing_time_ms" in data
            assert len(data["predictions"]) == 1


class TestModelEndpoints:
    """Test model information endpoints."""
    
    def test_model_info(self):
        """Test model info endpoint."""
        response = client.get("/model/info")
        
        if response.status_code == 200:
            data = response.json()
            assert "model_version" in data
            assert "model_type" in data
            assert "metrics" in data
        elif response.status_code == 503:
            # Model not loaded
            pass
    
    def test_model_version(self):
        """Test model version endpoint."""
        response = client.get("/model/version")
        
        if response.status_code == 200:
            data = response.json()
            assert "model_version" in data
    
    def test_model_features(self):
        """Test model features endpoint."""
        response = client.get("/model/features")
        
        if response.status_code == 200:
            data = response.json()
            assert "features" in data
            assert "n_features" in data


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_not_found(self):
        """Test 404 error for non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
    
    def test_405_method_not_allowed(self):
        """Test 405 error for wrong HTTP method."""
        response = client.post("/health")  # Should be GET
        assert response.status_code == 405


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
