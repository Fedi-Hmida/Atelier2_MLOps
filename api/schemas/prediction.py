"""
Prediction response schema definitions.

Defines Pydantic models for prediction responses with
comprehensive metadata and result formatting.
"""

from typing import List
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    Schema for individual prediction response.
    
    Contains prediction result with probability scores,
    risk categorization, and metadata.
    """
    
    prediction: str = Field(
        ...,
        description="Binary prediction result",
        example="No Churn"
    )
    
    churn_probability: float = Field(
        ...,
        description="Probability of customer churning (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        example=0.12
    )
    
    risk_category: str = Field(
        ...,
        description="Risk level category",
        example="Low Risk"
    )
    
    confidence: float = Field(
        ...,
        description="Model confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        example=0.88
    )
    
    timestamp: str = Field(
        ...,
        description="Prediction timestamp in ISO format",
        example="2025-11-21T10:30:45.123456"
    )
    
    model_version: str = Field(
        ...,
        description="Model version used for prediction",
        example="v1.0"
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "prediction": "No Churn",
                "churn_probability": 0.12,
                "risk_category": "Low Risk",
                "confidence": 0.88,
                "timestamp": "2025-11-21T10:30:45.123456",
                "model_version": "v1.0"
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Schema for batch prediction response.
    
    Contains multiple prediction results with
    batch-level metadata and performance metrics.
    """
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of individual predictions"
    )
    
    total_customers: int = Field(
        ...,
        description="Total number of customers processed",
        ge=1,
        example=10
    )
    
    high_risk_count: int = Field(
        ...,
        description="Number of high-risk customers",
        ge=0,
        example=2
    )
    
    medium_risk_count: int = Field(
        ...,
        description="Number of medium-risk customers",
        ge=0,
        example=3
    )
    
    low_risk_count: int = Field(
        ...,
        description="Number of low-risk customers",
        ge=0,
        example=5
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds",
        ge=0.0,
        example=125.5
    )
    
    timestamp: str = Field(
        ...,
        description="Batch processing timestamp in ISO format",
        example="2025-11-21T10:30:45.123456"
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": "No Churn",
                        "churn_probability": 0.12,
                        "risk_category": "Low Risk",
                        "confidence": 0.88,
                        "timestamp": "2025-11-21T10:30:45.123456",
                        "model_version": "v1.0"
                    }
                ],
                "total_customers": 10,
                "high_risk_count": 2,
                "medium_risk_count": 3,
                "low_risk_count": 5,
                "processing_time_ms": 125.5,
                "timestamp": "2025-11-21T10:30:45.123456"
            }
        }
