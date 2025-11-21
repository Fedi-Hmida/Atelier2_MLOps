"""
Model information schema definitions.

Defines Pydantic models for model metadata and
information endpoint responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ModelMetrics(BaseModel):
    """Schema for model performance metrics."""
    
    accuracy: Optional[float] = Field(
        None,
        description="Model accuracy score",
        ge=0.0,
        le=1.0,
        example=0.95
    )
    
    precision: Optional[float] = Field(
        None,
        description="Model precision score",
        ge=0.0,
        le=1.0,
        example=0.92
    )
    
    recall: Optional[float] = Field(
        None,
        description="Model recall score",
        ge=0.0,
        le=1.0,
        example=0.89
    )
    
    f1_score: Optional[float] = Field(
        None,
        description="Model F1 score",
        ge=0.0,
        le=1.0,
        example=0.90
    )
    
    roc_auc: Optional[float] = Field(
        None,
        description="ROC AUC score",
        ge=0.0,
        le=1.0,
        example=0.94
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.89,
                "f1_score": 0.90,
                "roc_auc": 0.94
            }
        }


class ModelInfoResponse(BaseModel):
    """
    Schema for model information response.
    
    Provides comprehensive metadata about the loaded model
    including version, type, metrics, and features.
    """
    
    model_version: str = Field(
        ...,
        description="Model version identifier",
        example="v1.0"
    )
    
    model_type: str = Field(
        ...,
        description="Type of machine learning model",
        example="GradientBoostingClassifier"
    )
    
    loaded_at: str = Field(
        ...,
        description="Timestamp when model was loaded",
        example="2025-11-21T09:00:00.000000"
    )
    
    training_date: Optional[str] = Field(
        None,
        description="Date when model was trained",
        example="2025-11-20"
    )
    
    metrics: Optional[ModelMetrics] = Field(
        None,
        description="Model performance metrics"
    )
    
    n_features: int = Field(
        ...,
        description="Number of features used by the model",
        ge=1,
        example=25
    )
    
    feature_names: List[str] = Field(
        default_factory=list,
        description="List of feature names",
        example=["Account_length", "Total_day_calls", "Customer_service_calls"]
    )
    
    training_samples: Optional[int] = Field(
        None,
        description="Number of samples used for training",
        ge=1,
        example=2666
    )
    
    test_samples: Optional[int] = Field(
        None,
        description="Number of samples used for testing",
        ge=1,
        example=667
    )
    
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model hyperparameters",
        example={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3
        }
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "model_version": "v1.0",
                "model_type": "GradientBoostingClassifier",
                "loaded_at": "2025-11-21T09:00:00.000000",
                "training_date": "2025-11-20",
                "metrics": {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.89,
                    "f1_score": 0.90,
                    "roc_auc": 0.94
                },
                "n_features": 25,
                "feature_names": ["Account_length", "Total_day_calls"],
                "training_samples": 2666,
                "test_samples": 667,
                "hyperparameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.1
                }
            }
        }
