"""
Schema definitions for API request/response models.
"""

from .customer import CustomerData, CustomerBatchRequest
from .prediction import PredictionResponse, BatchPredictionResponse
from .health import HealthResponse
from .model import ModelInfoResponse

__all__ = [
    "CustomerData",
    "CustomerBatchRequest",
    "PredictionResponse",
    "BatchPredictionResponse",
    "HealthResponse",
    "ModelInfoResponse",
]
