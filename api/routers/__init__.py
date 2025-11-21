"""
Router modules for API endpoints.
"""

from .prediction import router as prediction_router
from .health import router as health_router
from .model import router as model_router

__all__ = [
    "prediction_router",
    "health_router",
    "model_router",
]
