"""
Health check schema definitions.

Defines Pydantic models for health check and
system status monitoring endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Schema for health check response.
    
    Provides comprehensive system health status including
    model availability, dependencies, and performance metrics.
    """
    
    status: str = Field(
        ...,
        description="Overall health status",
        example="healthy"
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready",
        example=True
    )
    
    model_version: str = Field(
        ...,
        description="Currently loaded model version",
        example="v1.0"
    )
    
    uptime_seconds: float = Field(
        ...,
        description="API uptime in seconds",
        ge=0.0,
        example=3600.5
    )
    
    timestamp: str = Field(
        ...,
        description="Health check timestamp in ISO format",
        example="2025-11-21T10:30:45.123456"
    )
    
    dependencies: dict = Field(
        default_factory=dict,
        description="Status of external dependencies",
        example={
            "database": "healthy",
            "cache": "healthy",
            "storage": "healthy"
        }
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "v1.0",
                "uptime_seconds": 3600.5,
                "timestamp": "2025-11-21T10:30:45.123456",
                "dependencies": {
                    "model_service": "healthy",
                    "config": "healthy"
                }
            }
        }


class ReadinessResponse(BaseModel):
    """
    Schema for readiness probe response.
    
    Used by orchestration systems (Kubernetes, Docker Swarm)
    to determine if the service is ready to accept traffic.
    """
    
    ready: bool = Field(
        ...,
        description="Whether the service is ready to serve requests",
        example=True
    )
    
    reason: Optional[str] = Field(
        None,
        description="Reason if not ready",
        example="Model loading in progress"
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "ready": True,
                "reason": None
            }
        }


class LivenessResponse(BaseModel):
    """
    Schema for liveness probe response.
    
    Used by orchestration systems to determine if the
    service is alive and should not be restarted.
    """
    
    alive: bool = Field(
        ...,
        description="Whether the service is alive",
        example=True
    )
    
    timestamp: str = Field(
        ...,
        description="Timestamp in ISO format",
        example="2025-11-21T10:30:45.123456"
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "alive": True,
                "timestamp": "2025-11-21T10:30:45.123456"
            }
        }
