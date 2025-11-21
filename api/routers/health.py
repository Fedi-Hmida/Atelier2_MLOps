"""
Health check endpoints router.

Provides health, readiness, and liveness endpoints for
monitoring and orchestration systems.
"""

from datetime import datetime
from fastapi import APIRouter, status

from api.schemas.health import HealthResponse, ReadinessResponse, LivenessResponse
from api.services.model_service import model_service


router = APIRouter(
    prefix="/health",
    tags=["Health"],
    responses={
        200: {"description": "Success"},
    }
)


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="""
    Comprehensive health check endpoint that returns the overall
    health status of the API service.
    
    Checks:
    - Model loading status
    - Service uptime
    - Dependency health
    - Configuration status
    
    Used for monitoring and alerting systems.
    """,
    response_description="Detailed health status"
)
async def health_check() -> HealthResponse:
    """
    Perform comprehensive health check.
    
    Returns:
        Health status with model and service information
    """
    is_ready = model_service.is_ready()
    
    return HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        model_loaded=is_ready,
        model_version=model_service.model_version,
        uptime_seconds=model_service.get_uptime_seconds(),
        timestamp=datetime.now().isoformat(),
        dependencies={
            "model_service": "healthy" if is_ready else "unhealthy",
            "config": "healthy" if model_service.config else "unhealthy"
        }
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="""
    Kubernetes-style readiness probe.
    
    Indicates whether the service is ready to accept traffic.
    Returns 200 if ready, 503 if not ready.
    
    Use this for:
    - Load balancer health checks
    - Kubernetes readiness probes
    - Traffic routing decisions
    """,
    response_description="Service readiness status"
)
async def readiness_check() -> ReadinessResponse:
    """
    Check if service is ready to accept requests.
    
    Returns:
        Readiness status
    """
    is_ready = model_service.is_ready()
    
    return ReadinessResponse(
        ready=is_ready,
        reason=None if is_ready else "Model not loaded"
    )


@router.get(
    "/live",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="""
    Kubernetes-style liveness probe.
    
    Indicates whether the service is alive and functioning.
    If this endpoint fails, the service should be restarted.
    
    Use this for:
    - Kubernetes liveness probes
    - Container orchestration
    - Automatic restart decisions
    """,
    response_description="Service liveness status"
)
async def liveness_check() -> LivenessResponse:
    """
    Check if service is alive.
    
    Returns:
        Liveness status (always True unless service is crashed)
    """
    return LivenessResponse(
        alive=True,
        timestamp=datetime.now().isoformat()
    )


@router.get(
    "/startup",
    status_code=status.HTTP_200_OK,
    summary="Startup probe",
    description="""
    Kubernetes-style startup probe.
    
    Indicates whether the service has completed initialization.
    Used during application startup to prevent premature traffic routing.
    """,
    response_description="Startup completion status"
)
async def startup_check() -> dict:
    """
    Check if service startup is complete.
    
    Returns:
        Startup status with initialization details
    """
    is_ready = model_service.is_ready()
    
    return {
        "started": True,
        "initialized": is_ready,
        "model_loaded": is_ready,
        "timestamp": datetime.now().isoformat()
    }
