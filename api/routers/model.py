"""
Model information endpoints router.

Provides endpoints for retrieving model metadata,
version information, and performance metrics.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Any

from api.schemas.model import ModelInfoResponse
from api.services.model_service import model_service


router = APIRouter(
    prefix="/model",
    tags=["Model"],
    responses={
        503: {"description": "Service unavailable - model not loaded"},
        500: {"description": "Internal server error"},
    }
)


def get_model_service():
    """
    Dependency injection for model service.
    
    Raises:
        HTTPException: If model service not ready
    """
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service not ready. Model not loaded."
        )
    return model_service


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get model information",
    description="""
    Retrieve comprehensive information about the currently loaded model.
    
    Returns:
    - Model version and type
    - Performance metrics (accuracy, precision, recall, F1, ROC-AUC)
    - Feature information
    - Training metadata
    - Hyperparameters
    
    Useful for:
    - Model governance and tracking
    - Performance monitoring
    - Feature engineering validation
    - Model comparison
    """,
    response_description="Detailed model information and metrics"
)
async def get_model_info(
    service: Any = Depends(get_model_service)
) -> ModelInfoResponse:
    """
    Get comprehensive model information.
    
    Args:
        service: Injected model service
        
    Returns:
        Model information with metrics and metadata
        
    Raises:
        HTTPException: If retrieving info fails
    """
    try:
        info = service.get_model_info()
        return ModelInfoResponse(**info)
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get(
    "/version",
    status_code=status.HTTP_200_OK,
    summary="Get model version",
    description="Returns the currently loaded model version",
    response_description="Model version string"
)
async def get_model_version(
    service: Any = Depends(get_model_service)
) -> dict:
    """
    Get current model version.
    
    Args:
        service: Injected model service
        
    Returns:
        Dictionary with version information
    """
    return {
        "model_version": service.model_version,
        "loaded_at": service.loaded_at.isoformat() if service.loaded_at else None
    }


@router.get(
    "/features",
    status_code=status.HTTP_200_OK,
    summary="Get model features",
    description="""
    Returns the list of features expected by the model.
    
    Useful for:
    - Feature validation
    - Client implementation
    - Data preparation
    - Integration testing
    """,
    response_description="List of required feature names"
)
async def get_model_features(
    service: Any = Depends(get_model_service)
) -> dict:
    """
    Get model feature names.
    
    Args:
        service: Injected model service
        
    Returns:
        Dictionary with feature information
    """
    try:
        info = service.get_model_info()
        return {
            "features": info['feature_names'],
            "n_features": info['n_features'],
            "model_version": info['model_version']
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve features: {str(e)}"
        )


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Get model performance metrics",
    description="""
    Returns model performance metrics from testing/validation.
    
    Metrics include:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    
    These metrics were calculated during model training and validation.
    """,
    response_description="Model performance metrics"
)
async def get_model_metrics(
    service: Any = Depends(get_model_service)
) -> dict:
    """
    Get model performance metrics.
    
    Args:
        service: Injected model service
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        info = service.get_model_info()
        return {
            "metrics": info['metrics'],
            "model_version": info['model_version'],
            "training_samples": info.get('training_samples'),
            "test_samples": info.get('test_samples')
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )
