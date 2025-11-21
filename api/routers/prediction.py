"""
Prediction endpoints router.

Handles all prediction-related API endpoints including
single predictions, batch predictions, and prediction history.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from api.schemas.customer import CustomerData, CustomerBatchRequest
from api.schemas.prediction import PredictionResponse, BatchPredictionResponse
from api.services.model_service import model_service


router = APIRouter(
    prefix="/predict",
    tags=["Predictions"],
    responses={
        503: {"description": "Service unavailable - model not loaded"},
        500: {"description": "Internal server error"},
    }
)


def get_model_service():
    """
    Dependency injection for model service.
    
    Ensures model is loaded before processing requests.
    
    Raises:
        HTTPException: If model service not ready
    """
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service not ready. Model not loaded."
        )
    return model_service


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict churn for single customer",
    description="""
    Predict customer churn probability for a single customer.
    
    Returns:
    - **prediction**: Binary churn prediction (Churn/No Churn)
    - **churn_probability**: Probability score between 0 and 1
    - **risk_category**: Risk level (Low/Medium/High Risk)
    - **confidence**: Model confidence in the prediction
    - **timestamp**: When prediction was made
    - **model_version**: Version of model used
    
    Example usage:
    ```json
    {
        "State": "CA",
        "Area_code": 415,
        "International_plan": "No",
        "Voice_mail_plan": "Yes",
        "Account_length": 128,
        ...
    }
    ```
    """,
    response_description="Prediction result with probability and risk category"
)
async def predict_single_customer(
    customer: CustomerData,
    service: Any = Depends(get_model_service)
) -> PredictionResponse:
    """
    Predict churn for a single customer.
    
    Args:
        customer: Customer data with all required features
        service: Injected model service
        
    Returns:
        Prediction response with churn probability and risk level
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Convert Pydantic model to dict
        customer_dict = customer.dict()
        
        # Make prediction
        result = service.predict_single(customer_dict)
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input data: {str(e)}"
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict churn for multiple customers",
    description="""
    Predict customer churn probability for multiple customers in a single request.
    
    Efficient batch processing for multiple predictions with aggregated statistics.
    Maximum 1000 customers per request to prevent timeout.
    
    Returns aggregated metrics including:
    - Individual predictions for each customer
    - Risk category distribution
    - Processing time
    - Total customer count
    """,
    response_description="Batch prediction results with statistics"
)
async def predict_batch_customers(
    request: CustomerBatchRequest,
    service: Any = Depends(get_model_service)
) -> BatchPredictionResponse:
    """
    Predict churn for multiple customers.
    
    Args:
        request: Batch request with list of customers
        service: Injected model service
        
    Returns:
        Batch prediction response with all results and statistics
        
    Raises:
        HTTPException: If batch prediction fails
    """
    try:
        # Validate batch size
        if len(request.customers) > 1000:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Batch size exceeds maximum of 1000 customers"
            )
        
        # Convert customers to list of dicts
        customers_data = [customer.dict() for customer in request.customers]
        
        # Make batch prediction
        result = service.predict_batch(customers_data)
        
        return BatchPredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input data: {str(e)}"
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get(
    "/example",
    response_model=Dict[str, Any],
    summary="Get example customer data",
    description="Returns example customer data that can be used for testing predictions",
    response_description="Example customer data in correct format"
)
async def get_example_customer() -> Dict[str, Any]:
    """
    Get example customer data for testing.
    
    Returns:
        Example customer data dictionary
    """
    return {
        "example_low_risk": {
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
        },
        "example_high_risk": {
            "State": "TX",
            "Area_code": 408,
            "International_plan": "Yes",
            "Voice_mail_plan": "No",
            "Account_length": 45,
            "Number_vmail_messages": 0,
            "Total_day_minutes": 350.5,
            "Total_day_calls": 130,
            "Total_day_charge": 59.59,
            "Total_eve_minutes": 280.2,
            "Total_eve_calls": 105,
            "Total_eve_charge": 23.82,
            "Total_night_minutes": 250.5,
            "Total_night_calls": 95,
            "Total_night_charge": 11.27,
            "Total_intl_minutes": 15.5,
            "Total_intl_calls": 5,
            "Total_intl_charge": 4.19,
            "Customer_service_calls": 5
        }
    }
