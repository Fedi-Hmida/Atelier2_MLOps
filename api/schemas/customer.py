"""
Customer data schema definitions.

Defines Pydantic models for customer input data validation
with comprehensive field validation and example data.
"""

from typing import List
from pydantic import BaseModel, Field, validator


class CustomerData(BaseModel):
    """
    Schema for individual customer data input.
    
    Validates all required features for churn prediction with
    appropriate type checking, range validation, and business rules.
    """
    
    # Categorical features
    State: str = Field(
        ...,
        description="US State code (e.g., 'CA', 'TX', 'NY')",
        min_length=2,
        max_length=2,
        example="CA"
    )
    
    Area_code: int = Field(
        ...,
        description="Telephone area code",
        ge=200,
        le=999,
        example=415
    )
    
    International_plan: str = Field(
        ...,
        description="International calling plan subscription",
        example="No"
    )
    
    Voice_mail_plan: str = Field(
        ...,
        description="Voice mail service subscription",
        example="Yes"
    )
    
    # Account information
    Account_length: int = Field(
        ...,
        description="Number of days the account has been active",
        ge=1,
        le=500,
        example=128
    )
    
    Number_vmail_messages: int = Field(
        ...,
        description="Number of voice mail messages",
        ge=0,
        le=100,
        example=25
    )
    
    # Day usage metrics
    Total_day_minutes: float = Field(
        ...,
        description="Total minutes of day calls",
        ge=0.0,
        le=500.0,
        example=265.1
    )
    
    Total_day_calls: int = Field(
        ...,
        description="Total number of day calls",
        ge=0,
        le=200,
        example=110
    )
    
    Total_day_charge: float = Field(
        ...,
        description="Total charges for day calls in USD",
        ge=0.0,
        le=100.0,
        example=45.07
    )
    
    # Evening usage metrics
    Total_eve_minutes: float = Field(
        ...,
        description="Total minutes of evening calls",
        ge=0.0,
        le=500.0,
        example=197.4
    )
    
    Total_eve_calls: int = Field(
        ...,
        description="Total number of evening calls",
        ge=0,
        le=200,
        example=99
    )
    
    Total_eve_charge: float = Field(
        ...,
        description="Total charges for evening calls in USD",
        ge=0.0,
        le=100.0,
        example=16.78
    )
    
    # Night usage metrics
    Total_night_minutes: float = Field(
        ...,
        description="Total minutes of night calls",
        ge=0.0,
        le=500.0,
        example=244.7
    )
    
    Total_night_calls: int = Field(
        ...,
        description="Total number of night calls",
        ge=0,
        le=200,
        example=91
    )
    
    Total_night_charge: float = Field(
        ...,
        description="Total charges for night calls in USD",
        ge=0.0,
        le=100.0,
        example=11.01
    )
    
    # International usage metrics
    Total_intl_minutes: float = Field(
        ...,
        description="Total minutes of international calls",
        ge=0.0,
        le=50.0,
        example=10.0
    )
    
    Total_intl_calls: int = Field(
        ...,
        description="Total number of international calls",
        ge=0,
        le=50,
        example=3
    )
    
    Total_intl_charge: float = Field(
        ...,
        description="Total charges for international calls in USD",
        ge=0.0,
        le=20.0,
        example=2.70
    )
    
    # Customer service
    Customer_service_calls: int = Field(
        ...,
        description="Number of calls to customer service",
        ge=0,
        le=20,
        example=1
    )
    
    @validator('State')
    def validate_state_format(cls, v):
        """Validate state code is uppercase."""
        return v.upper()
    
    @validator('International_plan', 'Voice_mail_plan')
    def validate_yes_no_fields(cls, v):
        """Validate Yes/No fields."""
        if v not in ['Yes', 'No', 'yes', 'no']:
            raise ValueError('Must be "Yes" or "No"')
        return v.capitalize()
    
    @validator('Total_day_charge', 'Total_eve_charge', 'Total_night_charge', 'Total_intl_charge')
    def validate_charge_consistency(cls, v):
        """Validate charges are reasonable."""
        if v < 0:
            raise ValueError('Charges cannot be negative')
        return round(v, 2)
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
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
        }


class CustomerBatchRequest(BaseModel):
    """
    Schema for batch prediction requests.
    
    Accepts multiple customer records for efficient batch processing.
    """
    
    customers: List[CustomerData] = Field(
        ...,
        description="List of customer data records",
        min_items=1,
        max_items=1000  # Prevent DoS attacks
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
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
        }
