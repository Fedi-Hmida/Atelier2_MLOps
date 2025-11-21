"""
Service layer for machine learning model operations.

Implements business logic for model management, predictions,
and inference operations with proper error handling and logging.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_pipeline import load_model, predict_churn
from config_loader import load_config, Config


class ModelService:
    """
    Service class for ML model operations.
    
    Manages model lifecycle, loading, caching, and prediction operations.
    Implements singleton pattern to ensure single model instance across requests.
    """
    
    _instance: Optional['ModelService'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model service (only once due to singleton)."""
        if not ModelService._initialized:
            self.artifacts: Optional[Dict[str, Any]] = None
            self.config: Optional[Config] = None
            self.model_version: str = "v1.0"
            self.loaded_at: Optional[datetime] = None
            self.startup_time: datetime = datetime.now()
            ModelService._initialized = True
    
    def load(self, config_path: str = "config/config.yaml", model_version: str = "v1.0") -> None:
        """
        Load model artifacts and configuration.
        
        Args:
            config_path: Path to configuration file
            model_version: Model version to load
            
        Raises:
            FileNotFoundError: If model or config files not found
            Exception: If model loading fails
        """
        try:
            # Load configuration
            self.config = load_config(config_path)
            print(f"✓ Configuration loaded from: {config_path}")
            
            # Get models directory from config
            models_dir = self.config.get('persistence.models_dir', './models')
            
            # Load model artifacts
            self.artifacts = load_model(
                model_dir=models_dir,
                version=model_version
            )
            
            self.model_version = model_version
            self.loaded_at = datetime.now()
            
            print(f"✓ Model service initialized successfully")
            print(f"  Version: {self.model_version}")
            print(f"  Loaded at: {self.loaded_at.isoformat()}")
            
        except FileNotFoundError as e:
            print(f"✗ Model files not found: {e}")
            raise
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def is_ready(self) -> bool:
        """
        Check if model service is ready for predictions.
        
        Returns:
            True if model is loaded and ready, False otherwise
        """
        return self.artifacts is not None and 'model' in self.artifacts
    
    def get_uptime_seconds(self) -> float:
        """
        Get service uptime in seconds.
        
        Returns:
            Uptime in seconds
        """
        return (datetime.now() - self.startup_time).total_seconds()
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single customer.
        
        Args:
            customer_data: Customer feature dictionary
            
        Returns:
            Prediction results with metadata
            
        Raises:
            RuntimeError: If model not loaded
            ValueError: If input data invalid
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Service not ready.")
        
        try:
            # Convert API schema field names to model feature names
            model_input = self._convert_api_to_model_features(customer_data)
            
            # Make prediction
            results = predict_churn(model_input, self.artifacts)
            
            # Format response
            prediction_response = {
                'prediction': results['churn_labels'][0],
                'churn_probability': float(results['probabilities'][0]),
                'risk_category': results['risk_categories'][0],
                'confidence': float(max(
                    results['probabilities'][0],
                    1 - results['probabilities'][0]
                )),
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_version
            }
            
            return prediction_response
            
        except KeyError as e:
            raise ValueError(f"Missing required feature: {e}")
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_batch(self, customers_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make predictions for multiple customers.
        
        Args:
            customers_data: List of customer feature dictionaries
            
        Returns:
            Batch prediction results with aggregated statistics
            
        Raises:
            RuntimeError: If model not loaded
            ValueError: If input data invalid
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Service not ready.")
        
        start_time = datetime.now()
        predictions = []
        risk_counts = {'Low Risk': 0, 'Medium Risk': 0, 'High Risk': 0}
        
        try:
            for customer_data in customers_data:
                # Make individual prediction
                prediction = self.predict_single(customer_data)
                predictions.append(prediction)
                
                # Track risk categories
                risk_category = prediction['risk_category']
                risk_counts[risk_category] = risk_counts.get(risk_category, 0) + 1
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            batch_response = {
                'predictions': predictions,
                'total_customers': len(predictions),
                'high_risk_count': risk_counts['High Risk'],
                'medium_risk_count': risk_counts['Medium Risk'],
                'low_risk_count': risk_counts['Low Risk'],
                'processing_time_ms': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
            return batch_response
            
        except Exception as e:
            raise RuntimeError(f"Batch prediction failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information and metadata.
        
        Returns:
            Dictionary with model information
            
        Raises:
            RuntimeError: If model not loaded
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Service not ready.")
        
        metadata = self.artifacts.get('metadata', {})
        feature_config = self.artifacts.get('feature_config', {})
        
        # Extract metrics
        test_metrics = metadata.get('test_metrics', {})
        metrics = {
            'accuracy': test_metrics.get('accuracy'),
            'precision': test_metrics.get('precision'),
            'recall': test_metrics.get('recall'),
            'f1_score': test_metrics.get('f1_score'),
            'roc_auc': test_metrics.get('roc_auc')
        }
        
        model_info = {
            'model_version': self.model_version,
            'model_type': metadata.get('model_type', 'Unknown'),
            'loaded_at': self.loaded_at.isoformat() if self.loaded_at else None,
            'training_date': metadata.get('training_date'),
            'metrics': metrics,
            'n_features': metadata.get('n_features', len(feature_config.get('feature_names', []))),
            'feature_names': feature_config.get('feature_names', []),
            'training_samples': metadata.get('training_samples'),
            'test_samples': metadata.get('test_samples'),
            'hyperparameters': metadata.get('best_params', {})
        }
        
        return model_info
    
    def _convert_api_to_model_features(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert API schema field names to model feature names.
        
        API uses underscores (Total_day_minutes) while model uses
        spaces (Total day minutes).
        
        Args:
            api_data: Data with API field names
            
        Returns:
            Data with model feature names
        """
        # Mapping from API field names to model feature names
        field_mapping = {
            'State': 'State',
            'Area_code': 'Area code',
            'International_plan': 'International plan',
            'Voice_mail_plan': 'Voice mail plan',
            'Account_length': 'Account length',
            'Number_vmail_messages': 'Number vmail messages',
            'Total_day_minutes': 'Total day minutes',
            'Total_day_calls': 'Total day calls',
            'Total_day_charge': 'Total day charge',
            'Total_eve_minutes': 'Total eve minutes',
            'Total_eve_calls': 'Total eve calls',
            'Total_eve_charge': 'Total eve charge',
            'Total_night_minutes': 'Total night minutes',
            'Total_night_calls': 'Total night calls',
            'Total_night_charge': 'Total night charge',
            'Total_intl_minutes': 'Total intl minutes',
            'Total_intl_calls': 'Total intl calls',
            'Total_intl_charge': 'Total intl charge',
            'Customer_service_calls': 'Customer service calls'
        }
        
        # Convert field names
        model_data = {}
        for api_field, model_field in field_mapping.items():
            if api_field in api_data:
                model_data[model_field] = api_data[api_field]
        
        return model_data


# Global model service instance
model_service = ModelService()
