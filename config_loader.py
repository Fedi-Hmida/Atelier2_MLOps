"""
Configuration Loader Module
============================

This module provides utilities for loading and validating configuration files
for the ML pipeline. It supports YAML configuration files with environment-specific
overrides and validation.

Best Practices:
- Separates configuration from code
- Supports multiple environments (dev, prod)
- Validates configuration structure
- Provides sensible defaults
- Environment variable substitution

Author: ML Engineering Team
Date: November 20, 2025
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


class Config:
    """
    Configuration manager for ML pipeline.
    
    Loads and validates YAML configuration files with support for
    environment-specific overrides and dynamic path resolution.
    
    Attributes:
        config_path (Path): Path to the configuration file
        data (dict): Loaded configuration data
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            ConfigurationError: If config file not found or invalid
        """
        self.config_path = Path(config_path)
        self.data = self._load_config()
        self._validate_config()
        self._resolve_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration data
            
        Raises:
            ConfigurationError: If file not found or parsing fails
        """
        if not self.config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create a configuration file or specify a valid path."
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ConfigurationError("Configuration file is empty")
            
            print(f"✓ Configuration loaded from: {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _validate_config(self):
        """
        Validate required configuration sections and fields.
        
        Raises:
            ConfigurationError: If required fields are missing
        """
        required_sections = ['data', 'model', 'preprocessing', 'persistence']
        required_data_fields = ['train_path', 'test_path', 'target_column']
        
        # Check required sections
        for section in required_sections:
            if section not in self.data:
                raise ConfigurationError(
                    f"Missing required section: '{section}' in configuration"
                )
        
        # Check required data fields
        for field in required_data_fields:
            if field not in self.data['data']:
                raise ConfigurationError(
                    f"Missing required field: 'data.{field}' in configuration"
                )
        
        # Validate model type
        valid_models = ['xgboost', 'random_forest', 'gradient_boosting']
        model_type = self.data.get('model', {}).get('type')
        if model_type not in valid_models:
            raise ConfigurationError(
                f"Invalid model type: '{model_type}'. "
                f"Must be one of: {valid_models}"
            )
        
        print("✓ Configuration validation passed")
    
    def _resolve_paths(self):
        """
        Resolve relative paths to absolute paths based on project root.
        
        Converts all path fields in configuration to absolute paths,
        making the configuration portable across different working directories.
        """
        # Get project root (parent of config directory)
        if self.config_path.parent.name == 'config':
            project_root = self.config_path.parent.parent
        else:
            project_root = Path.cwd()
        
        # Resolve data paths
        train_path = self.data['data']['train_path']
        test_path = self.data['data']['test_path']
        
        if not Path(train_path).is_absolute():
            self.data['data']['train_path'] = str(project_root / train_path)
        
        if not Path(test_path).is_absolute():
            self.data['data']['test_path'] = str(project_root / test_path)
        
        # Resolve model directory
        models_dir = self.data['persistence'].get('models_dir', 'models')
        if not Path(models_dir).is_absolute():
            self.data['persistence']['models_dir'] = str(project_root / models_dir)
        
        # Resolve log directory if logging is enabled
        if self.data.get('logging', {}).get('save_logs', False):
            log_dir = self.data['logging'].get('log_dir', 'logs')
            if not Path(log_dir).is_absolute():
                self.data['logging']['log_dir'] = str(project_root / log_dir)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Dot-separated key path (e.g., 'data.train_path')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get('model.type')
            'xgboost'
            >>> config.get('model.optimization.n_trials')
            50
        """
        keys = key.split('.')
        value = self.data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get complete data configuration section."""
        return self.data.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get complete model configuration section."""
        return self.data.get('model', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get complete preprocessing configuration section."""
        return self.data.get('preprocessing', {})
    
    def get_persistence_config(self) -> Dict[str, Any]:
        """Get complete persistence configuration section."""
        return self.data.get('persistence', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get complete evaluation configuration section."""
        return self.data.get('evaluation', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self.data.copy()
    
    def display(self):
        """Display formatted configuration summary."""
        print("\n" + "=" * 70)
        print("CONFIGURATION SUMMARY".center(70))
        print("=" * 70)
        
        # Project info
        project = self.data.get('project', {})
        print(f"\nProject: {project.get('name', 'N/A')}")
        print(f"Version: {project.get('version', 'N/A')}")
        
        # Data paths
        print(f"\nData Configuration:")
        print(f"  Training: {self.get('data.train_path')}")
        print(f"  Testing:  {self.get('data.test_path')}")
        print(f"  Target:   {self.get('data.target_column')}")
        
        # Model config
        print(f"\nModel Configuration:")
        print(f"  Type: {self.get('model.type')}")
        print(f"  Optimization: {self.get('model.optimization.enabled')}")
        if self.get('model.optimization.enabled'):
            print(f"  Trials: {self.get('model.optimization.n_trials')}")
        
        # Paths
        print(f"\nPersistence:")
        print(f"  Models: {self.get('persistence.models_dir')}")
        
        print("=" * 70)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use default.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        
    Returns:
        Config object with loaded configuration
        
    Example:
        >>> config = load_config('config/config.yaml')
        >>> train_path = config.get('data.train_path')
    """
    if config_path is None:
        # Try to find config in standard locations
        possible_paths = [
            'config/config.yaml',
            'config.yaml',
            '../config/config.yaml'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                config_path = path
                break
        else:
            raise ConfigurationError(
                "No configuration file found. Please specify config path or "
                "create config/config.yaml"
            )
    
    return Config(config_path)


def get_config_path_from_env() -> Optional[str]:
    """
    Get configuration path from environment variable.
    
    Returns:
        Configuration path from ML_CONFIG_PATH environment variable
        
    Example:
        export ML_CONFIG_PATH=config/config.prod.yaml
    """
    return os.getenv('ML_CONFIG_PATH')


# Example usage
if __name__ == "__main__":
    """Example usage of configuration loader."""
    
    try:
        # Load default configuration
        config = load_config('config/config.yaml')
        
        # Display configuration
        config.display()
        
        # Access configuration values
        print("\n\nExample Access Patterns:")
        print(f"Train path: {config.get('data.train_path')}")
        print(f"Model type: {config.get('model.type')}")
        print(f"N trials: {config.get('model.optimization.n_trials')}")
        print(f"Random state: {config.get('model.random_state')}")
        
        # Get configuration sections
        data_config = config.get_data_config()
        print(f"\nData config keys: {list(data_config.keys())}")
        
    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
