# Configuration System - MLOps Best Practices

## Overview

The Atelier2 project has been refactored to use a **configuration-driven architecture**, eliminating all hardcoded dataset paths and parameters. This makes the pipeline truly reusable and production-ready.

## Changes Summary

### 1. **Configuration Files** (`config/`)
- `config.yaml` - Default configuration
- `config.dev.yaml` - Fast iteration for development
- `config.prod.yaml` - Full optimization for production

### 2. **Configuration Loader** (`config_loader.py`)
- Type-safe configuration loading
- Automatic path resolution
- Validation and error handling
- Environment-specific overrides

### 3. **Pipeline Updates**
- **`model_pipeline.py`** - All default values removed
- **`main.py`** - Now accepts `Config` object instead of hardcoded params
- Functions are now truly reusable with any dataset

### 4. **Data Organization**
- CSV files moved to `data/` folder
- Clear separation of code and data
- Ready for different datasets

## Usage

### Basic Usage
```bash
# Use default configuration
python main.py

# Use specific configuration
python main.py --config config/config.dev.yaml
```

### Development Mode (Fast)
```bash
# Quick training without optimization
python main.py --config config/config.dev.yaml

# Or using Makefile
make train-dev
```

### Production Mode (Optimized)
```bash
# Full optimization with 100 trials
python main.py --config config/config.prod.yaml

# Or using Makefile
make train-prod
```

### CLI Overrides
```bash
# Override specific settings
python main.py --config config/config.yaml --model random_forest --no-optimize

# Override data paths
python main.py --train-data data/new_train.csv --test-data data/new_test.csv
```

### Makefile Commands
```bash
make train          # Default config
make train-dev      # Development mode
make train-prod     # Production mode
make train-fast     # No optimization
make data           # Only data preparation
```

## Configuration Structure

```yaml
project:
  name: "Customer Churn Prediction"
  version: "1.0.0"

data:
  train_path: "data/churn-bigml-80.csv"
  test_path: "data/churn-bigml-20.csv"
  target_column: "Churn"
  
  encoding:
    binary_columns: [...]
    onehot_columns: [...]
  
  outlier_columns: [...]

model:
  type: "xgboost"
  random_state: 42
  
  optimization:
    enabled: true
    n_trials: 50
  
  default_params:
    xgboost: {...}

preprocessing:
  balance_method: "smoteenn"
  scaler: "standard"
  correlation_threshold: 0.95

persistence:
  models_dir: "models"
  version_format: "v{version}"

evaluation:
  cv_folds: 5
  metrics: [...]
  top_features: 15
```

## Adding New Datasets

1. **Place CSV files in `data/` folder**
2. **Create/update configuration:**
   ```yaml
   data:
     train_path: "data/your_train.csv"
     test_path: "data/your_test.csv"
     target_column: "YourTarget"
   ```
3. **Run pipeline:**
   ```bash
   python main.py --config your_config.yaml
   ```

## Benefits

✅ **No Hardcoded Values** - All paths and parameters in config  
✅ **Environment-Specific** - Different configs for dev/prod  
✅ **Reusable Code** - Same code works with any dataset  
✅ **Version Control** - Easy to track config changes  
✅ **MLOps Ready** - Follows industry best practices  
✅ **Flexible** - CLI can override any config value  
✅ **Type Safe** - Configuration validation on load  

## Migration Notes

### Before (Hardcoded)
```python
def prepare_data(
    train_path: str = "churn-bigml-80.csv",  # ❌ Hardcoded
    test_path: str = "churn-bigml-20.csv",   # ❌ Hardcoded
):
```

### After (Configuration-Driven)
```python
def prepare_data(
    train_path: str,  # ✅ Required parameter
    test_path: str,   # ✅ Required parameter
    target_col: str,  # ✅ Configurable
):
```

Configuration provides the values:
```python
config = load_config('config/config.yaml')
prepare_data(
    train_path=config.get('data.train_path'),
    test_path=config.get('data.test_path'),
    target_col=config.get('data.target_column'),
)
```

## Best Practices

1. **Never commit sensitive data** - Add to `.gitignore`
2. **Use environment variables** for secrets (future enhancement)
3. **Version your configurations** alongside code
4. **Test with dev config** before production
5. **Document configuration changes** in commit messages

## Future Enhancements

- [ ] Environment variable substitution in config
- [ ] Remote configuration storage (S3, Azure Blob)
- [ ] Configuration schema validation (JSON Schema)
- [ ] Experiment tracking integration (MLflow)
- [ ] Dynamic dataset discovery
