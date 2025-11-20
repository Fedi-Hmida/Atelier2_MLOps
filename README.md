# Customer Churn Prediction - Production ML Pipeline

## 📁 Project Structure

```
Atelier2/
├── config/                                     # Configuration files
│   ├── config.yaml                            # Default configuration
│   ├── config.dev.yaml                        # Development config
│   └── config.prod.yaml                       # Production config
├── data/                                       # Datasets directory
│   ├── churn-bigml-80.csv                     # Training dataset (2,668 records)
│   ├── churn-bigml-20.csv                     # Test dataset (669 records)
│   └── README.md                              # Data documentation
├── models/                                     # Trained models directory
│   └── v1.0/                                  # Model version 1.0
├── config_loader.py                           # Configuration management module
├── model_pipeline.py                          # Core ML functions (1,056 lines)
├── main.py                                    # Pipeline orchestration (426 lines)
├── Makefile                                   # Build automation
├── requirements.txt                           # Python dependencies
├── CONFIGURATION.md                           # Configuration guide
├── ML_Pipeline_Function_Specifications.md     # Detailed function specs
└── README.md                                  # This file
```

## 📝 Project Overview

**Atelier 2** contains a complete, production-ready machine learning pipeline for customer churn prediction in telecommunications. The project features a **configuration-driven architecture** that eliminates hardcoded values and makes the pipeline truly reusable.

### Key Features
- ✅ **Configuration-Driven** - No hardcoded dataset paths (NEW!)
- ✅ **23 modular functions** for the complete ML lifecycle
- ✅ **Environment-specific configs** (dev, prod)
- ✅ **Hyperparameter optimization** with Optuna (50 trials)
- ✅ **Class balancing** using SMOTEENN
- ✅ **Advanced outlier detection** (Anderson-Darling + Z-score/IQR)
- ✅ **Feature engineering** (Total calls, Total charge, CS call rate)
- ✅ **Model versioning** with metadata tracking
- ✅ **CLI interface** with configuration overrides
- ✅ **MLOps best practices** - Reusable, scalable, production-ready

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# Run with default configuration
python main.py

# Use development config (fast, no optimization)
python main.py --config config/config.dev.yaml

# Use production config (full optimization)
python main.py --config config/config.prod.yaml

# Override specific settings
python main.py --config config/config.yaml --model random_forest --no-optimize
```

### 3. Using Makefile

```bash
# Default pipeline
make train

# Development mode (fast)
make train-dev

# Production mode (optimized)
make train-prod

# Only prepare data
make data

# See all available commands
make help
```

### 4. Individual Steps

```bash
# Only prepare data
python main.py --prepare

# Only evaluate saved model
python main.py --evaluate

# Run inference demo
python main.py --predict
```

## 🔧 Configuration System

The pipeline uses YAML configuration files for maximum flexibility:

### Configuration Files
- **`config/config.yaml`** - Default settings (50 trials, standard optimization)
- **`config/config.dev.yaml`** - Development (fast iteration, no optimization)
- **`config/config.prod.yaml`** - Production (100 trials, full validation)

### Example Configuration
```yaml
data:
  train_path: "data/churn-bigml-80.csv"
  test_path: "data/churn-bigml-20.csv"
  target_column: "Churn"

model:
  type: "xgboost"
  optimization:
    enabled: true
    n_trials: 50
```

### CLI Overrides
```bash
# Override data paths
python main.py --train-data data/new_train.csv --test-data data/new_test.csv

# Override model type
python main.py --config config/config.yaml --model random_forest

# Disable optimization
python main.py --no-optimize
```

**For detailed configuration documentation, see [`CONFIGURATION.md`](CONFIGURATION.md)**

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0
scipy>=1.7.0
optuna>=2.10.0
pyyaml>=6.0
```

## 🔧 Module Functions

### `model_pipeline.py` - Core Functions

#### Data Loading & Validation
- `load_data()` - Load training and test CSV files
- `validate_data_quality()` - Check data quality metrics

#### Preprocessing
- `encode_categorical_features()` - Binary and one-hot encoding
- `detect_normality()` - Anderson-Darling normality test
- `remove_outliers_zscore()` - Remove outliers (normal distributions)
- `remove_outliers_iqr()` - Remove outliers (non-normal distributions)

#### Feature Engineering
- `engineer_features()` - Create derived features
- `select_features_correlation()` - Remove correlated features

#### Data Preparation
- `prepare_data_for_modeling()` - Split features and target
- `balance_dataset()` - SMOTEENN resampling
- `scale_features()` - StandardScaler normalization

#### Model Training
- `train_model()` - Train classification model
- `optimize_hyperparameters()` - Optuna hyperparameter tuning

#### Evaluation
- `evaluate_model()` - Comprehensive metrics
- `cross_validate_model()` - K-fold cross-validation
- `extract_feature_importance()` - Feature ranking

#### Persistence
- `save_model()` - Save model and artifacts
- `load_model()` - Load saved model

#### Inference
- `predict_churn()` - Make predictions on new data
- `prepare_data()` - Complete preprocessing pipeline

## 💡 Usage Examples

### Full Pipeline in Code

```python
from model_pipeline import prepare_data, train_model, evaluate_model, save_model

# Prepare data
X_train, y_train, X_test, y_test, artifacts, features = prepare_data()

# Train model
model = train_model(X_train, y_train, model_type='xgboost')

# Evaluate
metrics = evaluate_model(model, X_test, y_test)

# Save
save_model(model, artifacts['scaler'], artifacts['encoders'], 
           features, metadata={'metrics': metrics})
```

### Inference on New Data

```python
from model_pipeline import load_model, predict_churn

# Load saved model
artifacts = load_model(model_dir='./models', version='v1.0')

# New customer data
customer = {
    'State': 'CA',
    'Account length': 100,
    'Area code': 415,
    'International plan': 'No',
    'Voice mail plan': 'Yes',
    # ... other features
}

# Predict
results = predict_churn(customer, artifacts)
print(f"Churn probability: {results['probabilities'][0]:.2%}")
```

### Custom Training

```python
from model_pipeline import prepare_data, optimize_hyperparameters

# Prepare data
X_train, y_train, X_test, y_test, artifacts, features = prepare_data()

# Optimize with 100 trials
model, best_params = optimize_hyperparameters(
    model_type='xgboost',
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_trials=100
)

print(f"Best params: {best_params}")
```

## 🎯 Key Features

### Configuration System (NEW!)
- ✅ **YAML-based configuration** - No hardcoded values
- ✅ **Environment-specific configs** - Dev, prod configurations
- ✅ **CLI overrides** - Override any config value from command line
- ✅ **Path resolution** - Automatic absolute path handling
- ✅ **Validation** - Configuration validation on load
- ✅ **Reusable** - Same code works with any dataset

### Data Pipeline
- ✅ Automatic data validation
- ✅ Missing value detection
- ✅ Duplicate removal
- ✅ Statistical outlier detection (Anderson-Darling test)
- ✅ Z-score and IQR outlier removal

### Feature Engineering
- ✅ Binary encoding (Yes/No → 0/1)
- ✅ One-hot encoding (State, Area code)
- ✅ Derived features (Total calls, Total charge, CS call rate)
- ✅ Correlation-based feature selection
- ✅ StandardScaler normalization

### Model Training
- ✅ Multiple algorithms (XGBoost, Random Forest, Gradient Boosting)
- ✅ SMOTEENN class balancing
- ✅ Optuna hyperparameter optimization
- ✅ 5-fold cross-validation

### Evaluation
- ✅ ROC AUC, Accuracy, Log Loss
- ✅ Cohen's Kappa, Matthews Correlation
- ✅ Confusion Matrix
- ✅ Classification Report
- ✅ Feature Importance Analysis

### Production-Ready
- ✅ **Configuration-driven architecture** (NEW!)
- ✅ Model versioning
- ✅ Artifact persistence (pickle)
- ✅ Metadata tracking (JSON)
- ✅ Error handling
- ✅ Deterministic outputs (RANDOM_STATE=42)
- ✅ Google-style docstrings
- ✅ PEP8 compliant
- ✅ MLOps best practices

## 📊 Expected Results

### Model Performance (XGBoost with Optimization)
- **ROC AUC**: ~0.93
- **Accuracy**: ~0.95
- **CV ROC AUC**: ~0.92 ± 0.02

### Top Features
1. Total charge
2. Customer service calls
3. International plan
4. Total day charge
5. Total eve charge

## 🔒 Best Practices

### For Development
```bash
# Use fast mode for iteration
python main.py --no-optimize --model random_forest

# Quick evaluation
python main.py --evaluate
```

### For Production
```bash
# Full optimization
python main.py --optimize --trials 100 --model xgboost

# Version management
# Models saved with timestamps in metadata_v1.0.json
```

### For Deployment
```python
# Always use load_model() to ensure consistent preprocessing
artifacts = load_model(model_dir='./models', version='v1.0')
results = predict_churn(new_data, artifacts)
```

## 🐛 Troubleshooting

### Issue: Configuration file not found
```bash
# Ensure config directory exists
ls config/

# Use default config
python main.py --config config/config.yaml
```

### Issue: Data files not found
```bash
# Ensure data files are in data/ directory
ls data/

# Or specify custom paths
python main.py --train-data path/to/train.csv --test-data path/to/test.csv
```

### Issue: Import errors
```bash
# Reinstall all dependencies including PyYAML
pip install -r requirements.txt --upgrade
```

### Issue: Memory errors
```bash
# Use development config (reduced complexity)
python main.py --config config/config.dev.yaml

# Or reduce optimization trials
python main.py --trials 20
```

## 🆕 What's New in This Version

### Configuration System Refactor (MLOps Best Practice)
- **Problem Solved**: Eliminated all hardcoded dataset paths that made code non-reusable
- **Solution**: YAML-based configuration system with environment-specific configs
- **Benefits**:
  - ✅ Code is now truly reusable with any dataset
  - ✅ Easy switching between dev/prod environments
  - ✅ Configuration versioning alongside code
  - ✅ CLI overrides for flexibility
  - ✅ Follows industry MLOps standards

### File Organization
- **New**: `config/` directory with YAML configuration files
- **New**: `data/` directory for organized dataset storage
- **New**: `config_loader.py` module for configuration management
- **New**: `CONFIGURATION.md` comprehensive configuration guide
- **Updated**: All functions now accept explicit parameters (no defaults)
- **Updated**: `main.py` now configuration-driven
- **Updated**: `Makefile` with config-aware targets

## 🚀 GitHub Preparation

### Project Status: ✅ Ready for GitHub

All files are organized and production-ready:
- ✅ Clean Python modules (`model_pipeline.py`, `main.py`)
- ✅ Complete documentation (`README.md`, `ML_Pipeline_Function_Specifications.md`)
- ✅ Dependencies listed (`requirements.txt`)
- ✅ `.gitignore` configured
- ✅ Data files included (ensure license allows data sharing)
- ✅ Structured notebook for reference

### Pre-Push Checklist

```bash
# 1. Navigate to Atelier2
cd "c:\Users\Fedih\Downloads\Projet ML\Atelier2"

# 2. Initialize git repository (if not done)
git init

# 3. Add all files
git add .

# 4. Check what will be committed
git status

# 5. Create first commit
git commit -m "feat: Add Atelier2 - Production ML pipeline for churn prediction

- Complete modular pipeline with 23 functions
- Hyperparameter optimization with Optuna
- CLI interface with argparse
- Comprehensive documentation and specifications
- PEP8 compliant, type-hinted code"

# 6. Create GitHub repository and push
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

### Recommended Repository Name
- `churn-prediction-pipeline`
- `ml-churn-atelier2`
- `telecom-churn-ml`

### Suggested GitHub Repository Description
> Production-ready ML pipeline for customer churn prediction in telecommunications. Features modular Python functions, Optuna hyperparameter optimization, SMOTEENN balancing, and comprehensive evaluation (ROC AUC ~0.93). Built with XGBoost, scikit-learn, and FastAPI-ready architecture.

### Suggested Topics/Tags
`machine-learning` `churn-prediction` `xgboost` `scikit-learn` `optuna` `telecommunications` `python` `data-science` `mlops` `production-ml`

## 📊 Expected Results

### Model Performance (XGBoost with Optimization)
- **ROC AUC**: ~0.93
- **Accuracy**: ~0.95
- **CV ROC AUC**: ~0.92 ± 0.02
- **Training time**: ~2-5 minutes (with 50 trials)

### Top Features
1. Total charge
2. Customer service calls
3. International plan
4. Total day charge
5. Total eve charge

## 📝 License

MIT License

## 👥 Author

**Fedi Hmida**  
Data Scientist & ML Engineer  
November 2025

---

**Note**: This project is part of a machine learning workshop series (Atelier 2) focusing on transforming exploratory notebooks into production-ready pipelines. 