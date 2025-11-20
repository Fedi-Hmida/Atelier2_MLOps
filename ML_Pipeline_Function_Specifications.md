# Customer Churn Prediction - ML Pipeline Function Specifications
## Production-Ready Modular Architecture

**Project:** Customer Churn Prediction for Telecommunications  
**Date:** November 20, 2025  
**Author:** Senior ML Engineering Team  
**Version:** 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [Module 1: Data Loading & Validation](#module-1-data-loading--validation)
3. [Module 2: Exploratory Data Analysis](#module-2-exploratory-data-analysis)
4. [Module 3: Data Preprocessing](#module-3-data-preprocessing)
5. [Module 4: Feature Engineering](#module-4-feature-engineering)
6. [Module 5: Data Preparation](#module-5-data-preparation)
7. [Module 6: Model Training](#module-6-model-training)
8. [Module 7: Model Evaluation](#module-7-model-evaluation)
9. [Module 8: Model Persistence](#module-8-model-persistence)
10. [Module 9: Inference Pipeline](#module-9-inference-pipeline)
11. [Utility Functions](#utility-functions)
12. [Dependencies & Installation](#dependencies--installation)
13. [Usage Examples](#usage-examples)

---

## Overview

This document provides detailed specifications for each function in the modular customer churn prediction pipeline. Each function is designed to be:

- **Independent**: Can be tested and used standalone
- **Reusable**: Works across different datasets with minimal modification
- **Production-ready**: Includes error handling, logging, and validation
- **Well-documented**: Clear inputs, outputs, and purpose

### Pipeline Architecture

```
Data Loading → Validation → EDA → Preprocessing → Feature Engineering 
    ↓
Feature Selection → Data Splitting → Balancing → Scaling
    ↓
Model Training → Hyperparameter Optimization → Evaluation
    ↓
Model Saving → Model Loading → Inference
```

---

## Module 1: Data Loading & Validation

### Function 1.1: `load_data()`

**Goal:** Load training and test datasets from CSV files with basic validation

**Inputs:**
- `train_path` (str): Path to training CSV file
  - Default: `'churn-bigml-80.csv'`
- `test_path` (str): Path to test CSV file
  - Default: `'churn-bigml-20.csv'`
- `validate_schema` (bool): Whether to validate column schema
  - Default: `True`

**Outputs:**
- `train_df` (pd.DataFrame): Training dataset
- `test_df` (pd.DataFrame): Test dataset

**Required Libraries:**
```python
import pandas as pd
import os
from pathlib import Path
```

**Function Signature:**
```python
def load_data(train_path='churn-bigml-80.csv', 
              test_path='churn-bigml-20.csv',
              validate_schema=True):
    """
    Load training and test datasets from CSV files
    
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If schema validation fails
    """
```

**Best Practices:**
- Check file existence before loading
- Validate column names match expected schema
- Log dataset shapes and basic info
- Handle missing files gracefully with informative errors
- Support both relative and absolute paths

**Error Handling:**
```python
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training file not found: {train_path}")
```

---

### Function 1.2: `validate_data_quality()`

**Goal:** Perform comprehensive data quality checks and return detailed report

**Inputs:**
- `df` (pd.DataFrame): Dataset to validate
- `dataset_name` (str): Name for logging (e.g., "Training", "Test")
  - Default: `"Dataset"`
- `check_duplicates` (bool): Whether to check for duplicate rows
  - Default: `True`
- `print_report` (bool): Whether to print report to console
  - Default: `True`

**Outputs:**
- `quality_report` (dict): Dictionary containing:
  - `dataset_name` (str): Dataset identifier
  - `shape` (tuple): Dataset dimensions
  - `missing_values` (dict): Missing value count per column
  - `duplicate_count` (int): Number of duplicate rows
  - `data_types` (dict): Data type per column
  - `unique_counts` (dict): Number of unique values per column
  - `memory_usage` (float): Memory usage in MB

**Required Libraries:**
```python
import pandas as pd
import numpy as np
```

**Function Signature:**
```python
def validate_data_quality(df, dataset_name="Dataset", 
                         check_duplicates=True, 
                         print_report=True):
    """
    Validate data quality and return comprehensive report
    
    Returns:
        dict: Quality metrics and statistics
    """
```

**Best Practices:**
- Calculate memory usage to identify large datasets
- Identify columns with high missing value rates (>20%)
- Flag potential data type issues (e.g., numeric stored as string)
- Return structured dictionary for programmatic access
- Generate human-readable summary when print_report=True

**Example Output:**
```python
{
    'dataset_name': 'Training Set',
    'shape': (2667, 20),
    'missing_values': {'State': 0, 'Account length': 0, ...},
    'duplicate_count': 0,
    'data_types': {'State': 'object', 'Account length': 'int64', ...},
    'unique_counts': {'State': 51, 'Area code': 3, ...},
    'memory_usage': 0.42
}
```

---

## Module 2: Exploratory Data Analysis

### Function 2.1: `perform_eda()`

**Goal:** Generate comprehensive exploratory data analysis with visualizations

**Inputs:**
- `df` (pd.DataFrame): Dataset to analyze
- `target_col` (str): Target variable column name
  - Default: `'Churn'`
- `categorical_threshold` (int): Max unique values to treat as categorical
  - Default: `50`
- `save_plots` (bool): Whether to save plots to disk
  - Default: `False`
- `output_dir` (str): Directory for saved plots
  - Default: `'./plots/'`
- `figsize` (tuple): Figure size for plots
  - Default: `(15, 10)`

**Outputs:**
- `eda_summary` (dict): Dictionary containing:
  - `statistical_summary` (pd.DataFrame): Descriptive statistics
  - `categorical_columns` (list): List of categorical columns
  - `numerical_columns` (list): List of numerical columns
  - `target_distribution` (dict): Target variable value counts
  - `correlation_matrix` (pd.DataFrame): Correlation matrix for numerical features
- Generates visualization files (if `save_plots=True`)

**Required Libraries:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
```

**Function Signature:**
```python
def perform_eda(df, target_col='Churn', 
                categorical_threshold=50,
                save_plots=False, 
                output_dir='./plots/',
                figsize=(15, 10)):
    """
    Perform comprehensive exploratory data analysis
    
    Returns:
        dict: EDA summary statistics and insights
    """
```

**Best Practices:**
- Automatically distinguish categorical vs numerical features
- Generate distribution plots for all features
- Create target variable analysis (churn rate by categories)
- Calculate correlation matrix for numerical features
- Use consistent color schemes and plot styles
- Add timestamps to saved plots to prevent overwriting
- Close plots after saving to free memory

**Visualization Types:**
1. Categorical features: Count plots
2. Numerical features: Histograms with KDE
3. Target distribution: Count plot with percentages
4. Churn by categories: Grouped bar charts
5. Correlation heatmap: Annotated heatmap

---

## Module 3: Data Preprocessing

### Function 3.1: `identify_column_types()`

**Goal:** Automatically categorize columns into categorical and numerical

**Inputs:**
- `df` (pd.DataFrame): Input dataset
- `categorical_threshold` (int): Max unique values for categorical classification
  - Default: `50`
- `force_categorical` (list): Columns to force as categorical
  - Default: `[]`
- `force_numerical` (list): Columns to force as numerical
  - Default: `[]`
- `exclude_cols` (list): Columns to exclude from analysis
  - Default: `[]`

**Outputs:**
- `column_config` (dict): Dictionary with:
  - `categorical` (list): Categorical column names
  - `numerical` (list): Numerical column names
  - `binary` (list): Binary column names (2 unique values)
  - `target` (str): Target column name (if identified)

**Required Libraries:**
```python
import pandas as pd
import numpy as np
```

**Function Signature:**
```python
def identify_column_types(df, categorical_threshold=50,
                         force_categorical=None,
                         force_numerical=None,
                         exclude_cols=None):
    """
    Automatically categorize columns by type
    
    Returns:
        dict: Column categorization
    """
```

**Best Practices:**
- Use heuristics: `unique_count < threshold` or `dtype == 'object'`
- Identify binary columns separately (exactly 2 unique values)
- Allow manual override for domain-specific knowledge
- Exclude target variable from categorization
- Log categorization results for transparency

---

### Function 3.2: `encode_categorical_features()`

**Goal:** Encode categorical variables using appropriate encoding strategies

**Inputs:**
- `train_df` (pd.DataFrame): Training dataset
- `test_df` (pd.DataFrame): Test dataset (optional)
  - Default: `None`
- `binary_cols` (list): Columns for binary encoding
  - Default: `['International plan', 'Voice mail plan']`
- `binary_mapping` (dict): Custom binary mapping
  - Default: `{'No': 0, 'Yes': 1, False: 0, True: 1}`
- `onehot_cols` (list): Columns for one-hot encoding
  - Default: `['State', 'Area code']`
- `target_col` (str): Target column name
  - Default: `'Churn'`
- `return_encoders` (bool): Whether to return fitted encoders
  - Default: `True`

**Outputs:**
- `train_encoded` (pd.DataFrame): Encoded training data
- `test_encoded` (pd.DataFrame): Encoded test data (if test_df provided)
- `encoders` (dict): Dictionary of fitted encoder objects
  - Keys: encoder names (e.g., 'state_encoder', 'area_encoder')
  - Values: Fitted encoder objects

**Required Libraries:**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
```

**Function Signature:**
```python
def encode_categorical_features(train_df, test_df=None,
                                binary_cols=None,
                                binary_mapping=None,
                                onehot_cols=None,
                                target_col='Churn',
                                return_encoders=True):
    """
    Encode categorical features using multiple strategies
    
    Returns:
        tuple: (train_encoded, test_encoded, encoders)
    """
```

**Best Practices:**
- Convert target variable to integer (0/1) first
- Apply binary mapping before one-hot encoding
- Use `OneHotEncoder` with `handle_unknown='ignore'`
- Fit encoders on training data only
- Store encoders for inference pipeline
- Ensure consistent feature order between train/test
- Handle missing categories in test set gracefully
- Preserve DataFrame index during encoding

**Encoding Strategy:**
1. **Binary Encoding**: For Yes/No columns → {0, 1}
2. **One-Hot Encoding**: For categorical with many levels (State, Area code)
3. **Target Encoding**: Convert True/False to 1/0

---

### Function 3.3: `detect_normality()`

**Goal:** Test numerical features for normal distribution using Anderson-Darling test

**Inputs:**
- `df` (pd.DataFrame): Dataset to analyze
- `columns` (list): Columns to test for normality
- `significance_level` (float): Significance level (α)
  - Default: `0.05`
- `test_method` (str): Statistical test to use
  - Default: `'anderson'` (options: 'anderson', 'shapiro')

**Outputs:**
- `normality_results` (dict): Dictionary with:
  - `normal_columns` (list): Normally distributed columns
  - `non_normal_columns` (list): Non-normally distributed columns
  - `test_statistics` (dict): Detailed test statistics per column

**Required Libraries:**
```python
from scipy.stats import anderson, shapiro
import pandas as pd
import numpy as np
```

**Function Signature:**
```python
def detect_normality(df, columns, 
                    significance_level=0.05,
                    test_method='anderson'):
    """
    Test columns for normal distribution
    
    Returns:
        dict: Normality test results
    """
```

**Best Practices:**
- Use Anderson-Darling test (more robust for large datasets)
- Use critical value at index 2 (5% significance level)
- Log test statistics for documentation
- Return structured results for downstream processing
- Consider sample size when interpreting results

**Interpretation:**
- Anderson-Darling: `statistic < critical_values[2]` → Normal
- Shapiro-Wilk: `p_value > significance_level` → Normal

---

### Function 3.4: `remove_outliers_zscore()`

**Goal:** Remove outliers from normally distributed features using Z-score method

**Inputs:**
- `df` (pd.DataFrame): Input dataset
- `columns` (list): Columns to process for outliers
- `threshold` (float): Z-score threshold (typically 3.0)
  - Default: `3.0`
- `method` (str): Outlier removal method
  - Default: `'remove'` (options: 'remove', 'cap', 'winsorize')

**Outputs:**
- `df_cleaned` (pd.DataFrame): Dataset with outliers handled
- `outlier_report` (dict): Dictionary with:
  - `original_shape` (tuple): Original dataset dimensions
  - `final_shape` (tuple): Final dataset dimensions
  - `removed_count` (int): Number of rows removed
  - `removed_percentage` (float): Percentage of data removed
  - `outliers_per_column` (dict): Outlier count per column

**Required Libraries:**
```python
import pandas as pd
import numpy as np
from scipy.stats import zscore
```

**Function Signature:**
```python
def remove_outliers_zscore(df, columns, 
                          threshold=3.0,
                          method='remove'):
    """
    Remove outliers using Z-score method for normal distributions
    
    Returns:
        tuple: (df_cleaned, outlier_report)
    """
```

**Best Practices:**
- Calculate Z-scores for specified columns only
- Remove rows where ANY column has |z| > threshold
- Log percentage of data removed (warn if >10%)
- Preserve original index for traceability
- Consider alternative methods if too much data removed:
  - **Capping**: Replace outliers with threshold values
  - **Winsorization**: Replace with percentile values

**Algorithm:**
1. Calculate Z-scores: `z = (x - μ) / σ`
2. Identify outliers: `|z| > threshold`
3. Filter DataFrame: Keep rows where all features satisfy threshold

---

### Function 3.5: `remove_outliers_iqr()`

**Goal:** Remove outliers from non-normal features using IQR method

**Inputs:**
- `df` (pd.DataFrame): Input dataset
- `columns` (list): Columns to process for outliers
- `multiplier` (float): IQR multiplier for bounds
  - Default: `1.5` (standard)
  - Conservative: `3.0`
- `method` (str): Outlier handling method
  - Default: `'remove'`

**Outputs:**
- `df_cleaned` (pd.DataFrame): Dataset with outliers handled
- `outlier_report` (dict): Similar to zscore function

**Required Libraries:**
```python
import pandas as pd
import numpy as np
```

**Function Signature:**
```python
def remove_outliers_iqr(df, columns, 
                       multiplier=1.5,
                       method='remove'):
    """
    Remove outliers using IQR method for non-normal distributions
    
    Returns:
        tuple: (df_cleaned, outlier_report)
    """
```

**Best Practices:**
- Calculate IQR per column independently
- Use standard multiplier of 1.5 (adjustable for domain needs)
- Remove rows iteratively column by column
- Track removal statistics per column
- Consider data loss vs. outlier impact trade-off

**Algorithm:**
1. Calculate quartiles: `Q1 = 25th percentile`, `Q3 = 75th percentile`
2. Calculate IQR: `IQR = Q3 - Q1`
3. Define bounds:
   - Lower: `Q1 - multiplier × IQR`
   - Upper: `Q3 + multiplier × IQR`
4. Filter: Keep rows within bounds

---

## Module 4: Feature Engineering

### Function 4.1: `engineer_features()`

**Goal:** Create derived features based on domain knowledge and hypotheses

**Inputs:**
- `df` (pd.DataFrame): Input dataset
- `feature_config` (dict): Configuration for features to create
  - Default: Uses standard telecom churn features
- `inplace` (bool): Whether to modify DataFrame in place
  - Default: `False`

**Outputs:**
- `df_engineered` (pd.DataFrame): Dataset with new features
- `feature_metadata` (dict): Metadata about created features

**Required Libraries:**
```python
import pandas as pd
import numpy as np
```

**Function Signature:**
```python
def engineer_features(df, feature_config=None, inplace=False):
    """
    Create derived features based on domain knowledge
    
    Returns:
        tuple: (df_engineered, feature_metadata)
    """
```

**Engineered Features:**

1. **Total Calls** (`Total calls`)
   - Formula: `Total day calls + Total eve calls + Total night calls + Total intl calls`
   - Purpose: Aggregate usage metric

2. **Total Charge** (`Total charge`)
   - Formula: `Total day charge + Total eve charge + Total night charge + Total intl charge`
   - Purpose: Validate hypothesis that high costs drive churn

3. **Customer Service Calls Rate** (`CScalls Rate`)
   - Formula: `Customer service calls / Account length`
   - Purpose: Normalize dissatisfaction by account tenure
   - Validates hypothesis that high service calls indicate churn risk

4. **Average Call Duration** (optional)
   - Formula: `Total minutes / Total calls`

5. **Revenue per Call** (optional)
   - Formula: `Total charge / Total calls`

**Best Practices:**
- Handle division by zero (use np.where or fillna)
- Document feature formulas in metadata
- Make function extensible for future features
- Validate new features don't have NaN/Inf values
- Log new feature statistics (mean, std, range)

**Feature Config Example:**
```python
feature_config = {
    'total_calls': {
        'columns': ['Total day calls', 'Total eve calls', 'Total night calls', 'Total intl calls'],
        'operation': 'sum'
    },
    'total_charge': {
        'columns': ['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge'],
        'operation': 'sum'
    },
    'cscalls_rate': {
        'numerator': 'Customer service calls',
        'denominator': 'Account length',
        'operation': 'divide'
    }
}
```

---

### Function 4.2: `select_features_correlation()`

**Goal:** Remove highly correlated and redundant features

**Inputs:**
- `train_df` (pd.DataFrame): Training dataset
- `test_df` (pd.DataFrame): Test dataset (optional)
- `correlation_threshold` (float): Threshold for removing correlated features
  - Default: `0.95`
- `features_to_drop` (list): Manually specified features to remove
  - Default: `None`
- `method` (str): Correlation method
  - Default: `'pearson'` (options: 'pearson', 'spearman')

**Outputs:**
- `train_selected` (pd.DataFrame): Training data with selected features
- `test_selected` (pd.DataFrame): Test data with selected features
- `selection_report` (dict): Dictionary with:
  - `dropped_features` (list): List of removed features with reasons
  - `correlation_pairs` (list): High correlation pairs found
  - `final_feature_count` (int): Number of features retained

**Required Libraries:**
```python
import pandas as pd
import numpy as np
```

**Function Signature:**
```python
def select_features_correlation(train_df, test_df=None,
                               correlation_threshold=0.95,
                               features_to_drop=None,
                               method='pearson'):
    """
    Remove highly correlated and redundant features
    
    Returns:
        tuple: (train_selected, test_selected, selection_report)
    """
```

**Best Practices:**
- Calculate correlation matrix on training data only
- Identify pairs with `|correlation| > threshold`
- For each pair, drop the feature less correlated with target
- If no target correlation, drop the second feature in pair
- Remove manually specified features (domain knowledge)
- Ensure same features dropped from train and test
- Log removed features with correlation values

**Features to Remove (Telecom Churn):**
- `Total day minutes` → highly correlated with `Total day charge` (>0.99)
- `Total eve minutes` → highly correlated with `Total eve charge` (>0.99)
- `Total night minutes` → highly correlated with `Total night charge` (>0.99)
- `Total intl minutes` → highly correlated with `Total intl charge` (>0.99)
- `Voice mail plan` → redundant with `Number vmail messages`

**Algorithm:**
```python
# Find correlation pairs
corr_matrix = train_df.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Find features to drop
to_drop = [column for column in upper_triangle.columns 
           if any(upper_triangle[column] > threshold)]
```

---

## Module 5: Data Preparation

### Function 5.1: `prepare_data_for_modeling()`

**Goal:** Split features and target, prepare final datasets for modeling

**Inputs:**
- `train_df` (pd.DataFrame): Processed training dataset
- `test_df` (pd.DataFrame): Processed test dataset
- `target_col` (str): Target variable column name
  - Default: `'Churn'`
- `validate_features` (bool): Validate feature consistency
  - Default: `True`

**Outputs:**
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series): Training labels
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series): Test labels
- `prep_metadata` (dict): Metadata about preparation

**Required Libraries:**
```python
import pandas as pd
import numpy as np
```

**Function Signature:**
```python
def prepare_data_for_modeling(train_df, test_df, 
                              target_col='Churn',
                              validate_features=True):
    """
    Split features and target for modeling
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, prep_metadata)
    """
```

**Best Practices:**
- Validate target column exists in both datasets
- Ensure same features in train and test (no data leakage)
- Check for missing values after split
- Calculate class distribution statistics
- Warn if severe class imbalance detected (<20% minority)
- Return metadata for logging and reproducibility

**Validation Checks:**
- Target column exists
- Features match between train/test
- No missing values introduced
- Data types consistent
- No duplicate indices

---

### Function 5.2: `balance_dataset()`

**Goal:** Balance imbalanced dataset using SMOTEENN technique

**Inputs:**
- `X` (pd.DataFrame or np.ndarray): Features
- `y` (pd.Series or np.ndarray): Labels
- `sampling_strategy` (float or str): Resampling strategy
  - Default: `0.428` (30:70 minority:majority ratio)
  - Options: float (ratio) or 'auto', 'minority', 'not minority'
- `random_state` (int): Random seed for reproducibility
  - Default: `42`
- `n_neighbors` (int): Number of nearest neighbors for SMOTE
  - Default: `5`

**Outputs:**
- `X_resampled` (np.ndarray): Balanced features
- `y_resampled` (np.ndarray): Balanced labels
- `balance_report` (dict): Dictionary with:
  - `original_distribution` (dict): Class counts before balancing
  - `resampled_distribution` (dict): Class counts after balancing
  - `samples_added` (int): Number of synthetic samples added
  - `samples_removed` (int): Number of samples removed by ENN

**Required Libraries:**
```python
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import pandas as pd
import numpy as np
```

**Function Signature:**
```python
def balance_dataset(X, y, 
                   sampling_strategy=0.428,
                   random_state=42,
                   n_neighbors=5):
    """
    Balance imbalanced dataset using SMOTEENN
    
    Returns:
        tuple: (X_resampled, y_resampled, balance_report)
    """
```

**Best Practices:**
- Use SMOTEENN (combines SMOTE oversampling + ENN undersampling)
- Target 30:70 ratio (not 50:50) to maintain some imbalance
- Preserve feature names if input is DataFrame
- Log class distribution before and after
- Warn if resampling changes dataset size dramatically (>50%)
- Handle edge cases (already balanced data)

**SMOTEENN Advantages:**
1. **SMOTE**: Creates synthetic minority samples
2. **ENN**: Removes noisy samples from both classes
3. Result: Cleaner, more balanced dataset

**Algorithm:**
```
Original: 85% majority, 15% minority
    ↓ SMOTE (oversample minority)
Intermediate: 70% majority, 30% minority
    ↓ ENN (clean borders)
Final: ~70% majority, ~30% minority (cleaner)
```

---

### Function 5.3: `scale_features()`

**Goal:** Standardize features using StandardScaler (z-score normalization)

**Inputs:**
- `X_train` (pd.DataFrame or np.ndarray): Training features
- `X_test` (pd.DataFrame or np.ndarray): Test features
- `scaler_type` (str): Type of scaler
  - Default: `'standard'` (options: 'standard', 'minmax', 'robust')
- `return_scaler` (bool): Whether to return fitted scaler
  - Default: `True`
- `feature_names` (list): Feature names (if input is ndarray)
  - Default: `None`

**Outputs:**
- `X_train_scaled` (np.ndarray): Scaled training features
- `X_test_scaled` (np.ndarray): Scaled test features
- `scaler` (StandardScaler): Fitted scaler object (if requested)

**Required Libraries:**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
import numpy as np
```

**Function Signature:**
```python
def scale_features(X_train, X_test, 
                  scaler_type='standard',
                  return_scaler=True,
                  feature_names=None):
    """
    Standardize features using specified scaler
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
```

**Best Practices:**
- **CRITICAL**: Fit scaler ONLY on training data
- Transform both train and test using same fitted scaler
- Store scaler for inference pipeline
- Handle both DataFrame and ndarray inputs
- Preserve feature order
- Verify scaling: mean ≈ 0, std ≈ 1 (for StandardScaler)

**Scaler Options:**
1. **StandardScaler**: `z = (x - μ) / σ`
   - Best for: Normally distributed features
   - Result: Mean=0, Std=1

2. **MinMaxScaler**: `x_scaled = (x - min) / (max - min)`
   - Best for: Bounded distributions, neural networks
   - Result: Range [0, 1]

3. **RobustScaler**: Uses median and IQR
   - Best for: Data with outliers
   - Result: Robust to outliers

---

## Module 6: Model Training

### Function 6.1: `compare_baseline_models()`

**Goal:** Train and compare multiple classification algorithms

**Inputs:**
- `X_train` (np.ndarray): Scaled training features
- `y_train` (np.ndarray): Training labels
- `X_test` (np.ndarray): Scaled test features
- `y_test` (np.ndarray): Test labels
- `models_dict` (dict): Dictionary of models to train
  - Default: Uses standard 10 algorithms
- `scoring_metrics` (list): Metrics to calculate
  - Default: `['accuracy', 'precision', 'recall', 'f1']`
- `cv_folds` (int): Number of cross-validation folds
  - Default: `5`

**Outputs:**
- `results_df` (pd.DataFrame): Comparison table with all metrics
- `trained_models` (dict): Dictionary of fitted model objects
- `training_times` (dict): Training time per model

**Required Libraries:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import time
```

**Function Signature:**
```python
def compare_baseline_models(X_train, y_train, X_test, y_test,
                           models_dict=None,
                           scoring_metrics=None,
                           cv_folds=5):
    """
    Train and compare multiple classification models
    
    Returns:
        tuple: (results_df, trained_models, training_times)
    """
```

**Default Models:**
1. Logistic Regression (class_weight='balanced')
2. Support Vector Machine (class_weight='balanced')
3. Decision Tree (class_weight='balanced')
4. Random Forest (class_weight='balanced')
5. Naive Bayes
6. K-Nearest Neighbors
7. Gradient Boosting
8. AdaBoost
9. XGBoost
10. Neural Network (MLP)

**Best Practices:**
- Use `class_weight='balanced'` for imbalanced data
- Measure training time for each model
- Calculate multiple metrics (not just accuracy)
- Sort results by primary metric (F1 or ROC AUC)
- Use consistent `random_state=42` across all models
- Handle models that don't support class_weight parameter
- Suppress convergence warnings for quick comparison

**Output Format:**
```
| Model                  | Accuracy | Precision | Recall | F1    | Train Time |
|------------------------|----------|-----------|--------|-------|------------|
| Random Forest          | 0.9452   | 0.8234    | 0.7891 | 0.806 | 2.34s      |
| XGBoost                | 0.9423   | 0.8156    | 0.7823 | 0.798 | 1.87s      |
| Gradient Boosting      | 0.9389   | 0.8012    | 0.7745 | 0.787 | 3.21s      |
...
```

---

### Function 6.2: `optimize_hyperparameters()`

**Goal:** Optimize model hyperparameters using Optuna framework

**Inputs:**
- `model_type` (str): Model to optimize
  - Options: `'random_forest'`, `'xgboost'`, `'gradient_boosting'`
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training labels
- `X_test` (np.ndarray): Test features (for validation)
- `y_test` (np.ndarray): Test labels (for validation)
- `n_trials` (int): Number of Optuna trials
  - Default: `50`
- `optimization_metric` (str): Metric to optimize
  - Default: `'accuracy'` (options: 'accuracy', 'roc_auc', 'f1')
- `timeout` (int): Maximum optimization time in seconds
  - Default: `None` (no timeout)
- `random_state` (int): Random seed
  - Default: `42`

**Outputs:**
- `best_model` (estimator): Model trained with best parameters
- `best_params` (dict): Best hyperparameters found
- `study` (optuna.Study): Complete Optuna study object
- `optimization_history` (pd.DataFrame): Trial history

**Required Libraries:**
```python
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pandas as pd
```

**Function Signature:**
```python
def optimize_hyperparameters(model_type, X_train, y_train, 
                            X_test, y_test,
                            n_trials=50,
                            optimization_metric='accuracy',
                            timeout=None,
                            random_state=42):
    """
    Optimize hyperparameters using Optuna
    
    Returns:
        tuple: (best_model, best_params, study, optimization_history)
    """
```

**Hyperparameter Search Spaces:**

**Random Forest:**
```python
{
    'n_estimators': (100, 500, step=50),
    'max_depth': (5, 30, step=5),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20),
    'max_features': ['sqrt', 'log2', None]
}
```

**XGBoost:**
```python
{
    'n_estimators': (50, 300, step=50),
    'learning_rate': (0.01, 0.2, log=True),
    'max_depth': (3, 10),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'gamma': (0, 5)
}
```

**Best Practices:**
- Use validation set (not train set) for optimization
- Set reasonable search space bounds
- Use TPE (Tree-structured Parzen Estimator) sampler
- Silence Optuna logging during optimization
- Save study object for analysis
- Plot optimization history
- Consider early stopping if no improvement

**Post-Optimization:**
```python
# Analyze optimization
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Visualization
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

---

## Module 7: Model Evaluation

### Function 7.1: `extract_feature_importance()`

**Goal:** Extract, rank, and visualize feature importances from trained model

**Inputs:**
- `model` (estimator): Trained model with `feature_importances_` attribute
- `feature_names` (list): List of feature names
- `top_n` (int): Number of top features to return
  - Default: `20`
- `plot` (bool): Whether to generate visualization
  - Default: `False`
- `importance_type` (str): Type of importance (for XGBoost)
  - Default: `'weight'` (options: 'weight', 'gain', 'cover')

**Outputs:**
- `importance_df` (pd.DataFrame): Sorted feature importance scores
- `top_features` (list): Names of top N features
- Plot (optional): Bar chart of feature importances

**Required Libraries:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

**Function Signature:**
```python
def extract_feature_importance(model, feature_names, 
                               top_n=20,
                               plot=False,
                               importance_type='weight'):
    """
    Extract and rank feature importances
    
    Returns:
        tuple: (importance_df, top_features)
    """
```

**Best Practices:**
- Sort features by importance (descending)
- Normalize importances to sum to 1.0
- Create horizontal bar plot for easy reading
- Handle different model types (RF, XGBoost, etc.)
- For XGBoost, support multiple importance types:
  - **weight**: Number of times feature is used
  - **gain**: Average gain when feature is used
  - **cover**: Average coverage of splits using feature

**Output Format:**
```python
importance_df = pd.DataFrame({
    'Feature': ['Total charge', 'Customer service calls', ...],
    'Importance': [0.2341, 0.1823, ...],
    'Rank': [1, 2, ...]
})
```

---

### Function 7.2: `evaluate_model_comprehensive()`

**Goal:** Comprehensive model evaluation with multiple metrics and visualizations

**Inputs:**
- `model` (estimator): Trained model
- `X_test` (np.ndarray): Test features
- `y_test` (np.ndarray): True test labels
- `model_name` (str): Model name for reporting
  - Default: `'Model'`
- `plot_roc` (bool): Whether to plot ROC curve
  - Default: `True`
- `plot_confusion_matrix` (bool): Whether to plot confusion matrix
  - Default: `True`

**Outputs:**
- `metrics_dict` (dict): All evaluation metrics
- Displays: Confusion matrix, classification report, ROC curve

**Required Libraries:**
```python
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, log_loss,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np
```

**Function Signature:**
```python
def evaluate_model_comprehensive(model, X_test, y_test, 
                                model_name='Model',
                                plot_roc=True,
                                plot_confusion_matrix=True):
    """
    Comprehensive model evaluation with multiple metrics
    
    Returns:
        dict: Evaluation metrics
    """
```

**Metrics Calculated:**
1. **Accuracy**: Overall correctness
2. **ROC AUC Score**: Area under ROC curve
3. **Log Loss**: Probabilistic loss
4. **Cohen's Kappa**: Agreement accounting for chance
5. **Matthews Correlation Coefficient (MCC)**: Balanced metric
6. **Precision**: TP / (TP + FP) per class
7. **Recall**: TP / (TP + FN) per class
8. **F1-Score**: Harmonic mean of precision and recall
9. **Confusion Matrix**: True/False Positives/Negatives

**Best Practices:**
- Handle both binary and multiclass classification
- Generate probability predictions for probabilistic metrics
- Create publication-quality visualizations
- Print formatted classification report
- Return structured dictionary for logging
- Include confidence intervals for key metrics

**Visualization Requirements:**
1. **ROC Curve**: With AUC score in legend
2. **Confusion Matrix**: Annotated heatmap with percentages
3. **Precision-Recall Curve**: For imbalanced datasets

---

### Function 7.3: `cross_validate_model()`

**Goal:** Perform k-fold cross-validation with comprehensive reporting

**Inputs:**
- `model` (estimator): Model to validate
- `X` (np.ndarray): Features
- `y` (np.ndarray): Labels
- `cv` (int): Number of folds
  - Default: `5`
- `scoring` (str or list): Scoring metric(s)
  - Default: `'roc_auc'`
- `stratified` (bool): Use stratified k-fold
  - Default: `True`
- `n_jobs` (int): Number of parallel jobs
  - Default: `-1` (use all cores)

**Outputs:**
- `cv_results` (dict): Dictionary with:
  - `scores` (np.ndarray): CV scores per fold
  - `mean_score` (float): Mean CV score
  - `std_score` (float): Standard deviation
  - `ci_lower` (float): Lower 95% confidence interval
  - `ci_upper` (float): Upper 95% confidence interval
  - `fold_details` (list): Detailed results per fold

**Required Libraries:**
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import numpy as np
import pandas as pd
```

**Function Signature:**
```python
def cross_validate_model(model, X, y, 
                        cv=5,
                        scoring='roc_auc',
                        stratified=True,
                        n_jobs=-1):
    """
    Perform k-fold cross-validation
    
    Returns:
        dict: CV results and statistics
    """
```

**Best Practices:**
- Use StratifiedKFold for imbalanced classification
- Calculate 95% confidence interval: `mean ± 1.96 * std`
- Report both mean and std (std indicates stability)
- Support multiple scoring metrics simultaneously
- Use all available CPU cores for speed
- Plot CV scores across folds

**Interpretation:**
- **High mean, low std**: Consistent, reliable model
- **High mean, high std**: Unstable, may overfit some folds
- **Low mean, low std**: Consistently poor performance
- **CI includes baseline**: Model may not be better than random

---

## Module 8: Model Persistence

### Function 8.1: `save_model_artifacts()`

**Goal:** Save trained model and all preprocessing artifacts with metadata

**Inputs:**
- `model` (estimator): Trained model to save
- `scaler` (StandardScaler): Fitted scaler
- `encoders` (dict): Dictionary of fitted encoders
- `feature_names` (list): List of feature names
- `metadata` (dict): Additional metadata (metrics, params, etc.)
- `output_dir` (str): Directory to save artifacts
  - Default: `'./models/'`
- `model_name` (str): Base name for model files
  - Default: `'churn_model'`
- `version` (str): Model version
  - Default: `'v1.0'`

**Outputs:**
- `artifact_paths` (dict): Paths to all saved files
- Returns: Success status and file locations

**Required Libraries:**
```python
import pickle
import joblib
import json
import os
from datetime import datetime
from pathlib import Path
```

**Function Signature:**
```python
def save_model_artifacts(model, scaler, encoders, 
                        feature_names, metadata=None,
                        output_dir='./models/',
                        model_name='churn_model',
                        version='v1.0'):
    """
    Save trained model and preprocessing artifacts
    
    Returns:
        dict: Paths to saved artifacts
    """
```

**Saved Artifacts:**
1. **Model**: `{model_name}_{version}.pkl`
2. **Scaler**: `scaler_{version}.pkl`
3. **Encoders**: `encoders_{version}.pkl`
4. **Feature Names**: `feature_names_{version}.json`
5. **Metadata**: `metadata_{version}.json`

**Metadata Structure:**
```json
{
    "model_type": "XGBoost",
    "version": "v1.0",
    "training_date": "2025-11-20 14:30:00",
    "best_params": {...},
    "metrics": {
        "accuracy": 0.9452,
        "roc_auc": 0.9234,
        ...
    },
    "cv_scores": [0.92, 0.93, 0.91, 0.94, 0.92],
    "num_training_samples": 2500,
    "num_test_samples": 667,
    "selected_features": [...],
    "python_version": "3.9.7",
    "sklearn_version": "1.0.2",
    "xgboost_version": "1.5.0"
}
```

**Best Practices:**
- Create versioned directory structure: `models/v1.0/`
- Save with both pickle and joblib (redundancy)
- Include library versions in metadata
- Validate files saved successfully
- Calculate and store file checksums (MD5/SHA256)
- Compress large models with joblib compression
- Log all artifact paths for retrieval

**Directory Structure:**
```
models/
├── v1.0/
│   ├── xgboost_churn_model_v1.0.pkl
│   ├── scaler_v1.0.pkl
│   ├── encoders_v1.0.pkl
│   ├── feature_names_v1.0.json
│   └── metadata_v1.0.json
├── v1.1/
│   └── ...
└── registry.json  # Model version registry
```

---

### Function 8.2: `load_model_artifacts()`

**Goal:** Load saved model and preprocessing objects for inference

**Inputs:**
- `model_dir` (str): Directory containing model artifacts
  - Default: `'./models/'`
- `version` (str): Model version to load
  - Default: `'latest'` (loads most recent)
- `validate` (bool): Validate loaded artifacts
  - Default: `True`

**Outputs:**
- `pipeline_dict` (dict): Dictionary containing:
  - `model`: Loaded model
  - `scaler`: Loaded scaler
  - `encoders`: Loaded encoders dictionary
  - `feature_names`: List of feature names
  - `metadata`: Model metadata

**Required Libraries:**
```python
import pickle
import joblib
import json
import os
from pathlib import Path
```

**Function Signature:**
```python
def load_model_artifacts(model_dir='./models/',
                        version='latest',
                        validate=True):
    """
    Load saved model and preprocessing artifacts
    
    Returns:
        dict: Loaded pipeline components
    """
```

**Best Practices:**
- Try joblib first, fallback to pickle
- Validate all required files exist before loading
- Verify loaded objects have expected attributes
- Check version compatibility (sklearn, xgboost versions)
- Log loaded artifact information
- Handle missing or corrupted files gracefully
- Support loading by version or 'latest'

**Validation Checks:**
1. Model has `predict()` and `predict_proba()` methods
2. Scaler has `transform()` method
3. Encoders have `transform()` methods
4. Feature names match metadata
5. Number of features consistent

**Error Handling:**
```python
try:
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Try pickle fallback
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
```

---

## Module 9: Inference Pipeline

### Function 9.1: `predict_churn()`

**Goal:** End-to-end inference pipeline for new customer data

**Inputs:**
- `customer_data` (pd.DataFrame or dict): New customer data
- `artifacts` (dict): Loaded model artifacts from `load_model_artifacts()`
- `return_proba` (bool): Return probability scores
  - Default: `True`
- `return_risk_category` (bool): Return risk level labels
  - Default: `True`
- `threshold` (float): Classification threshold
  - Default: `0.5`

**Outputs:**
- `predictions` (np.ndarray): Binary predictions (0 or 1)
- `probabilities` (np.ndarray): Churn probability scores (if requested)
- `risk_categories` (list): Risk level labels (if requested)
  - 'Low Risk': prob < 0.3
  - 'Medium Risk': 0.3 ≤ prob < 0.7
  - 'High Risk': prob ≥ 0.7
- `prediction_details` (dict): Comprehensive prediction information

**Required Libraries:**
```python
import pandas as pd
import numpy as np
```

**Function Signature:**
```python
def predict_churn(customer_data, artifacts, 
                 return_proba=True,
                 return_risk_category=True,
                 threshold=0.5):
    """
    End-to-end inference pipeline for churn prediction
    
    Returns:
        dict: Predictions, probabilities, and risk categories
    """
```

**Pipeline Steps:**

1. **Input Validation**
   - Convert dict to DataFrame if needed
   - Validate required columns present
   - Check data types

2. **Categorical Encoding**
   - Apply binary mapping: {'No': 0, 'Yes': 1}
   - One-hot encode State and Area code using fitted encoders

3. **Feature Engineering**
   - Create Total calls
   - Create Total charge
   - Create CScalls Rate

4. **Feature Selection**
   - Remove correlated features
   - Select important features (from metadata)

5. **Feature Scaling**
   - Apply fitted scaler

6. **Prediction**
   - Generate predictions
   - Calculate probabilities
   - Assign risk categories

**Best Practices:**
- Handle missing features gracefully (use defaults or raise informative error)
- Validate input data format and values
- Apply same preprocessing steps as training
- Use fitted transformers (no re-fitting!)
- Provide detailed error messages
- Support both single prediction and batch predictions
- Return comprehensive results dictionary

**Output Format:**
```python
{
    'predictions': array([1, 0, 1]),  # Binary predictions
    'probabilities': array([0.78, 0.23, 0.65]),  # Churn probabilities
    'risk_categories': ['High Risk', 'Low Risk', 'Medium Risk'],
    'churn_labels': ['Churn', 'No Churn', 'Churn'],
    'confidence': array([0.78, 0.77, 0.65]),  # Max probability per sample
    'feature_values': DataFrame(...)  # Processed features (for debugging)
}
```

**Example Usage:**
```python
# Single customer
customer = {
    'State': 'TX',
    'Account length': 120,
    'Area code': 415,
    'International plan': 'Yes',
    'Voice mail plan': 'No',
    'Number vmail messages': 0,
    'Total day minutes': 265.1,
    'Total day calls': 110,
    'Total day charge': 45.07,
    'Customer service calls': 4,
    ...
}

result = predict_churn(customer, artifacts)
print(f"Churn Probability: {result['probabilities'][0]:.2%}")
print(f"Risk Category: {result['risk_categories'][0]}")
```

---

## Utility Functions

### Function U.1: `setup_logging()`

**Goal:** Configure logging for the ML pipeline

**Inputs:**
- `log_level` (str): Logging level
  - Default: `'INFO'`
- `log_file` (str): Path to log file
  - Default: `None` (console only)
- `log_format` (str): Log message format
  - Default: Standard format

**Outputs:**
- Configured logger object

**Required Libraries:**
```python
import logging
from datetime import datetime
```

**Best Practices:**
- Log to both console and file
- Include timestamps, level, and module name
- Rotate log files when they get large
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)

---

### Function U.2: `create_config()`

**Goal:** Create and validate configuration dictionary

**Inputs:**
- `config_path` (str): Path to YAML config file
  - Default: `'config.yaml'`

**Outputs:**
- `config` (dict): Configuration dictionary

**Required Libraries:**
```python
import yaml
import json
```

**Best Practices:**
- Store all hyperparameters, paths, and settings in config
- Validate config against schema
- Support both YAML and JSON formats
- Provide sensible defaults

---

### Function U.3: `generate_report()`

**Goal:** Generate comprehensive PDF/HTML report of model performance

**Inputs:**
- `results` (dict): All evaluation results
- `output_format` (str): 'pdf' or 'html'
- `output_path` (str): Where to save report

**Outputs:**
- Generated report file

**Required Libraries:**
```python
from jinja2 import Template
import matplotlib.pyplot as plt
from weasyprint import HTML  # for PDF generation
```

**Best Practices:**
- Include executive summary
- Add all visualizations
- Format metrics in tables
- Include model metadata

---

## Dependencies & Installation

### Core Libraries

```bash
# Data manipulation
pandas>=1.3.0
numpy>=1.21.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0

# Hyperparameter Optimization
optuna>=2.10.0

# Statistical Tests
scipy>=1.7.0

# Model Persistence
joblib>=1.1.0

# Configuration
pyyaml>=5.4.0

# Utilities
python-dateutil>=2.8.0
```

### Installation

```bash
# Create virtual environment
python -m venv churn_env
source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Or install individually
pip install pandas numpy matplotlib seaborn scikit-learn xgboost \
            imbalanced-learn optuna scipy joblib pyyaml
```

---

## Usage Examples

### Example 1: Complete Training Pipeline

```python
from src import *

# 1. Load and validate data
train_df, test_df = load_data()
validate_data_quality(train_df, "Training")
validate_data_quality(test_df, "Test")

# 2. EDA (optional)
eda_results = perform_eda(train_df, save_plots=True)

# 3. Preprocess
train_enc, test_enc, encoders = encode_categorical_features(train_df, test_df)

# 4. Remove outliers
normality_results = detect_normality(train_enc, numeric_columns)
train_clean, _ = remove_outliers_zscore(
    train_enc, normality_results['normal_columns']
)
train_clean, _ = remove_outliers_iqr(
    train_clean, normality_results['non_normal_columns']
)

# 5. Feature engineering
train_eng = engineer_features(train_clean)
test_eng = engineer_features(test_enc)

# 6. Feature selection
train_sel, test_sel, _ = select_features_correlation(train_eng, test_eng)

# 7. Prepare for modeling
X_train, y_train, X_test, y_test, _ = prepare_data_for_modeling(
    train_sel, test_sel
)

# 8. Balance and scale
X_balanced, y_balanced, _ = balance_dataset(X_train, y_train)
X_train_scaled, X_test_scaled, scaler = scale_features(X_balanced, X_test)

# 9. Train and optimize
results_df, models, _ = compare_baseline_models(
    X_train_scaled, y_balanced, X_test_scaled, y_test
)

best_model, best_params, study, _ = optimize_hyperparameters(
    'xgboost', X_train_scaled, y_balanced, X_test_scaled, y_test
)

# 10. Evaluate
metrics = evaluate_model_comprehensive(
    best_model, X_test_scaled, y_test, plot_roc=True
)
cv_results = cross_validate_model(best_model, X_train_scaled, y_balanced)

# 11. Save
save_model_artifacts(
    best_model, scaler, encoders, 
    feature_names=X_train.columns.tolist(),
    metadata={'metrics': metrics, 'cv_results': cv_results}
)
```

### Example 2: Inference Pipeline

```python
# Load artifacts
artifacts = load_model_artifacts('./models/', version='v1.0')

# New customer data
new_customer = {
    'State': 'CA',
    'Account length': 150,
    'Area code': 408,
    'International plan': 'Yes',
    'Voice mail plan': 'No',
    'Number vmail messages': 0,
    'Total day minutes': 300.5,
    'Total day calls': 95,
    'Total day charge': 51.09,
    'Total eve minutes': 200.3,
    'Total eve calls': 80,
    'Total eve charge': 17.03,
    'Total night minutes': 180.2,
    'Total night calls': 70,
    'Total night charge': 8.11,
    'Total intl minutes': 12.5,
    'Total intl calls': 5,
    'Total intl charge': 3.38,
    'Customer service calls': 5
}

# Make prediction
result = predict_churn(new_customer, artifacts)

print(f"Prediction: {result['churn_labels'][0]}")
print(f"Probability: {result['probabilities'][0]:.2%}")
print(f"Risk Level: {result['risk_categories'][0]}")
```

### Example 3: Batch Predictions

```python
# Load new customers from CSV
new_customers = pd.read_csv('new_customers.csv')

# Load artifacts
artifacts = load_model_artifacts()

# Predict
results = predict_churn(new_customers, artifacts)

# Create results DataFrame
results_df = pd.DataFrame({
    'Customer_ID': new_customers['Customer_ID'],
    'Churn_Prediction': results['churn_labels'],
    'Churn_Probability': results['probabilities'],
    'Risk_Category': results['risk_categories']
})

# Save results
results_df.to_csv('churn_predictions.csv', index=False)

# Generate retention campaign list (High Risk customers)
high_risk = results_df[results_df['Risk_Category'] == 'High Risk']
high_risk.to_csv('retention_campaign_targets.csv', index=False)
```

---

## Notes & Best Practices Summary

### General Principles

1. **Modularity**: Each function does one thing well
2. **Reusability**: Functions work across different datasets
3. **Reproducibility**: Set random seeds, log parameters
4. **Error Handling**: Validate inputs, provide clear error messages
5. **Documentation**: Docstrings with examples for all functions
6. **Testing**: Unit tests with >80% coverage
7. **Logging**: Comprehensive logging at appropriate levels
8. **Versioning**: Semantic versioning for models and code

### Production Deployment

1. **API Development**: Use FastAPI or Flask
2. **Containerization**: Docker for consistent environments
3. **Monitoring**: Track model performance in production
4. **Data Drift**: Monitor for distribution changes
5. **Model Retraining**: Automated pipelines for updates
6. **A/B Testing**: Compare model versions in production
7. **Security**: API authentication, input validation
8. **Scalability**: Batch predictions, caching, load balancing

### Code Quality

1. **Type Hints**: Use Python type annotations
2. **Linting**: flake8, pylint, black for formatting
3. **CI/CD**: Automated testing and deployment
4. **Version Control**: Git with meaningful commits
5. **Code Review**: Peer review before merging
6. **Documentation**: Keep README and docs updated

---

**End of Specification Document**

*This document provides comprehensive specifications for building a production-ready customer churn prediction pipeline. Each function is designed to be independent, testable, and reusable across different projects.*
