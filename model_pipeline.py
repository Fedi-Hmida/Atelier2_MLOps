import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
# Hyperparameter optimization
import optuna
import pandas as pd
# Resampling
from imblearn.combine import SMOTEENN
# Statistical tests
from scipy.stats import anderson, zscore
# Model training
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# Metrics and validation
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, log_loss,
                             matthews_corrcoef, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score
# Preprocessing and scaling
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Global constants
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================


def load_data(
    train_path: str, test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets from CSV files.

    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file

    Returns:
        Tuple of (train_df, test_df)

    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if train_df.empty or test_df.empty:
        raise ValueError("Loaded dataframes are empty")

    print(f"✓ Data loaded: Train={train_df.shape}, Test={test_df.shape}")
    return train_df, test_df


def validate_data_quality(
    df: pd.DataFrame, dataset_name: str = "Dataset"
) -> Dict[str, Any]:
    """
    Validate data quality and return comprehensive report.

    Args:
        df: Dataset to validate
        dataset_name: Name for logging

    Returns:
        Dictionary with quality metrics
    """
    report = {
        "dataset_name": dataset_name,
        "shape": df.shape,
        "missing_values": df.isna().sum().sum(),
        "duplicate_count": df.duplicated().sum(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }

    print(f"\n{dataset_name} Quality Report:")
    print(f"  Shape: {report['shape']}")
    print(f"  Missing values: {report['missing_values']}")
    print(f"  Duplicates: {report['duplicate_count']}")

    return report


# ============================================================================
# PREPROCESSING
# ============================================================================


def encode_categorical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    binary_cols: List[str] = None,
    onehot_cols: List[str] = None,
    target_col: str = "Churn",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical variables using binary and one-hot encoding.

    Args:
        train_df: Training dataset
        test_df: Test dataset
        binary_cols: Columns for binary encoding
        onehot_cols: Columns for one-hot encoding
        target_col: Target column name

    Returns:
        Tuple of (train_encoded, test_encoded, encoders)
    """
    if binary_cols is None:
        binary_cols = ["International plan", "Voice mail plan"]
    if onehot_cols is None:
        onehot_cols = ["State", "Area code"]

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    encoders = {}

    # Encode target
    train_encoded[target_col] = train_encoded[target_col].astype(int)
    test_encoded[target_col] = test_encoded[target_col].astype(int)

    # Binary encoding
    binary_mapping = {"No": 0, "Yes": 1, False: 0, True: 1}
    for col in binary_cols:
        if col in train_encoded.columns:
            train_encoded[col] = train_encoded[col].map(binary_mapping)
            test_encoded[col] = test_encoded[col].map(binary_mapping)

    # One-hot encoding
    for col in onehot_cols:
        if col in train_encoded.columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

            # Fit on train, transform both
            encoded_train = encoder.fit_transform(train_encoded[[col]])
            encoded_test = encoder.transform(test_encoded[[col]])

            # Create DataFrames
            feature_names = encoder.get_feature_names_out([col])
            train_encoded_df = pd.DataFrame(
                encoded_train, columns=feature_names, index=train_encoded.index
            )
            test_encoded_df = pd.DataFrame(
                encoded_test, columns=feature_names, index=test_encoded.index
            )

            # Drop original, add encoded
            train_encoded = train_encoded.drop(col, axis=1)
            test_encoded = test_encoded.drop(col, axis=1)
            train_encoded = pd.concat([train_encoded, train_encoded_df], axis=1)
            test_encoded = pd.concat([test_encoded, test_encoded_df], axis=1)

            encoders[f'{col.lower().replace(" ", "_")}_encoder'] = encoder

    print(f"✓ Categorical encoding complete: {len(encoders)} encoders created")
    return train_encoded, test_encoded, encoders


def detect_normality(
    df: pd.DataFrame, columns: List[str], significance_level: float = 0.05
) -> Dict[str, List[str]]:
    """
    Test columns for normal distribution using Anderson-Darling test.

    Args:
        df: Dataset to analyze
        columns: Columns to test
        significance_level: Significance level (default 0.05)

    Returns:
        Dictionary with 'normal_columns' and 'non_normal_columns' lists
    """
    normal_columns = []
    non_normal_columns = []

    for column in columns:
        if column in df.columns:
            result = anderson(df[column])
            if result.statistic < result.critical_values[2]:  # 5% level
                normal_columns.append(column)
            else:
                non_normal_columns.append(column)

    print(
        f"✓ Normality test: {len(normal_columns)} normal, {len(non_normal_columns)} non-normal"
    )
    return {"normal_columns": normal_columns, "non_normal_columns": non_normal_columns}


def remove_outliers_zscore(
    df: pd.DataFrame, columns: List[str], threshold: float = 3.0
) -> Tuple[pd.DataFrame, int]:
    """
    Remove outliers using Z-score method for normally distributed features.

    Args:
        df: Input dataset
        columns: Columns to process
        threshold: Z-score threshold (default 3.0)

    Returns:
        Tuple of (cleaned_df, removed_count)
    """
    original_shape = df.shape[0]

    if columns:
        z_scores = zscore(df[columns])
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < threshold).all(axis=1)
        df_cleaned = df[filtered_entries].copy()
    else:
        df_cleaned = df.copy()

    removed_count = original_shape - df_cleaned.shape[0]
    print(
        f"✓ Z-score outlier removal: {removed_count} rows removed ({removed_count/original_shape*100:.2f}%)"
    )

    return df_cleaned, removed_count


def remove_outliers_iqr(
    df: pd.DataFrame, columns: List[str], multiplier: float = 1.5
) -> Tuple[pd.DataFrame, int]:
    """
    Remove outliers using IQR method for non-normal distributions.

    Args:
        df: Input dataset
        columns: Columns to process
        multiplier: IQR multiplier (default 1.5)

    Returns:
        Tuple of (cleaned_df, removed_count)
    """
    original_shape = df.shape[0]
    df_cleaned = df.copy()

    for column in columns:
        if column in df_cleaned.columns:
            Q1 = df_cleaned[column].quantile(0.25)
            Q3 = df_cleaned[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            df_cleaned = df_cleaned[
                (df_cleaned[column] >= lower_bound)
                & (df_cleaned[column] <= upper_bound)
            ]

    removed_count = original_shape - df_cleaned.shape[0]
    print(
        f"✓ IQR outlier removal: {removed_count} rows removed ({removed_count/original_shape*100:.2f}%)"
    )

    return df_cleaned, removed_count


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features based on domain knowledge.

    Args:
        df: Input dataset

    Returns:
        Dataset with engineered features
    """
    df_eng = df.copy()

    # Total calls across all periods
    call_columns = [
        "Total day calls",
        "Total eve calls",
        "Total night calls",
        "Total intl calls",
    ]
    if all(col in df_eng.columns for col in call_columns):
        df_eng["Total calls"] = df_eng[call_columns].sum(axis=1)

    # Total charge across all periods
    charge_columns = [
        "Total day charge",
        "Total eve charge",
        "Total night charge",
        "Total intl charge",
    ]
    if all(col in df_eng.columns for col in charge_columns):
        df_eng["Total charge"] = df_eng[charge_columns].sum(axis=1)

    # Customer service calls rate
    if (
        "Customer service calls" in df_eng.columns
        and "Account length" in df_eng.columns
    ):
        df_eng["CScalls Rate"] = df_eng["Customer service calls"] / df_eng[
            "Account length"
        ].replace(0, 1)

    print(f"✓ Feature engineering: 3 new features created")
    return df_eng


def select_features_correlation(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features_to_drop: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Remove highly correlated and redundant features.

    Args:
        train_df: Training dataset
        test_df: Test dataset
        features_to_drop: Features to remove

    Returns:
        Tuple of (train_selected, test_selected, dropped_features)
    """
    if features_to_drop is None:
        features_to_drop = [
            "Total day minutes",
            "Total eve minutes",
            "Total night minutes",
            "Total intl minutes",
            "Voice mail plan",
        ]

    # Drop features from both datasets
    dropped = []
    for feature in features_to_drop:
        if feature in train_df.columns:
            train_df = train_df.drop(feature, axis=1)
            dropped.append(feature)
        if feature in test_df.columns:
            test_df = test_df.drop(feature, axis=1)

    print(f"✓ Feature selection: {len(dropped)} correlated features removed")
    return train_df, test_df, dropped


# ============================================================================
# DATA PREPARATION
# ============================================================================


def prepare_data_for_modeling(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "Churn"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split features and target for modeling.

    Args:
        train_df: Training dataset
        test_df: Test dataset
        target_col: Target column name

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]

    print(f"✓ Data prepared: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"  Train churn rate: {y_train.mean()*100:.2f}%")
    print(f"  Test churn rate: {y_test.mean()*100:.2f}%")

    return X_train, y_train, X_test, y_test


def balance_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: float = 0.428,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance imbalanced dataset using SMOTEENN.

    Args:
        X: Features
        y: Labels
        sampling_strategy: Desired minority/majority ratio
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    original_dist = y.value_counts().to_dict()

    smote_enn = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)

    new_dist = pd.Series(y_resampled).value_counts().to_dict()

    print(f"✓ Data balanced: {len(y)} → {len(y_resampled)} samples")
    print(f"  Original: {original_dist}")
    print(f"  Resampled: {new_dist}")

    return X_resampled, y_resampled


def scale_features(
    X_train: Union[pd.DataFrame, np.ndarray], X_test: Union[pd.DataFrame, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features using StandardScaler.

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(
        f"✓ Features scaled: mean={X_train_scaled.mean():.6f}, std={X_train_scaled.std():.6f}"
    )

    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# MODEL TRAINING
# ============================================================================


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "xgboost",
    random_state: int = RANDOM_STATE,
    **model_params,
) -> Any:
    """
    Train a classification model.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model ('xgboost', 'random_forest', 'gradient_boosting')
        random_state: Random seed
        **model_params: Additional model parameters

    Returns:
        Trained model
    """
    if model_type == "xgboost":
        model = XGBClassifier(random_state=random_state, **model_params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=random_state, **model_params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=random_state, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    print(f"✓ Model trained successfully")

    return model


def optimize_hyperparameters(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int = 50,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Optimize hyperparameters using Optuna.

    Args:
        model_type: Type of model ('xgboost' or 'random_forest')
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_trials: Number of optimization trials
        random_state: Random seed

    Returns:
        Tuple of (best_model, best_params)
    """

    def objective(trial):
        if model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "random_state": random_state,
            }
            model = XGBClassifier(**params)
        elif model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 30, step=5),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "random_state": random_state,
            }
            model = RandomForestClassifier(**params)
        else:
            raise ValueError(f"Optimization not supported for: {model_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")

    print(f"Optimizing {model_type} hyperparameters ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    print(f"✓ Optimization complete: Best accuracy = {study.best_value:.4f}")

    # Train final model with best params
    if model_type == "xgboost":
        best_model = XGBClassifier(**best_params, random_state=random_state)
    else:
        best_model = RandomForestClassifier(**best_params, random_state=random_state)

    best_model.fit(X_train, y_train)

    return best_model, best_params


# ============================================================================
# MODEL EVALUATION
# ============================================================================


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with multiple metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        model_name: Model name for reporting

    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "log_loss": log_loss(y_test, y_pred_proba),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
    }

    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - EVALUATION RESULTS")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


def cross_validate_model(
    model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = "roc_auc"
) -> Dict[str, float]:
    """
    Perform k-fold cross-validation.

    Args:
        model: Model to validate
        X: Features
        y: Labels
        cv: Number of folds
        scoring: Scoring metric

    Returns:
        Dictionary with CV results
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    results = {
        "mean_score": cv_scores.mean(),
        "std_score": cv_scores.std(),
        "min_score": cv_scores.min(),
        "max_score": cv_scores.max(),
    }

    print(f"\n{cv}-Fold Cross-Validation ({scoring}):")
    print(f"  Mean: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    print(f"  Range: [{results['min_score']:.4f}, {results['max_score']:.4f}]")

    return results


def extract_feature_importance(
    model: Any, feature_names: List[str], top_n: int = 15
) -> pd.DataFrame:
    """
    Extract and rank feature importances.

    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importances
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute")

    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} Important Features:")
    print(importance_df.head(top_n).to_string(index=False))

    return importance_df


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================


def save_model(
    model: Any,
    scaler: StandardScaler,
    encoders: Dict[str, Any],
    feature_names: List[str],
    metadata: Dict[str, Any],
    output_dir: str = "./models",
    model_name: str = "churn_model",
    version: str = "v1.0",
) -> Dict[str, str]:
    """
    Save trained model and all preprocessing artifacts.

    Args:
        model: Trained model
        scaler: Fitted scaler
        encoders: Dictionary of encoders
        feature_names: List of feature names
        metadata: Model metadata (metrics, params, etc.)
        output_dir: Output directory
        model_name: Base model name
        version: Model version

    Returns:
        Dictionary of saved file paths
    """
    output_path = Path(output_dir) / version
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save model
    model_path = output_path / f"{model_name}_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    paths["model"] = str(model_path)

    # Save scaler
    scaler_path = output_path / f"scaler_{version}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    paths["scaler"] = str(scaler_path)

    # Save encoders
    encoders_path = output_path / f"encoders_{version}.pkl"
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)
    paths["encoders"] = str(encoders_path)

    # Save feature names
    features_path = output_path / f"feature_names_{version}.json"
    with open(features_path, "w") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)
    paths["features"] = str(features_path)

    # Save metadata
    metadata["version"] = version
    metadata["saved_at"] = datetime.now().isoformat()
    metadata_path = output_path / f"metadata_{version}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    paths["metadata"] = str(metadata_path)

    print(f"\n✓ Model artifacts saved to: {output_path}")
    for key, path in paths.items():
        print(f"  {key}: {Path(path).name}")

    return paths


def load_model(model_dir: str = "./models", version: str = "v1.0") -> Dict[str, Any]:
    """
    Load saved model and preprocessing artifacts.

    Args:
        model_dir: Directory containing model artifacts
        version: Model version to load

    Returns:
        Dictionary with model, scaler, encoders, and metadata
    """
    model_path = Path(model_dir) / version

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    artifacts = {}

    # Load model
    model_file = list(model_path.glob("churn_model_*.pkl"))[0]
    with open(model_file, "rb") as f:
        artifacts["model"] = pickle.load(f)

    # Load scaler
    scaler_file = list(model_path.glob("scaler_*.pkl"))[0]
    with open(scaler_file, "rb") as f:
        artifacts["scaler"] = pickle.load(f)

    # Load encoders
    encoders_file = list(model_path.glob("encoders_*.pkl"))[0]
    with open(encoders_file, "rb") as f:
        artifacts["encoders"] = pickle.load(f)

    # Load feature names
    features_file = list(model_path.glob("feature_names_*.json"))[0]
    with open(features_file, "r") as f:
        artifacts["feature_config"] = json.load(f)

    # Load metadata
    metadata_file = list(model_path.glob("metadata_*.json"))[0]
    with open(metadata_file, "r") as f:
        artifacts["metadata"] = json.load(f)

    print(f"✓ Model artifacts loaded from: {model_path}")

    return artifacts


# ============================================================================
# INFERENCE
# ============================================================================


def predict_churn(
    customer_data: Union[pd.DataFrame, Dict], artifacts: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Make churn predictions on new customer data.

    Args:
        customer_data: New customer data (DataFrame or dict)
        artifacts: Loaded model artifacts

    Returns:
        Dictionary with predictions, probabilities, and risk categories
    """
    # Convert dict to DataFrame if needed
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])

    df = customer_data.copy()

    # Apply same preprocessing pipeline
    # 1. Binary encoding
    binary_mapping = {"No": 0, "Yes": 1, False: 0, True: 1}
    for col in ["International plan", "Voice mail plan"]:
        if col in df.columns:
            df[col] = df[col].map(binary_mapping)

    # 2. One-hot encoding
    encoders = artifacts["encoders"]
    for col, encoder_key in [
        ("State", "state_encoder"),
        ("Area code", "area_code_encoder"),
    ]:
        if col in df.columns and encoder_key in encoders:
            encoder = encoders[encoder_key]
            encoded = encoder.transform(df[[col]])
            feature_names = encoder.get_feature_names_out([col])
            df_encoded = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            df = df.drop(col, axis=1)
            df = pd.concat([df, df_encoded], axis=1)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Remove correlated features
    correlated = [
        "Total day minutes",
        "Total eve minutes",
        "Total night minutes",
        "Total intl minutes",
        "Voice mail plan",
    ]
    df = df.drop(columns=[c for c in correlated if c in df.columns], errors="ignore")

    # 5. Select features (ensure same order as training)
    feature_names = artifacts["feature_config"]["feature_names"]
    df_selected = df[feature_names]

    # 6. Scale
    scaler = artifacts["scaler"]
    df_scaled = scaler.transform(df_selected)

    # 7. Predict
    model = artifacts["model"]
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    # 8. Risk categories
    risk_categories = []
    for prob in probabilities:
        if prob < 0.3:
            risk_categories.append("Low Risk")
        elif prob < 0.7:
            risk_categories.append("Medium Risk")
        else:
            risk_categories.append("High Risk")

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "risk_categories": risk_categories,
        "churn_labels": ["No Churn" if p == 0 else "Churn" for p in predictions],
    }


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================


def prepare_data(
    train_path: str,
    test_path: str,
    target_col: str,
    binary_cols: Optional[List[str]] = None,
    onehot_cols: Optional[List[str]] = None,
    outlier_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], List[str]]:
    """
    Complete data preparation pipeline.

    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        target_col: Target column name
        binary_cols: Columns for binary encoding
        onehot_cols: Columns for one-hot encoding
        outlier_cols: Columns for outlier detection

    Returns:
        Tuple of (X_train_scaled, y_train, X_test_scaled, y_test, artifacts, feature_names)
    """
    print("\n" + "=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    train_df, test_df = load_data(train_path, test_path)
    validate_data_quality(train_df, "Training")
    validate_data_quality(test_df, "Test")

    # 2. Encode categorical features
    train_enc, test_enc, encoders = encode_categorical_features(
        train_df, test_df, binary_cols=binary_cols, onehot_cols=onehot_cols
    )

    # 3. Detect normality and remove outliers
    if outlier_cols is None:
        outlier_cols = [
            "Account length",
            "Total day minutes",
            "Total day calls",
            "Total day charge",
            "Total eve minutes",
            "Total eve calls",
            "Total eve charge",
            "Total night minutes",
            "Total night calls",
            "Total night charge",
            "Total intl minutes",
            "Total intl calls",
            "Total intl charge",
        ]
    normality_results = detect_normality(train_enc, outlier_cols)
    train_clean, _ = remove_outliers_zscore(
        train_enc, normality_results["normal_columns"]
    )
    train_clean, _ = remove_outliers_iqr(
        train_clean, normality_results["non_normal_columns"]
    )

    # 4. Feature engineering
    train_eng = engineer_features(train_clean)
    test_eng = engineer_features(test_enc)

    # 5. Feature selection
    train_sel, test_sel, _ = select_features_correlation(train_eng, test_eng)

    # 6. Prepare for modeling
    X_train, y_train, X_test, y_test = prepare_data_for_modeling(
        train_sel, test_sel, target_col
    )

    # 7. Balance dataset
    X_balanced, y_balanced = balance_dataset(X_train, y_train)

    # 8. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_balanced, X_test)

    # Store feature names and artifacts
    feature_names = X_train.columns.tolist()
    artifacts = {"scaler": scaler, "encoders": encoders, "feature_names": feature_names}

    print("\n✓ Data preparation complete")

    return X_train_scaled, y_balanced, X_test_scaled, y_test, artifacts, feature_names


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CHURN PREDICTION ML PIPELINE")
    print("=" * 80)

    # Prepare data
    X_train, y_train, X_test, y_test, artifacts, feature_names = prepare_data()

    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)

    # Train models
    models = {
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        logloss = log_loss(y_test, y_proba)

        results[name] = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "log_loss": logloss,
            "model": model,
        }

        print(f"\n{name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        print(f"  Log Loss:  {logloss:.4f}")

    # Find best model
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    best_model_name = max(results, key=lambda x: results[x]["roc_auc"])
    best_model = results[best_model_name]["model"]

    print(f"\nBest Model: {best_model_name}")
    print(f"  ROC AUC: {results[best_model_name]['roc_auc']:.4f}")

    # Save best model
    model_dir = Path("models/v1.0")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"best_model_v1.0.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    # Save artifacts
    artifacts_path = model_dir / "artifacts_v1.0.pkl"
    with open(artifacts_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Artifacts saved to: {artifacts_path}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
