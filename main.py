import argparse
import sys
from pathlib import Path

import numpy as np

# Import configuration loader
from config_loader import Config, ConfigurationError, load_config

# Import all pipeline functions
from model_pipeline import (  # Data preparation; Model training; Evaluation; Persistence; Inference
    cross_validate_model, evaluate_model, extract_feature_importance,
    load_model, optimize_hyperparameters, predict_churn, prepare_data,
    save_model, train_model)


def run_full_pipeline(config: Config):
    """
    Execute complete ML pipeline with configuration.
    
    Args:
        config: Configuration object with all pipeline settings
    """
    
    # Extract configuration
    data_config = config.get_data_config()
    model_config = config.get_model_config()
    preprocessing_config = config.get_preprocessing_config()
    persistence_config = config.get_persistence_config()
    evaluation_config = config.get_evaluation_config()
    
    train_path = data_config['train_path']
    test_path = data_config['test_path']
    target_col = data_config['target_column']
    model_type = model_config['type']
    optimize = model_config['optimization']['enabled']
    n_trials = model_config['optimization']['n_trials']

    print("\n" + "=" * 70)
    print(" CUSTOMER CHURN PREDICTION - ML PIPELINE ".center(70))
    print("=" * 70)
    
    # Display configuration
    config.display()

    # ========================================================================
    # STEP 1: DATA PREPARATION
    # ========================================================================
    print("\n[STEP 1/5] DATA PREPARATION")
    print("-" * 70)

    X_train, y_train, X_test, y_test, artifacts, feature_names = prepare_data(
        train_path=train_path,
        test_path=test_path,
        target_col=target_col,
        binary_cols=data_config.get('encoding', {}).get('binary_columns'),
        onehot_cols=data_config.get('encoding', {}).get('onehot_columns'),
        outlier_cols=data_config.get('outlier_columns'),
    )

    # ========================================================================
    # STEP 2: MODEL TRAINING
    # ========================================================================
    print("-" * 70)

    if optimize:
        print(f"\nRunning hyperparameter optimization with {n_trials} trials...")
        model, best_params = optimize_hyperparameters(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_trials=n_trials,
        )
        print(f"\nBest hyperparameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    else:
        # Train with default parameters from config
        params = model_config.get('default_params', {}).get(model_type, {})
        
        if not params:
            # Fallback defaults
            if model_type == "xgboost":
                params = {
                    "n_estimators": 200,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                }
            else:
                params = {
                    "n_estimators": 300,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                }

        model = train_model(X_train, y_train, model_type=model_type, **params)
        best_params = params

    # ========================================================================
    # STEP 3: MODEL EVALUATION
    # ========================================================================
    print("\n[STEP 3/5] MODEL EVALUATION")
    print("-" * 70)

    # Comprehensive evaluation
    metrics = evaluate_model(model, X_test, y_test, model_name=model_type)

    # Cross-validation
    cv_folds = evaluation_config.get('cv_folds', 5)
    cv_scoring = evaluation_config.get('cv_scoring', 'roc_auc')
    cv_results = cross_validate_model(
        model, X_train, y_train, cv=cv_folds, scoring=cv_scoring
    )

    # Feature importance
    top_n = evaluation_config.get('top_features', 15)
    importance_df = extract_feature_importance(model, feature_names, top_n=top_n)

    # ========================================================================
    # STEP 4: MODEL PERSISTENCE
    # ========================================================================
    print("\n[STEP 4/5] MODEL PERSISTENCE")
    print("-" * 70)

    metadata = {
        "model_type": model_type,
        "hyperparameters": best_params,
        "test_metrics": metrics,
        "cv_results": cv_results,
        "n_features": len(feature_names),
        "training_samples": len(y_train),
        "test_samples": len(y_test),
        "configuration": config.to_dict(),
    }
    
    models_dir = persistence_config.get('models_dir', './models')
    version_format = persistence_config.get('version_format', 'v{version}')
    version = version_format.format(version=config.get('project.version', '1.0'))

    saved_paths = save_model(
        model=model,
        scaler=artifacts["scaler"],
        encoders=artifacts["encoders"],
        feature_names=feature_names,
        metadata=metadata,
        output_dir=models_dir,
        model_name="churn_model",
        version=version,
    )

    # ========================================================================
    # STEP 5: INFERENCE DEMO
    # ========================================================================
    print("\n[STEP 5/5] INFERENCE DEMONSTRATION")
    print("-" * 70)

    # Load saved model
    loaded_artifacts = load_model(model_dir="./models", version="v1.0")

    # Create sample customer data
    sample_customer = {
        "State": "NY",
        "Account length": 128,
        "Area code": 415,
        "International plan": "No",
        "Voice mail plan": "Yes",
        "Number vmail messages": 25,
        "Total day minutes": 265.1,
        "Total day calls": 110,
        "Total day charge": 45.07,
        "Total eve minutes": 197.4,
        "Total eve calls": 99,
        "Total eve charge": 16.78,
        "Total night minutes": 244.7,
        "Total night calls": 91,
        "Total night charge": 11.01,
        "Total intl minutes": 10.0,
        "Total intl calls": 3,
        "Total intl charge": 2.70,
        "Customer service calls": 1,
    }

    print("\nSample Customer Data:")
    print(f"  Account Length: {sample_customer['Account length']} days")
    print(
        f"  Total Charge: ${sample_customer['Total day charge'] + sample_customer['Total eve charge'] + sample_customer['Total night charge'] + sample_customer['Total intl charge']:.2f}"
    )
    print(f"  Customer Service Calls: {sample_customer['Customer service calls']}")

    # Make prediction
    results = predict_churn(sample_customer, loaded_artifacts)

    print("\n✓ Prediction Results:")
    print(f"  Churn Prediction: {results['churn_labels'][0]}")
    print(f"  Churn Probability: {results['probabilities'][0]:.2%}")
    print(f"  Risk Category: {results['risk_categories'][0]}")

    # ========================================================================


def run_prepare_only(config: Config):
    """Run data preparation step only."""
    print("\n[DATA PREPARATION ONLY]")
    
    data_config = config.get_data_config()
    
    X_train, y_train, X_test, y_test, artifacts, feature_names = prepare_data(
        train_path=data_config['train_path'],
        test_path=data_config['test_path'],
        target_col=data_config['target_column'],
        binary_cols=data_config.get('encoding', {}).get('binary_columns'),
        onehot_cols=data_config.get('encoding', {}).get('onehot_columns'),
        outlier_cols=data_config.get('outlier_columns'),
    )
    print(
        f"\n✓ Data prepared: {len(feature_names)} features, {len(y_train)} training samples"
    )


def run_train_only(config: Config):
    """Run training step only (requires preprocessed data)."""
    print("\n[MODEL TRAINING ONLY]")
    print("Note: This requires pre-saved training data. Run full pipeline first.")

    model_type = config.get('model.type')
    print(f"Training {model_type} model...")
    print("Use the full pipeline for end-to-end execution.")


def run_evaluate_only():
    """Evaluate a saved model."""
    print("\n[MODEL EVALUATION ONLY]")

    try:
        artifacts = load_model(model_dir="./models", version="v1.0")
        print("✓ Model loaded successfully")
        print(f"\nModel Metadata:")
        metadata = artifacts["metadata"]
        print(f"  Model Type: {metadata['model_type']}")
        print(f"  Test ROC AUC: {metadata['test_metrics']['roc_auc']:.4f}")
        print(f"  Test Accuracy: {metadata['test_metrics']['accuracy']:.4f}")
        print(f"  Trained on: {metadata.get('saved_at', 'N/A')}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Run the full pipeline first to create a model.")


def run_predict_demo():
    """Run inference demonstration."""
    print("\n[INFERENCE DEMONSTRATION]")

    try:
        artifacts = load_model(model_dir="./models", version="v1.0")

        # High-risk customer example
        high_risk_customer = {
            "State": "TX",
            "Account length": 45,
            "Area code": 408,
            "International plan": "Yes",
            "Voice mail plan": "No",
            "Number vmail messages": 0,
            "Total day minutes": 320.5,
            "Total day calls": 120,
            "Total day charge": 54.49,
            "Total eve minutes": 180.2,
            "Total eve calls": 95,
            "Total eve charge": 15.32,
            "Total night minutes": 200.5,
            "Total night calls": 85,
            "Total night charge": 9.02,
            "Total intl minutes": 15.5,
            "Total intl calls": 5,
            "Total intl charge": 4.19,
            "Customer service calls": 5,
        }

        results = predict_churn(high_risk_customer, artifacts)

        print("\nHigh-Risk Customer Profile:")
        print(
            f"  Customer Service Calls: {high_risk_customer['Customer service calls']}"
        )
        print(f"  International Plan: {high_risk_customer['International plan']}")

        print("\n✓ Prediction:")
        print(f"  Churn: {results['churn_labels'][0]}")
        print(f"  Probability: {results['probabilities'][0]:.2%}")
        print(f"  Risk: {results['risk_categories'][0]}")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("Run the full pipeline first to create a model.")


def main():
    """Main execution function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python main.py
  
  # Use specific configuration file
  python main.py --config config/config.prod.yaml
  
  # Use development configuration
  python main.py --config config/config.dev.yaml
  
  # Override specific settings
  python main.py --config config/config.yaml --model random_forest --no-optimize
  
  # Only prepare data
  python main.py --prepare
  
  # Run inference
  python main.py --predict
        """,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml)",
    )

    # Pipeline control
    parser.add_argument(
        "--prepare", action="store_true", help="Only run data preparation"
    )
    parser.add_argument("--train", action="store_true", help="Only run model training")
    parser.add_argument(
        "--evaluate", action="store_true", help="Only evaluate saved model"
    )
    parser.add_argument(
        "--predict", action="store_true", help="Run inference demonstration"
    )

    # Configuration overrides (optional)
    parser.add_argument(
        "--train-data",
        type=str,
        help="Override training data path from config",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Override test data path from config",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "random_forest", "gradient_boosting"],
        help="Override model type from config",
    )
    parser.add_argument(
        "--optimize",
        dest="optimize",
        action="store_true",
        help="Enable hyperparameter optimization",
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help="Disable hyperparameter optimization",
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="Override number of optimization trials from config",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        print(f"\nLoading configuration from: {args.config}")
        config = load_config(args.config)
    except ConfigurationError as e:
        print(f"\n✗ Configuration Error: {e}")
        print("\nPlease ensure you have a valid configuration file.")
        print("You can create one using the provided templates in config/")
        sys.exit(1)

    # Apply CLI overrides to configuration
    if args.train_data:
        config.data['data']['train_path'] = args.train_data
        print(f"Override: Using training data from {args.train_data}")
    
    if args.test_data:
        config.data['data']['test_path'] = args.test_data
        print(f"Override: Using test data from {args.test_data}")
    
    if args.model:
        config.data['model']['type'] = args.model
        print(f"Override: Using model type {args.model}")
    
    if args.optimize is not None:
        config.data['model']['optimization']['enabled'] = args.optimize
        print(f"Override: Optimization {'enabled' if args.optimize else 'disabled'}")
    
    if args.trials:
        config.data['model']['optimization']['n_trials'] = args.trials
        print(f"Override: Using {args.trials} optimization trials")

    # Validate data files exist if running operations that need them
    if not (args.evaluate or args.predict):
        train_path = config.get('data.train_path')
        test_path = config.get('data.test_path')
        
        if not Path(train_path).exists():
            print(f"✗ Error: Training data not found: {train_path}")
            sys.exit(1)
        if not Path(test_path).exists():
            print(f"✗ Error: Test data not found: {test_path}")
            sys.exit(1)

    # Execute requested operation
    try:
        if args.prepare:
            run_prepare_only(config)
        elif args.train:
            run_train_only(config)
        elif args.evaluate:
            run_evaluate_only()
        elif args.predict:
            run_predict_demo()
        else:
            # Run full pipeline
            run_full_pipeline(config)

    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
