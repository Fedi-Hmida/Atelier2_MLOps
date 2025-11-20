

import argparse
import sys
from pathlib import Path
import numpy as np

# Import all pipeline functions
from model_pipeline import (
    # Data preparation
    prepare_data,
    
    # Model training
    train_model,
    optimize_hyperparameters,
    
    # Evaluation
    evaluate_model,
    cross_validate_model,
    extract_feature_importance,
    
    # Persistence
    save_model,
    load_model,
    
    # Inference
    predict_churn
)


def run_full_pipeline(
    train_path: str = 'churn-bigml-80.csv',
    test_path: str = 'churn-bigml-20.csv',
    optimize: bool = True,
    n_trials: int = 50,
    model_type: str = 'xgboost'
):
    """
    Execute the complete ML pipeline.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        optimize: Whether to run hyperparameter optimization
        n_trials: Number of optimization trials
        model_type: Model type ('xgboost', 'random_forest')
    """
    print("\n" + "="*70)
    print(" CUSTOMER CHURN PREDICTION - ML PIPELINE ".center(70))
    print("="*70)
    
    # ========================================================================
    # STEP 1: DATA PREPARATION
    # ========================================================================
    print("\n[STEP 1/5] DATA PREPARATION")
    print("-" * 70)
    
    X_train, y_train, X_test, y_test, artifacts, feature_names = prepare_data(
        train_path=train_path,
        test_path=test_path
    )
    
    # ========================================================================
    # STEP 2: MODEL TRAINING
    # ========================================================================
    print("\n[STEP 2/5] MODEL TRAINING")
    print("-" * 70)
    
    if optimize:
        print(f"\nRunning hyperparameter optimization with {n_trials} trials...")
        model, best_params = optimize_hyperparameters(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_trials=n_trials
        )
        print(f"\nBest hyperparameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    else:
        # Train with default parameters
        if model_type == 'xgboost':
            params = {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        else:
            params = {
                'n_estimators': 300,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2
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
    cv_results = cross_validate_model(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Feature importance
    importance_df = extract_feature_importance(model, feature_names, top_n=15)
    
    # ========================================================================
    # STEP 4: MODEL PERSISTENCE
    # ========================================================================
    print("\n[STEP 4/5] MODEL PERSISTENCE")
    print("-" * 70)
    
    metadata = {
        'model_type': model_type,
        'hyperparameters': best_params,
        'test_metrics': metrics,
        'cv_results': cv_results,
        'n_features': len(feature_names),
        'training_samples': len(y_train),
        'test_samples': len(y_test)
    }
    
    saved_paths = save_model(
        model=model,
        scaler=artifacts['scaler'],
        encoders=artifacts['encoders'],
        feature_names=feature_names,
        metadata=metadata,
        output_dir='./models',
        model_name='churn_model',
        version='v1.0'
    )
    
    # ========================================================================
    # STEP 5: INFERENCE DEMO
    # ========================================================================
    print("\n[STEP 5/5] INFERENCE DEMONSTRATION")
    print("-" * 70)
    
    # Load saved model
    loaded_artifacts = load_model(model_dir='./models', version='v1.0')
    
    # Create sample customer data
    sample_customer = {
        'State': 'NY',
        'Account length': 128,
        'Area code': 415,
        'International plan': 'No',
        'Voice mail plan': 'Yes',
        'Number vmail messages': 25,
        'Total day minutes': 265.1,
        'Total day calls': 110,
        'Total day charge': 45.07,
        'Total eve minutes': 197.4,
        'Total eve calls': 99,
        'Total eve charge': 16.78,
        'Total night minutes': 244.7,
        'Total night calls': 91,
        'Total night charge': 11.01,
        'Total intl minutes': 10.0,
        'Total intl calls': 3,
        'Total intl charge': 2.70,
        'Customer service calls': 1
    }
    
    print("\nSample Customer Data:")
    print(f"  Account Length: {sample_customer['Account length']} days")
    print(f"  Total Charge: ${sample_customer['Total day charge'] + sample_customer['Total eve charge'] + sample_customer['Total night charge'] + sample_customer['Total intl charge']:.2f}")
    print(f"  Customer Service Calls: {sample_customer['Customer service calls']}")
    
    # Make prediction
    results = predict_churn(sample_customer, loaded_artifacts)
    
    print("\n✓ Prediction Results:")
    print(f"  Churn Prediction: {results['churn_labels'][0]}")
    print(f"  Churn Probability: {results['probabilities'][0]:.2%}")
    print(f"  Risk Category: {results['risk_categories'][0]}")
    
    # ========================================================================
    # PIPELINE SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print(" PIPELINE EXECUTION COMPLETE ".center(70))
    print("="*70)
    print(f"\n✓ Model Type: {model_type.upper()}")
    print(f"✓ Test ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"✓ Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ CV ROC AUC: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    print(f"✓ Model saved to: ./models/v1.0/")
    print("\n" + "="*70 + "\n")


def run_prepare_only(train_path: str, test_path: str):
    """Run data preparation step only."""
    print("\n[DATA PREPARATION ONLY]")
    X_train, y_train, X_test, y_test, artifacts, feature_names = prepare_data(
        train_path=train_path,
        test_path=test_path
    )
    print(f"\n✓ Data prepared: {len(feature_names)} features, {len(y_train)} training samples")


def run_train_only(model_type: str, optimize: bool, n_trials: int):
    """Run training step only (requires preprocessed data)."""
    print("\n[MODEL TRAINING ONLY]")
    print("Note: This requires pre-saved training data. Run full pipeline first.")
    
    # This is a simplified version - in production, you'd load saved preprocessed data
    print(f"Training {model_type} model...")
    print("Use the full pipeline for end-to-end execution.")


def run_evaluate_only():
    """Evaluate a saved model."""
    print("\n[MODEL EVALUATION ONLY]")
    
    try:
        artifacts = load_model(model_dir='./models', version='v1.0')
        print("✓ Model loaded successfully")
        print(f"\nModel Metadata:")
        metadata = artifacts['metadata']
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
        artifacts = load_model(model_dir='./models', version='v1.0')
        
        # High-risk customer example
        high_risk_customer = {
            'State': 'TX',
            'Account length': 45,
            'Area code': 408,
            'International plan': 'Yes',
            'Voice mail plan': 'No',
            'Number vmail messages': 0,
            'Total day minutes': 320.5,
            'Total day calls': 120,
            'Total day charge': 54.49,
            'Total eve minutes': 180.2,
            'Total eve calls': 95,
            'Total eve charge': 15.32,
            'Total night minutes': 200.5,
            'Total night calls': 85,
            'Total night charge': 9.02,
            'Total intl minutes': 15.5,
            'Total intl calls': 5,
            'Total intl charge': 4.19,
            'Customer service calls': 5
        }
        
        results = predict_churn(high_risk_customer, artifacts)
        
        print("\nHigh-Risk Customer Profile:")
        print(f"  Customer Service Calls: {high_risk_customer['Customer service calls']}")
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
        description='Customer Churn Prediction ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with optimization
  python main.py
  
  # Run full pipeline without optimization (faster)
  python main.py --no-optimize
  
  # Use Random Forest instead of XGBoost
  python main.py --model random_forest
  
  # Only prepare data
  python main.py --prepare
  
  # Only evaluate saved model
  python main.py --evaluate
  
  # Run inference demo
  python main.py --predict
        """
    )
    
    # Pipeline control
    parser.add_argument(
        '--prepare',
        action='store_true',
        help='Only run data preparation'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Only run model training'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Only evaluate saved model'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Run inference demonstration'
    )
    
    # Data paths
    parser.add_argument(
        '--train-data',
        type=str,
        default='churn-bigml-80.csv',
        help='Path to training data (default: churn-bigml-80.csv)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default='churn-bigml-20.csv',
        help='Path to test data (default: churn-bigml-20.csv)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        choices=['xgboost', 'random_forest', 'gradient_boosting'],
        default='xgboost',
        help='Model type to train (default: xgboost)'
    )
    parser.add_argument(
        '--optimize',
        dest='optimize',
        action='store_true',
        default=True,
        help='Run hyperparameter optimization (default: True)'
    )
    parser.add_argument(
        '--no-optimize',
        dest='optimize',
        action='store_false',
        help='Skip hyperparameter optimization'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Validate data files exist if running full pipeline or prepare
    if not (args.train or args.evaluate or args.predict):
        if not Path(args.train_data).exists():
            print(f"✗ Error: Training data not found: {args.train_data}")
            sys.exit(1)
        if not Path(args.test_data).exists():
            print(f"✗ Error: Test data not found: {args.test_data}")
            sys.exit(1)
    
    # Execute requested operation
    try:
        if args.prepare:
            run_prepare_only(args.train_data, args.test_data)
        elif args.train:
            run_train_only(args.model, args.optimize, args.trials)
        elif args.evaluate:
            run_evaluate_only()
        elif args.predict:
            run_predict_demo()
        else:
            # Run full pipeline
            run_full_pipeline(
                train_path=args.train_data,
                test_path=args.test_data,
                optimize=args.optimize,
                n_trials=args.trials,
                model_type=args.model
            )
    
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
