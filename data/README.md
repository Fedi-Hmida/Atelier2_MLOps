# Data Directory

This directory contains all datasets used by the ML pipeline.

## Contents

- `churn-bigml-80.csv` - Training dataset (2,668 records, 80% split)
- `churn-bigml-20.csv` - Test dataset (669 records, 20% split)

## Dataset Information

**Source:** BigML Telecom Churn Dataset  
**Domain:** Telecommunications  
**Task:** Binary Classification (Churn Prediction)  
**Target Variable:** Churn (True/False)

### Features (20 total)
- **Categorical:** State, Area code, International plan, Voice mail plan
- **Numeric:** Account length, call minutes/calls/charges (day/eve/night/intl), customer service calls

## Adding New Datasets

To use different datasets with this pipeline:

1. Place your CSV files in this directory
2. Update the configuration file (`config/config.yaml`) with the new paths:
   ```yaml
   data:
     train_path: "data/your_train_data.csv"
     test_path: "data/your_test_data.csv"
   ```

## Data Privacy

- Never commit sensitive customer data to version control
- Add data files to `.gitignore` if they contain PII
- Use data encryption for production datasets
