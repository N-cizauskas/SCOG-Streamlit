# Streamlit GAN Application

**SCOG: Synthetic Control Generator using CTGAN**

## Description
Interactive Streamlit web application for generating synthetic control data for clinical trials using Conditional Tabular GANs (CTGAN).

## Main Files
- `streamlit_app.py` - Main Streamlit UI application
- `CTGAN_dec_adjustable.py` - Custom CTGAN implementation with adjustable parameters
- `data_loader.py` - Data loading and preprocessing utilities
- `evaluation.py` - Evaluation metrics (MAE, MSE, RMSE, SMD, k-anonymity, PSM)

## Features
- Upload CSV data or use sample datasets
- Automatic column classification (continuous, categorical, binary)
- Configurable model hyperparameters
- Training with early stopping
- Comprehensive evaluation metrics
- Distribution visualizations (histograms, PCA, Love plots)
- Propensity Score Matching
- k-Anonymity privacy assessment
- CSV download of synthetic data

## Sample Data
- `testdata3.csv` - Main test dataset (10,000 rows)
- `testdata4.csv` - Alternative test dataset
- `sample_data.csv` - Sample dataset without missing values
- `sample_data_with_nas.csv` - Sample dataset with missing values

## Quick Start
See `STREAMLIT_QUICK_START.md` and `QUICKSTART.md` for detailed instructions.

## Running the App
```bash
streamlit run streamlit_app.py
```

## Documentation
- GAN structure diagram: `gan.drawio.png`
