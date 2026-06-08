# Machine learning-assisted multiplexing using grayscale-programmable "all-sensitivity-in-one" SERS platform

This repository contains source code and spectral datasets for machine learning-assisted multiplexed SERS quantification of three analytes of uric acid, adenine, and cocaine.

The key feature of this platform is that multiple grayscale-programmable SERS substrates provide different Raman responses to the same mixture. These substrate-dependent responses are used together with spectral features for multiplexing.

## Files

```text
dataset-raw.csv              Raw SERS spectra
dataset-ALS-processed.csv    ALS baseline-corrected spectra

PLSR+MLP.py                  PLSR + MLP
PLSR+SVR.py                  PLSR + SVR
PLSR+KRR.py                  PLSR + kernel ridge regression
PCA+MLP.py                   PCA + MLP
PCA+SVR.py                   PCA + SVR
```

## Adjustable Parameters

The main settings are placed near the top of each script.

```python
# 1. Spectral data source: "raw", "als", or "both"
DATA_MODE = "raw"

# 2. Repeated spectra: "individual" or "average"
SPECTRUM_MODE = "individual"
INDIVIDUAL_SPECTRAL_NUMBERS = (1, 2, 3)
AVERAGE_SPECTRAL_GROUPS = (tuple(range(1, 16)),)

# 3. Substrate selection
SELECTED_SUBSTRATES = (4, 5, 6, 7)
```

Use `DATA_MODE = "both"` to concatenate raw and ALS-corrected spectra. Use `SELECTED_SUBSTRATES = None` to include all substrates.

Each script also contains a `PARAM_GRID` block for model-specific hyperparameter tuning.

## Data Splitting

To reduce data leakage, train/test splitting is group-aware. All spectra from the same mixture-substrate condition are assigned either to training or to testing, not both.

```python
condition_group = mixture + substrate number
```

The scripts use `GroupShuffleSplit` for the final train/test split and `GroupKFold` for cross-validation.

## Output

Each script saves prediction and metric CSV files. The metrics file reports R2, RMSE, MAE, best cross-validation R2, and the selected hyperparameters.
