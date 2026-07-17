# ames-housing-pipeline

A `ColumnTransformer` + `Pipeline` on the Ames Housing dataset, comparing Ridge Regression against a plain LinearRegression baseline under 5-fold cross-validated RMSE, rather than a single train/test split.

10 numeric and 6 categorical features are selected from the 81 available columns. Numeric features are median-imputed and scaled; categorical features are most-frequent-imputed and one-hot encoded. Both branches are fit fresh inside each cross-validation fold, so there's no leakage from the imputer or scaler statistics into the held-out fold.

## Setup

```bash
conda create -n ames-housing-pipeline python=3.11
conda activate ames-housing-pipeline
pip install pandas numpy scikit-learn jupyter
```

## Run

```bash
jupyter notebook notebooks/ames_housing_pipeline.ipynb
```

Run all cells top to bottom.

## Data

`data/ames_train.csv` is Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) `train.csv` (1460 rows, 81 columns), saved locally under a different name.

## Known limitations

- Feature selection (10 numeric + 6 categorical) was chosen for a first pass, not tuned or exhaustively justified.
- `Ridge` uses the default `alpha=1.0` — no hyperparameter search yet.
- The held-out test split from the original train/test split is untouched by cross-validation and not yet used for a final evaluation.
