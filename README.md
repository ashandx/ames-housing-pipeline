# ames-housing-pipeline

A `ColumnTransformer` + `Pipeline` on the Ames Housing dataset, comparing Ridge Regression against a plain LinearRegression baseline under 5-fold cross-validated RMSE, rather than a single train/test split.

10 numeric and 6 categorical features are selected from the 81 available columns. Numeric features are median-imputed and scaled; categorical features are most-frequent-imputed and one-hot encoded. Both branches are fit fresh inside each cross-validation fold, so there's no leakage from the imputer or scaler statistics into the held-out fold.

A second pass adds 8 engineered features (3 ratios, 2 interactions, 2 date-decomposition, 1 binned) plus `log1p` on `SalePrice` and the most-skewed numeric inputs (`GrLivArea`, `LotArea`, `TotalBsmtSF`). This dropped 5-fold CV RMSE from 34,090 to 32,181 for LinearRegression (-5.6%) and from 33,909 to 32,160 for Ridge (-5.2%).

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

Kaggle's actual `test.csv` (the unlabeled competition holdout set) isn't included in this repo. The notebook's final section produces a submission-format CSV as a stand-in, predicting on the existing held-out split rather than the real Kaggle test set — see the notebook's Section 10 for why. That output (`data/submission_stub.csv`) is gitignored, not committed.

## Known limitations

- Feature selection (10 numeric + 6 categorical, plus 8 engineered on top) was chosen for a first pass, not tuned or exhaustively justified.
- `Ridge` uses the default `alpha=1.0` — no hyperparameter search yet.
- No real Kaggle `test.csv` is available locally, so the "submission" step uses the existing held-out split as a format stand-in — it can't be scored on the actual leaderboard.
