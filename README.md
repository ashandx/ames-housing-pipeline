# ames-housing-pipeline

A `ColumnTransformer` + `Pipeline` on the Ames Housing dataset, comparing Ridge Regression against a plain LinearRegression baseline under 5-fold cross-validated RMSE, rather than a single train/test split.

10 numeric and 6 categorical features are selected from the 81 available columns. Numeric features are median-imputed and scaled; categorical features are most-frequent-imputed and one-hot encoded. Both branches are fit fresh inside each cross-validation fold, so there's no leakage from the imputer or scaler statistics into the held-out fold.

A second pass adds 8 engineered features (3 ratios, 2 interactions, 2 date-decomposition, 1 binned) plus `log1p` on `SalePrice` and the most-skewed numeric inputs (`GrLivArea`, `LotArea`, `TotalBsmtSF`). This dropped 5-fold CV RMSE from 34,090 to 32,181 for LinearRegression (-5.6%) and from 33,909 to 32,160 for Ridge (-5.2%).

A third pass tunes `Ridge`'s `alpha` via `GridSearchCV` (32,149, a negligible improvement over the default) and introduces `RandomForestRegressor` tuned via `RandomizedSearchCV` (31,404, -2.4% vs the Ridge baseline, with roughly a third of Ridge's fold-to-fold variance). Random Forest is the final model.

## Setup

```bash
conda create -n ames-housing-pipeline python=3.11
conda activate ames-housing-pipeline
pip install pandas numpy scikit-learn scipy matplotlib jupyter
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
- Random Forest's tuned pipeline reuses the linear-model preprocessor (`StandardScaler`, `log1p` skew transform) unchanged, even though trees don't need scaling — a deliberate simplification to keep every model comparable on the same feature matrix, not an oversight.
- No real Kaggle `test.csv` is available locally, so the "submission" step uses the existing held-out split as a format stand-in — it can't be scored on the actual leaderboard.
- The learning curve (Section 13) shows validation RMSE still improving at the largest available training size — the dataset is capped at 1,460 rows, so "more data would help" isn't actionable within this notebook.
