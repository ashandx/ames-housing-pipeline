# ames-housing-pipeline

A `ColumnTransformer` + `Pipeline` on the Ames Housing dataset, comparing Ridge Regression against a plain LinearRegression baseline under 5-fold cross-validated RMSE, rather than a single train/test split.

10 numeric and 6 categorical features are selected from the 81 available columns. Numeric features are median-imputed and scaled; categorical features are most-frequent-imputed and one-hot encoded. Both branches are fit fresh inside each cross-validation fold, so there's no leakage from the imputer or scaler statistics into the held-out fold.

A second pass adds 8 engineered features (3 ratios, 2 interactions, 2 date-decomposition, 1 binned) plus `log1p` on `SalePrice` and the most-skewed numeric inputs (`GrLivArea`, `LotArea`, `TotalBsmtSF`). This dropped 5-fold CV RMSE from 34,090 to 32,181 for LinearRegression (-5.6%) and from 33,909 to 32,160 for Ridge (-5.2%).

A third pass tunes `Ridge`'s `alpha` via `GridSearchCV` (32,149, a negligible improvement over the default) and introduces `RandomForestRegressor` tuned via `RandomizedSearchCV` (31,404, -2.4% vs the Ridge baseline, with roughly a third of Ridge's fold-to-fold variance). Random Forest is the final model.

A fourth pass (Milestone 3) evaluates encoding and model-family choices under a different lens: `StratifiedKFold` on `qcut`-binned `log1p(SalePrice)` deciles, scored in log space rather than dollars, so results aren't directly comparable to the dollar-scale figures above. Against a re-established baseline (0.1533 mean log-RMSE) and engineered-feature version (0.1498), ordinal-encoding all 10 shared quality-scale columns (`ExterQual`, `KitchenQual`, plus 8 more added for this test) gave the largest single improvement (0.1451), though it's confounded with adding those 8 previously-unused columns. Grouping rare `Neighborhood` categories improved mean RMSE only marginally (0.1490) and, on the specific thing it targeted, slightly *worsened* fold-to-fold std (0.0122 vs 0.0120) — Random Forest doesn't destabilize on sparse categories the way linear models do, so grouping traded away information without buying stability. Swapping in an untuned `HistGradientBoostingRegressor` (0.1461) beat the tuned Random Forest baseline anyway. Combining ordinal encoding with neighborhood grouping (0.1450) barely improved on ordinal encoding alone and had the worst std of all six configurations. This pass doesn't change `FINAL_MODEL_PIPE` or the submission file — it's a diagnostic comparison, not a resubmission.

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

`data/ames_train.csv` is Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) `train.csv` (1460 rows, 81 columns), saved locally under a different name. `data/ames_test.csv` is the same competition's `test.csv` (1459 rows, 80 columns, no `SalePrice`), saved locally the same way.

The notebook's submission sections still produce a format-stand-in CSV predicting on the existing held-out split rather than `ames_test.csv` — see the notebook's Section 10 for why. That output (`data/submission_stub.csv`) is gitignored, not committed. A real Kaggle submission fit on the full training data and predicting on `ames_test.csv` hasn't been run yet; it's a follow-up, not part of this notebook's current sections.

## Known limitations

- Feature selection (10 numeric + 6 categorical, plus 8 engineered on top) was chosen for a first pass, not tuned or exhaustively justified.
- Random Forest's tuned pipeline reuses the linear-model preprocessor (`StandardScaler`, `log1p` skew transform) unchanged, even though trees don't need scaling — a deliberate simplification to keep every model comparable on the same feature matrix, not an oversight.
- `data/ames_test.csv` (Kaggle's real test set) is now available locally, but the notebook's "submission" sections still predict on the existing held-out split as a format stand-in rather than on it — a real leaderboard submission hasn't been produced yet.
- The learning curve (Section 13) shows validation RMSE still improving at the largest available training size — the dataset is capped at 1,460 rows, so "more data would help" isn't actionable within this notebook.
- Milestone 3's ordinal-encoding row (Section 15) confounds two changes at once (adding 8 previously-unused quality columns and switching their encoding), so the ~3.1% mean improvement can't be cleanly attributed to ordinal encoding alone.
