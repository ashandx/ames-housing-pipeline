# ames-housing-pipeline

A `ColumnTransformer` + `Pipeline` on the Ames Housing dataset, comparing Ridge Regression against a plain LinearRegression baseline under 5-fold cross-validated RMSE, rather than a single train/test split.

10 numeric and 6 categorical features are selected from the 81 available columns. Numeric features are median-imputed and scaled; categorical features are most-frequent-imputed and one-hot encoded. Both branches are fit fresh inside each cross-validation fold, so there's no leakage from the imputer or scaler statistics into the held-out fold.

A second pass adds 8 engineered features (3 ratios, 2 interactions, 2 date-decomposition, 1 binned) plus `log1p` on `SalePrice` and the most-skewed numeric inputs (`GrLivArea`, `LotArea`, `TotalBsmtSF`). This dropped 5-fold CV RMSE from 34,090 to 32,181 for LinearRegression (-5.6%) and from 33,909 to 32,160 for Ridge (-5.2%).

A third pass tunes `Ridge`'s `alpha` via `GridSearchCV` (32,149, a negligible improvement over the default) and introduces `RandomForestRegressor` tuned via `RandomizedSearchCV` (31,404, -2.4% vs the Ridge baseline, with roughly a third of Ridge's fold-to-fold variance). Random Forest is the final model.

A fourth pass (Milestone 3) evaluates encoding and model-family choices under a different lens: `StratifiedKFold` on `qcut`-binned `log1p(SalePrice)` deciles, scored in log space rather than dollars, so results aren't directly comparable to the dollar-scale figures above. Against a re-established baseline (0.1533 mean log-RMSE) and engineered-feature version (0.1498), ordinal-encoding all 10 shared quality-scale columns (`ExterQual`, `KitchenQual`, plus 8 more added for this test) gave the largest single improvement (0.1451), though it's confounded with adding those 8 previously-unused columns. Grouping rare `Neighborhood` categories improved mean RMSE only marginally (0.1490) and, on the specific thing it targeted, slightly *worsened* fold-to-fold std (0.0122 vs 0.0120) — Random Forest doesn't destabilize on sparse categories the way linear models do, so grouping traded away information without buying stability. Swapping in an untuned `HistGradientBoostingRegressor` (0.1461) beat the tuned Random Forest baseline anyway. Combining ordinal encoding with neighborhood grouping (0.1450) barely improved on ordinal encoding alone and had the worst std of all six configurations. This pass doesn't change `FINAL_MODEL_PIPE` or the submission file — it's a diagnostic comparison, not a resubmission.

A fifth pass (Section 16) produces a real Kaggle submission, and deliberately doesn't use `FINAL_MODEL_PIPE`. Kaggle's leaderboard scores on log-space RMSE, the metric Milestone 3 evaluated under, not the dollar-scale metric `FINAL_MODEL_PIPE` was selected on. The engineered-features config closest to `FINAL_MODEL_PIPE`'s approach scores 0.1498 mean log-RMSE; ordinal encoding (Milestone 3 Row 3) beats that at 0.1451 and is used instead — not the marginally-lower-mean "best combination" row, since that trades a 0.0001 mean improvement for meaningfully worse std. The chosen pipeline is refit on 100% of `ames_train.csv` and predicts on the real `ames_test.csv`, written to `data/kaggle_submission.csv`.

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

Sections 10 and 14 still produce a format-stand-in CSV predicting on the existing held-out split rather than `ames_test.csv` — that output (`data/submission_stub.csv`) is gitignored, not committed, and is left as-is. Section 16 produces the real submission: `data/kaggle_submission.csv`, fit on 100% of `ames_train.csv` and predicting on `ames_test.csv`, committed to the repo.

## Known limitations

- Feature selection (10 numeric + 6 categorical, plus 8 engineered on top) was chosen for a first pass, not tuned or exhaustively justified.
- Random Forest's tuned pipeline reuses the linear-model preprocessor (`StandardScaler`, `log1p` skew transform) unchanged, even though trees don't need scaling — a deliberate simplification to keep every model comparable on the same feature matrix, not an oversight.
- The learning curve (Section 13) shows validation RMSE still improving at the largest available training size — the dataset is capped at 1,460 rows, so "more data would help" isn't actionable within this notebook.
- Milestone 3's ordinal-encoding row (Section 15) confounds two changes at once (adding 8 previously-unused quality columns and switching their encoding), so the ~3.1% mean improvement can't be cleanly attributed to ordinal encoding alone.
- `data/kaggle_submission.csv`'s pipeline imputes `KitchenQual`'s 1 test-set `NaN` as `'None'` (meaningful-NA, same as the other 9 quality columns) rather than as a genuinely unrecorded value — affects 1 of 1459 rows.
- `FINAL_MODEL_PIPE` (Section 12) and the real Kaggle submission (Section 16) are different configurations, selected under different metrics for good reason (see Section 16's opening cell) — `FINAL_MODEL_PIPE` was never updated to match, since Sections 7-14 are about dollar-scale model comparison, not the leaderboard submission.
