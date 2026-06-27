# Prague apartment price prediction

End-to-end ML project: scrape listings, clean and explore the data, train a model, serve predictions through a web app.

**Data**: ~4 600 apartment sale listings scraped from sreality.cz (November 2024, Prague only)

**Model**: CatBoost tuned with Optuna, median absolute prediction error ~7.8%

## Running the app

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Project structure

```
notebooks/
    02_data_cleaning.ipynb              raw data cleaning and feature engineering
    03_exploratory_data_analysis.ipynb  EDA, variable analysis, modelling notes
    04_split_and_pre_modeling.ipynb     train/val/test split, feature selection, preprocessing
    05_modeling.ipynb                   model training, evaluation, SHAP interpretation

scraper/
    sreality_webscraper.ipynb           selenium scraper for sreality.cz

app/
    streamlit_app.py                    prediction UI with district price map

data/
    raw/                                original scraped CSV
    processed/                          cleaned parquets, split datasets

models/                                 saved model and preprocessing artifacts
```

Notebooks are meant to be run in order (02 to 05). Each one saves outputs that the next one loads.

## Modeling

Linear models (Ridge, Lasso, ElasticNet) as a baseline, tree ensembles (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost) tuned with Optuna using 5-fold cross-validation. All tree models use label encoding for categoricals. CatBoost's native encoding was tested but rolled back — it broke sklearn-based visualization tools (SHAP beeswarm, partial dependence plots) with no meaningful performance gain on this dataset.

Final model selection based on validation RMSE. A weighted blend of the top 3 slightly outperformed CatBoost alone (0.169 vs 0.170 validation RMSE), but the gap is small enough that a single model was kept for simplicity of deployment.

Test set performance: RMSE ~0.19, MAE ~0.14, R2 ~0.87 (log-scale). Median absolute percentage error ~7.8% on actual CZK prices.

## Key findings from EDA

- `square_meters`, `layout`, and `district` are the strongest predictors (confirmed by SHAP and permutation importance)
- apartments with kitchenette layouts (+kt) are consistently more expensive per m2 than +1 layouts, driven by building age rather than size
- Praha 10 is the cheapest district despite containing some central cadastral areas like Vinohrady

## Notes

The scraper was built for the sreality.cz layout at the time of collection (November 2024). The website structure may have changed since. Model predictions reflect market conditions at that point.
