# 🏠 Housing Price Prediction

A machine learning project predicting housing prices using tree-based models (CatBoost, XGBoost, GradientBoosting, etc.) with ensembling, model interpretation, and deployment-ready artifacts.

---

## 📂 Project Structure

project/
├── data/ # raw + processed
├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02_Modeling.ipynb
│ ├── 03_Test_and_Interpretation.ipynb
│ └── 04_Deployment.ipynb
├── deployable/ # artifacts (model.joblib, preprocessing.joblib, plots/, model_card.txt)
├── requirements.txt
├── README.md # explains project
└── scraper/ (optional) # web scraping pipeline


---

## 🚀 Workflow

1. **Data Collection**  
   - Optional scraper (`scraper/`) collects housing data.  
   - Data stored in `data/`.

2. **Exploration & Feature Engineering**  
   - Run `notebooks/01_EDA.ipynb`.  
   - Clean, explore, and transform dataset.

3. **Model Training & Leaderboard**  
   - Run `notebooks/02_Modeling.ipynb`.  
   - Trains multiple models, tunes hyperparameters, and saves leaderboard.

4. **Final Evaluation & Interpretation**  
   - Run `notebooks/03_Test_and_Interpretation.ipynb`.  
   - Evaluates best models on the test set.  
   - Generates SHAP/PDP/Permutation Importance plots.  
   - Saves `model_card.txt`.

5. **Deployment**  
   - Run `notebooks/04_Deployment.ipynb`.  
   - Loads trained artifacts from `deployable/`.  
   - Provides a `predict_price(df)` wrapper for inference.

---

## 📊 Results

- **Best Model**: Weighted Blend (GradientBoosting + CatBoost + XGBoost)  
- **Test Metrics**:  
  - RMSE: `0.1745`  
  - MAE: `0.1224`  
  - R²: `0.8702`

See [`deployable/model_card.txt`](deployable/model_card.txt) for full documentation.

---

## 📈 Interpretability

- Permutation importance  
- SHAP values  
- Partial dependence plots  

Visuals available in `deployable/plots/`.

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
pip install -r requirements.txt
