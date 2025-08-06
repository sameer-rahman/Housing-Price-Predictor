# 🏠 Housing Price Prediction with Machine Learning

## TL;DR  
This project builds a high-performance housing price prediction model using XGBoost and a custom-engineered feature set on a 200K-row Kaggle dataset.  
Key contributions include a novel `luxury_score`, interpretable SHAP analysis, residual diagnostics, and baseline model comparisons.  
The final model achieves **RMSE ≈ $94,000** and **R² ≈ 0.94**.

---

## Project Overview  
This notebook presents an end-to-end housing price prediction pipeline, combining feature-rich structured data with domain-informed engineering. The goal is to capture both standard and luxury housing trends while ensuring interpretability and reproducibility.

### The approach includes:
- Feature design rooted in real-world housing signals  
- Model calibration and uncertainty analysis  
- Visual diagnostics and SHAP explanations

---

## Key Contributions

### Feature Engineering
- Introduced `luxury_score`: a composite metric capturing home size, finish quality, and bath count  
- Defined `is_luxury` and `luxury_x_*` interaction features to help model high-end pricing behavior  
- Applied log transformations and interaction terms  
- Handled missing data systematically

### Modeling and Evaluation
- Trained and compared XGBoost, LightGBM, CatBoost, and Random Forest  
- Tuned XGBoost hyperparameters using `RandomizedSearchCV`  
- Evaluated performance with RMSE, R², and residual plots

### Explainability and Diagnostics
- Applied SHAP to quantify feature impact and direction  
- Visualized prediction errors and residual variance  
- Included full model training diagnostics and baseline comparisons

---

## Technical Stack  
- Python, Pandas, NumPy, scikit-learn  
- XGBoost, LightGBM, CatBoost  
- SHAP, Matplotlib, Seaborn

---

## Final Model Performance

| Metric        | Value       |
|---------------|-------------|
| RMSE          | ~$94,000    |
| R² Score      | ~0.94       |
| Dataset Size  | ~200,000 rows |

---

## Repository Structure

```
HOUSING_PRICE_PREDICTOR/
├── archive/                    # Raw exploratory notebook (dev log)
│   └── raw_notebook.ipynb
├── catboost_info/             # CatBoost metadata (auto-generated)
├── data/                      # Dataset (not included in repo)
│   └── dataset.csv
├── models/                    # Saved model artifact
│   └── final_xgb_model.pkl
├── final_model_pipeline.ipynb # Cleaned production notebook
├── submission.ipynb_gitignore # Output exclusion file
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview (this file)
```

---
