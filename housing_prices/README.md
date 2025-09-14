# House Prices Solution

Solution for the Kaggle competition **[Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/kaggle-learn)**.  
**Task:** predict house prices based on tabular data with numerical and categorical features.   

---

## Approaches

### 1. Random Forest (scikit-learn)

- **EDA**: computed correlations with the target variable and selected features with correlation coefficient > 0.5.  
- **Feature engineering**: one-hot encoding for categorical features, handling missing values.  
- **Model**: RandomForestRegressor + GridSearchCV (5-fold CV).  
- **Best hyperparameters**:  
  - `n_estimators = 400`  
  - `max_depth = 16`  
  - `min_samples_leaf = 2`  
- **Result**: RMSE (CV) ≈ **0.161**, Public LB ≈ **18329.46**.  

---

### 2. CatBoost (gradient boosting by Yandex)

- **Key features**:  
  - Native support for categorical variables (no OHE needed).  
  - Handles missing values without manual imputation.  
  - Ordered boosting to prevent target leakage.  
- **Preprocessing**: categorical features converted to strings, numerical features left with NaN values.  
- **Model**: CatBoostRegressor (`depth=8`, `learning_rate=0.05`, `iterations=5000` with `early_stopping_rounds=200`).  
- **Evaluation**:  
  - Best CV-RMSLE ≈ **0.122** at ~1320 iterations.  
  - Kaggle metric: better than RandomForest (14336.9).  
- **Feature importance**: main features overlap with RF (`OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, …).  

---

## Tech Stack

- **Python**: pandas, numpy, matplotlib, seaborn  
- **ML**: scikit-learn, CatBoost
