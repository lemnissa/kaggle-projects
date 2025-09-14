# House Prices Solution

This project is my solution for the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).  
The goal is to predict the final price of residential homes based on 80+ features describing property characteristics.

## Steps
1. **Data loading and overview**  
   - Train and test datasets with 80+ features (categorical and numerical).  
   - Target variable: `SalePrice`.  

2. **Exploratory Data Analysis (EDA)**  
   - Calculated **Pearson correlation** between features and `SalePrice`.  
   - Selected features with correlation coefficient > 0.5.  
   - Strongest predictors:  
     - `OverallQual` (0.79)  
     - `GrLivArea` (0.71)  
     - `GarageCars` (0.64)  
     - `GarageArea` (0.62)  
     - `TotalBsmtSF` (0.61)  
     - `1stFlrSF` (0.61)  
     - `FullBath` (0.56)  
     - `TotRmsAbvGrd` (0.53)  
     - `YearBuilt` (0.52)  
     - `YearRemodAdd` (0.51)  

3. **Feature engineering**  
   - Converted categorical variables into numerical form with **one-hot encoding** (`pd.get_dummies`).  
   - Filled missing values in test data with zeros (`fillna(0)`).  

4. **Modeling**  
   - Used `RandomForestRegressor` as the main model.  
   - Performed **hyperparameter tuning** with `GridSearchCV` (5-fold cross-validation).  
   - Best parameters found:  
     - `n_estimators = 400`  
     - `max_depth = 16`  
     - `min_samples_leaf = 2`  

5. **Evaluation**  
   - Metric: RMSE on log-transformed target (`np.log1p`) to match Kaggle evaluation.  
   - Best cross-validation RMSE ≈ **0.161**.  

6. **Prediction and submission**  
   - Trained final model on the full training data.  
   - Generated predictions for the test set.  
   - Created `submission.csv` for Kaggle upload.  

## Result
- Final Random Forest model achieved **18329.45802** score on the Kaggle public leaderboard.  

## Tech Stack
Python, pandas, scikit-learn, Jupyter Notebook
