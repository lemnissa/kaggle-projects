# Titanic Solution

This project is my baseline solution for the Kaggle competition [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic).  
It demonstrates a simple end-to-end ML workflow: from exploratory analysis to model training and prediction.

## Steps
1. **Data loading**  
   - Train and test datasets from Kaggle (`train.csv`, `test.csv`).  

2. **Exploratory Data Analysis (EDA)**  
   - Calculated survival rate by gender:  
     - Women survival rate ≈ 74%  
     - Men survival rate ≈ 19%  

3. **Feature engineering**  
   - Selected features: `Pclass`, `Sex`, `SibSp`, `Parch`.  
   - Encoded categorical features with one-hot encoding.  

4. **Model**  
   - Trained a `RandomForestClassifier` (100 trees, max depth = 5).  

5. **Prediction and submission**  
   - Generated predictions for the test set.  
   - Created a `submission.csv` file for Kaggle.  

## Result
- Baseline Random Forest model achieved ~0.78 score on the public leaderboard.  

## Tech Stack
Python, pandas, scikit-learn, Jupyter Notebook
