# NLP with Disaster Tweets  

This project is my solution for the Kaggle competition **[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)**.  
The goal of the competition is to classify tweets as describing a real disaster or not.  

## Approach  

1. **Exploratory Data Analysis (EDA)**  
   - Inspected text length, most common words, and class balance.  
   - Cleaned text: lowercasing, removing punctuation, stopwords, URLs, mentions, and hashtags.  

2. **Baseline models**  
   - TF-IDF features + Logistic Regression.  
   - Naive Bayes and simple neural networks for comparison.  

3. **Transformer model**  
   - Fine-tuned **DistilBERT (HuggingFace Transformers)** for text classification.  
   - Used GPU on Kaggle for training.  
   - Applied stratified train/validation split and early stopping.  

4. **Evaluation**  
   - Metrics: accuracy, F1-score (main focus), and confusion matrix.  
   - DistilBERT significantly outperformed baseline models.  

## Tech stack  

- Python, pandas, numpy  
- scikit-learn, matplotlib, seaborn  
- HuggingFace Transformers (DistilBERT)  
- PyTorch  

## Results  

- Baseline (TF-IDF + Logistic Regression): ~0.77 F1-score  
- DistilBERT fine-tuning: ~0.84 F1-score (validation)  
