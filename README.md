# Sentiment Analysis of Reddit Posts for Premier League Teams

## Overview
This project performs sentiment analysis on Reddit posts related to the 20 Premier League soccer teams. The posts are scraped using the PRAW (Python Reddit API Wrapper) library, and the data is then preprocessed, cleaned, and analyzed using machine learning techniques to classify the sentiment as either positive or negative. We use a variety of machine learning models, including traditional methods like Naive Bayes, Logistic Regression, and advanced models like DistilBERT.

## Data Collection
The Reddit posts for the 20 Premier League teams are gathered using PRAW. The script fetches the most recent posts and their comments from subreddits dedicated to each team. This dataset is stored in a CSV file: `reddit_soccer_dataset.csv`.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Implementations](#model-implementations)
   - [BERT](#bert)
   - [Multinomial Naive Bayes (BNB)](#multinomial-naive-bayes-bnb)
   - [Logistic Regression (LR)](#logistic-regression-lr)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Analysis Plots](#analysis-plots)
6. [Conclusion](#conclusion)

## File Structure

```
NLP_Final_Project/
│
├── analysis_plots/
│   ├── confusion_matrix_bert.png
│   ├── confusion_matrix_bert_no_finetune.png
│   ├── confusion_matrix_BNB.png
│   ├── confusion_matrix_LR.png
│   ├── roc_curve_bert.png
│   ├── roc_curve_bert_no_finetune.png
│   ├── roc_curve_bnb.png
│   └── roc_curve_lr.png
│
├── data_processed/
│   ├── large_reddit_labelled.csv
│   └── reddit_rising.csv
|
├── extract_scripts/
│   ├── fetch_team_new_data.ipynb
│   └── fetch_team_rising_data.ipynb
|
├── model_scripts/
│   ├── bert_finetuned.py
│   ├── bert_pretrained.py
│   ├── reddit_bert_prediction.py
│   └── models_analysis.py
|
├── mylib/
│   ├── __init__.py
│   └── lib.py
|
├── notebooks/
│   ├── bert.ipynb
│   ├── understood.ipynb
│   ├── underv2 copy.ipynb
│   └── underv2.ipynb
│
├── transform_scripts/
│   └── process_rising_data.ipynb
|
├── README.md
└── requirements.txt
```
---

## Introduction
In this analysis, we explore three different models for performing text classification on a dataset. Each model is evaluated using common machine learning metrics such as accuracy, precision, recall, F1-score, and AUC. The primary objective is to understand how well each model performs on the given dataset and identify any strengths or weaknesses that may influence their use in practical applications.

---

## Data Preprocessing
Before applying machine learning models, the text data was cleaned and preprocessed. The following preprocessing steps were applied:
1. **Tokenization**: Texts were split into words or tokens.
2. **Lowercasing**: All text was converted to lowercase to ensure uniformity.
3. **Stopword Removal**: Common stopwords (e.g., "the", "is", etc.) were removed.
4. **Lemmatization**: Words were reduced to their base forms (e.g., "running" → "run").
5. **Vectorization**: Text was transformed into numerical features using techniques such as TF-IDF.

---

## Model Implementations

### BERT
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model pre-trained on large corpora of text. It excels at understanding the context of words in a sentence and can achieve state-of-the-art performance in various natural language processing tasks.

1. **Preprocessing**: Tokenization was done using BERT’s tokenizer.
2. **Training**: The model was fine-tuned on the labeled dataset using cross-entropy loss.

**Evaluation Results for BERT:**
- Accuracy: 0.89

### Multinomial Naive Bayes (BNB)
BNB is a probabilistic classifier based on Bayes' theorem, typically used for text classification. It assumes that the features (words) are conditionally independent, given the class.

1. **Preprocessing**: Text was vectorized using TF-IDF.
2. **Training**: The model was trained using the Naive Bayes classifier.

**Evaluation Results for BNB:**
- Accuracy: 0.78

### Logistic Regression (LR)
Logistic Regression is a linear model that predicts the probability of a class based on a linear combination of input features. It is widely used for binary and multiclass classification tasks.

1. **Preprocessing**: TF-IDF was used for vectorization.
2. **Training**: The model was trained using logistic regression with regularization.

**Evaluation Results for LR:**
- Accuracy: 0.85

---

## Evaluation Metrics
Each model's performance was evaluated using the following metrics:
- **Accuracy**: The proportion of correctly predicted instances out of the total instances.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positives.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.
- **AUC**: The area under the receiver operating characteristic curve, indicating the model's ability to discriminate between the classes.

---

## Analysis Plots

The following analysis plots visualize the performance of each model:

### 1. Confusion Matrix for BERT (No Fine-Tuning)
   ![BERT Confusion Matrix (No Fine-Tuning)](analysis_plots/confusion_matrix_bert_no_finetune.png)

### 2. Confusion Matrix for BERT 
   ![BERT Confusion Matrix](analysis_plots/confusion_matrix_bert.png)

### 3. Confusion Matrix for BNB 
   ![BNB Confusion Matrix](analysis_plots/confusion_matrix_BNB.png)

### 4. Confusion Matrix for LR  
   ![LR Confusion Matrix](analysis_plots/confusion_matrix_LR.png)

### 5. ROC Curve for BERT (No Fine-Tuning)  
   ![BERT ROC Curve (No Fine-Tuning)](analysis_plots/roc_curve_bert_no_finetune.png)

### 6. ROC Curve for BERT
   ![BERT ROC Curve](analysis_plots/roc_curve_bert.png)

### 7. ROC Curve for BNB  
   ![BNB ROC Curve](analysis_plots/roc_curve_bnb.png)

### 8. ROC Curve for LR  
   ![LR ROC Curve](analysis_plots/roc_curve_lr.png)

These plots provide a visual representation of each model's performance across different classification metrics, helping us understand how well each model differentiates between classes and handles misclassifications.

---

## Conclusion of Model Picking

Based on the evaluation metrics and analysis plots, we can draw the following conclusions:

- **BERT** outperforms both **BNB** and **LR** in terms of accuracy, precision, recall, F1-score, and AUC. Its ability to understand the context of text makes it a strong choice for text classification tasks.
- **BNB** performs well but lags behind the other two models in terms of recall and AUC. It is a simpler model that is computationally more efficient.
- **LR** strikes a balance between performance and computational efficiency, though it slightly underperforms compared to BERT.

## Model Selection and Final Analysis

### 1. Sentiment Analysis and Comparison with FanDuel Money Line Values
After training the DistilBERT model, we used it to perform sentiment analysis on a new dataset of Reddit posts, which we called the "Reddit Rising" dataset. This dataset contained recent posts related to the Premier League teams, and we applied the trained model to predict the sentiment for each post. The sentiment scores were aggregated by team, resulting in an average sentiment score for each team.

### 2. Final Results and Insights
The final step involved comparing the sentiment analysis results with the current FanDuel money line values, which indicate the odds or expectations for each team. By comparing team sentiment (as derived from the Reddit posts) with the betting odds, we sought to uncover any potential correlations or discrepancies between how fans feel about a team and the odds provided by FanDuel.

This analysis allows us to gauge whether the sentiment of the fan base is aligned with the betting markets. If a significant mismatch is identified, it may suggest a potential opportunity for betting or further investigation into factors affecting fan sentiment or team performance.

### Team Sentiment Scores

| Team                | Sentiment Score |
|---------------------|-----------------|
| Arsenal            | 0.809129        |
| Aston Villa        | 0.749518        |
| Bournemouth        | 0.804020        |
| Brentford          | 0.744275        |
| Brighton           | 0.826748        |
| Chelsea            | 0.783317        |
| Crystal Palace     | 0.782895        |
| Everton            | 0.627595        |
| Fulham             | 0.674912        |
| Ipswich Town       | 0.845815        |
| Leicester City     | 0.696281        |
| Liverpool          | 0.732999        |
| Man City           | 0.717557        |
| Man United         | 0.718289        |
| Newcastle          | 0.642105        |
| Nottingham Forest  | 0.687117        |
| Southampton        | 0.679167        |
| Tottenham          | 0.783784        |
| West Ham           | 0.762082        |
| Wolves             | 0.888112        |

The results from this analysis will be useful for understanding the relationship between public sentiment and betting behavior, providing valuable insights for both sports enthusiasts and bettors.
