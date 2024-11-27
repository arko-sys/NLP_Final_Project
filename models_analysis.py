from mylib.lib import *
import numpy as np
import pandas as pd
# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc


import nltk
nltk.download('wordnet')

# Load raw Reddit dataset.
reddit_data_extracted = pd.read_csv('../NLP_Final_Project/data_raw/reddit_soccer_dataset.csv')

# Fill missing values in 'comment' and 'replies' columns with empty strings to avoid errors.
reddit_data_extracted['clean_comment'] = reddit_data_extracted['comment'].fillna('')  # Replace NaN with an empty string
reddit_data_extracted['clean_replies'] = reddit_data_extracted['replies'].fillna('')  # Replace NaN with an empty string

# Combine the comment and replies into a one column for unified processing.
reddit_data_extracted['combined'] = reddit_data_extracted['clean_comment'].fillna('') + ' ' + reddit_data_extracted['clean_replies'].fillna('')

'''
Apply the Vader sentiment analyzer to compute sentiment scores for the combined text.
the function has been defined in mylib/lib.py
'''

reddit_data_extracted['overall_sentiment'] = reddit_data_extracted['combined'].apply(vader_sentiment)

'''
Categorize sentiment into binary labels 
- 1 for positive
- 0 for negative
'''
reddit_data_extracted['category'] = np.where(reddit_data_extracted['overall_sentiment'] >= 0.00, 1, 0)

combined, sentiment = list(reddit_data_extracted['combined']), list(reddit_data_extracted['category'])

data = reddit_data_extracted[["combined", "category"]]

# Limit dataset size to balance classes by taking an equal number of positive and negative samples.
limit = 13000
data_pos = data[data['category'] == 1][:limit]
data_neg = data[data['category'] == 0][:limit]

dataset = pd.concat([data_pos, data_neg])

# Save the labeled dataset to a processed CSV file.
dataset.to_csv("../NLP_Final_Project/data_processed/large_reddit_labelled.csv")

dataset['combined']=dataset['combined'].str.lower()

# Apply cleaning functions to remove stopwords, punctuation, URLs, and numbers.
dataset['combined'] = dataset['combined'].apply(lambda combined: cleaning_stopwords(combined))
dataset['combined']= dataset['combined'].apply(lambda x: cleaning_punctuations(x))
dataset['combined'] = dataset['combined'].apply(lambda x: cleaning_URLs(x))
dataset['combined'] = dataset['combined'].apply(lambda x: cleaning_numbers(x))

dataset = dataset.reset_index(drop=True)

'''
Tokenize the cleaned text into individual words.
Apply lemmatization to reduce words to their base form.
'''
tokenizer = RegexpTokenizer(r'\w+')
dataset['combined'] = dataset['combined'].apply(tokenizer.tokenize)
dataset['combined'] = dataset['combined'].apply(lambda x: lemmatizer_on_clean_column(x))

# Separate the features (combined text) and labels (category).
X = dataset.combined
y = dataset.category

# Converting tokenized clean_comment back to strings
X = X.apply(lambda x: ' '.join(x))

# Separating the 80% data for training data and 20% for testing data with a set random state to ensure same results
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state =26105111)

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names_out()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

def model_Evaluate(model, model_name):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plot_path = f'../NLP_Final_Project/analysis_plots/confusion_matrix_{model_name}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel, 'BNB')
y_pred1 = BNBmodel.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.savefig('../NLP_Final_Project/analysis_plots/roc_curve_bnb.png', dpi=300, bbox_inches='tight')

LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel, 'LR')
y_pred3 = LRmodel.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.savefig('../NLP_Final_Project/analysis_plots/roc_curve_lr.png', dpi=300, bbox_inches='tight')