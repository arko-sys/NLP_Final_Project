import sys
import os
import numpy as np
import pandas as pd
from datasets import Dataset
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mylib.lib import tokenize_function

# Load the tokenizer for DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the dataset
df = pd.read_csv('../data_processed/large_reddit_labelled.csv')  

print(df.shape)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["combined"], df["category"], test_size=0.2, random_state=42)

train_dataset = pd.concat([X_train, y_train], axis=1)
test_dataset = pd.concat([X_test, y_test], axis=1)

train_dataset = train_dataset.rename(columns={'category': 'label', 'combined':'text'})
test_dataset = test_dataset.rename(columns={'category': 'label', 'combined':'text'})

train_dataset = Dataset.from_pandas(train_dataset)
test_dataset = Dataset.from_pandas(test_dataset)

# Tokenize the training and test datasets using a custom tokenize function
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load the pre-trained DistilBERT model for binary classification (without fine-tuning)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Function for making predictions
def make_predictions(model, test_dataset):
    # Get the model to evaluation mode
    model.eval()
    
    predictions = []
    true_labels = []
    
    # Loop through test dataset and make predictions
    for i in range(len(test_dataset)):
        text = test_dataset[i]['text']
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class = np.argmax(logits.numpy(), axis=1)[0]  # Get the predicted class
        predictions.append(predicted_class)
        true_labels.append(test_dataset[i]['label'])

    return np.array(true_labels), np.array(predictions)

# Make predictions on the test set
y_test_preds, y_preds = make_predictions(model, test_dataset)

# Evaluate the model
print(classification_report(y_test, y_preds, target_names=['Negative', 'Positive']))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_preds)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - DistilBERT (Without Fine-tuning)')
plt.legend(loc="lower right")
plt.savefig('../analysis_plots/roc_curve_bert_no_finetune.png', dpi=300, bbox_inches='tight')

# Compute Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_preds)

# Plot Confusion Matrix
categories = ['Negative', 'Positive']
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

plt.figure(figsize=(6, 5))
sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='', xticklabels=categories, yticklabels=categories)
plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
plt.title("Confusion Matrix - DistilBERT (Without Fine-tuning)", fontdict={'size': 18}, pad=20)
plt.savefig('../analysis_plots/confusion_matrix_bert_no_finetune.png', dpi=300, bbox_inches='tight')
