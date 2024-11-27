import sys
import os
import numpy as np
import pandas as pd
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mylib.lib import tokenize_function

# Load the tokenizer for DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

df = pd.read_csv('data_processed/large_reddit_labelled.csv')  

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

# Set the format of datasets for PyTorch (input_ids, attention_mask, and labels)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load the pre-trained DistilBERT model for binary classification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define training arguments for the HuggingFace Trainer API
training_args = TrainingArguments(
    output_dir="./model_scripts/results",          # Directory for saved models
    evaluation_strategy="epoch",    # Evaluate every epoch
    learning_rate=2e-5,             # Learning rate
    per_device_train_batch_size=16, # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=3,             # Number of epochs
    weight_decay=0.01,              # Apply weight decay
    logging_dir='./model_scripts/logs',           # Directory for logging
    logging_steps=10,
    save_total_limit=1              # Save only the best checkpoint
)

# Initialize the HuggingFace Trainer with the model, data, and arguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.evaluate()
predictions = trainer.predict(test_dataset)

# Extract the predicted class labels from the model's predictions
y_preds = np.argmax(predictions.predictions, axis=1)

print(classification_report(y_test, y_preds, target_names=['Negative', 'Positive']))

fpr, tpr, _ = roc_curve(y_test, y_preds)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - DistilBERT')
plt.legend(loc="lower right")
plt.savefig('analysis_plots/roc_curve_bert.png', dpi=300, bbox_inches='tight')

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
plt.title("Confusion Matrix - DistilBERT", fontdict={'size': 18}, pad=20)
plt.savefig('analysis_plots/confusion_matrix_bert.png', dpi=300, bbox_inches='tight')
