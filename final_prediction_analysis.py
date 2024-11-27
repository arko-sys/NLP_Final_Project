import numpy as np
import pandas as pd
from mylib.lib import tokenize_function
from datasets import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments


# Load the tokenizer for DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

df = pd.read_csv('../NLP_Final_Project/data_processed/large_reddit_labelled.csv')  

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
    output_dir="./results",          # Directory for saved models
    evaluation_strategy="epoch",    # Evaluate every epoch
    learning_rate=2e-5,             # Learning rate
    per_device_train_batch_size=16, # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=3,             # Number of epochs
    weight_decay=0.01,              # Apply weight decay
    logging_dir='./logs',           # Directory for logging
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

predict_df = pd.read_csv('../NLP_Final_Project/data_processed/reddit_rising.csv')  # Replace with your file path
predict_dataset = predict_df.rename(columns={'combined':'text'})

predict_dataset = Dataset.from_pandas(predict_dataset)

predict_dataset = predict_dataset.map(tokenize_function, batched=True)
predictions = trainer.predict(predict_dataset)

y_preds = np.argmax(predictions.predictions, axis=1)

predict_df["category"] = y_preds

predict_df.groupby("team")["category"].mean()

