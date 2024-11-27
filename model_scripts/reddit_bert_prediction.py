import os
import sys
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mylib.lib import tokenize_function

def get_latest_checkpoint(checkpoint_dir='./results'):
    # Get list of all directories in the checkpoint directory
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    
    if not checkpoints:
        raise ValueError("No checkpoints found in the specified directory.")
    
    # Get the latest checkpoint by sorting numerically
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

# Get the latest checkpoint dynamically
latest_checkpoint = get_latest_checkpoint('./model_scripts/results')

# Load the tokenizer and fine-tuned model
tokenizer = DistilBertTokenizer.from_pretrained(latest_checkpoint)
model = DistilBertForSequenceClassification.from_pretrained(latest_checkpoint)

# Initialize the HuggingFace Trainer with the fine-tuned model
trainer = Trainer(
    model=model,
    tokenizer=tokenizer
)

# Load the new dataset for prediction
predict_df = pd.read_csv('data_processed/reddit_rising.csv')  # Replace with your file path

# Rename the column to match the expected input format
predict_dataset = predict_df.rename(columns={'combined': 'text'})

# Convert the DataFrame to a HuggingFace Dataset
predict_dataset = Dataset.from_pandas(predict_dataset)

# Tokenize the prediction dataset
predict_dataset = predict_dataset.map(tokenize_function, batched=True)

# Ensure the dataset format matches the model's input requirements
predict_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Make predictions using the fine-tuned model
predictions = trainer.predict(predict_dataset)

# Extract predicted class labels
y_preds = np.argmax(predictions.predictions, axis=1)

# Add predictions to the original DataFrame
predict_df["category"] = y_preds

# Perform analysis on the predictions
team_category_means = predict_df.groupby("team")["category"].mean()

# Print or save the results
print(team_category_means)

team_category_means.to_csv("data_processed/bert_predictions.csv")
