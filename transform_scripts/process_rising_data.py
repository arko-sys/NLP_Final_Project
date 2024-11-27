import pandas as pd

reddit_data_extracted = pd.read_csv('data_raw/reddit_rising.csv')

reddit_data_extracted['clean_comment'] = reddit_data_extracted['comment'].fillna('')  
reddit_data_extracted['clean_replies'] = reddit_data_extracted['replies'].fillna('')  

reddit_data_extracted['combined'] = reddit_data_extracted['clean_comment'].fillna('') + ' ' + reddit_data_extracted['clean_replies'].fillna('')

data = reddit_data_extracted[["team", "combined"]]

data.to_csv("data_processed/reddit_rising.csv")