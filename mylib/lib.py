from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import re
import string
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

STOPWORDS = set(stopwordlist)
punctuations_list = string.punctuation
vader_analyzer = SentimentIntensityAnalyzer()
lm = WordNetLemmatizer()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

def vader_sentiment(data):
    """
    This function uses VADER Sentiment Analyzer to compute
    the sentiment score for a given text. It returns the
    compound score if the input is a string, otherwise None.
    """
    if isinstance(data, str):
        sentiment = vader_analyzer.polarity_scores(data)
        return sentiment['compound']
    return None  

# created a function for lemmatizing the clean_column
def lemmatizer_on_clean_column(data):
    clean_column = [lm.lemmatize(word) for word in data]
    return clean_column

def cleaning_stopwords(data):
    """
    This function removes all stopwords from the input text.
    It returns a string with only the non-stopword tokens.
    """
    return " ".join([word for word in str(data).split() if word not in STOPWORDS])

def cleaning_punctuations(data):
    """
    This function removes punctuation characters from the input text.
    It uses a translation table to strip punctuations efficiently.
    """
    translator = str.maketrans('', '', punctuations_list)
    return data.translate(translator)

def cleaning_URLs(data):
    """
    This function removes URLs from the input text using a regular expression.
    It looks for patterns matching common URL structures and replaces them with a space.
    """
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)

def cleaning_numbers(data):
    """
    This function removes numeric characters from the input text.
    It uses a regular expression to identify and replace numbers with an empty string.
    """
    return re.sub('[0-9]+', '', data)


