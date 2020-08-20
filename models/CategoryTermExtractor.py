from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import TweetTokenizer
import pandas as pd

class CategoryTermExtractor(BaseEstimator, TransformerMixin):
    """
    New feature which identifies whether or not the message contains any of the category terms
    """
    def __init__(self, category_names):
        self.category_names = category_names

    def category_term(self, text):
        tweet_tokenizer = TweetTokenizer()
        message_list = tweet_tokenizer.tokenize(text)
        for message in message_list:
            if len(set(message).intersection(set(self.category_names))) >= 1:
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_categories = pd.Series(X).apply(self.category_term)
        return pd.DataFrame(X_categories)
