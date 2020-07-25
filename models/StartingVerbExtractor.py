from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
import pandas as pd

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    New feature - does message begin with verb
    """
    def __init__(self):
        pass

    def starting_verb(self, text):
        tweet_tokenizer = TweetTokenizer()
        message_list = tweet_tokenizer.tokenize(text)
        for message in message_list:
            pos_tags = pos_tag(message)
            first_term, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP']:
                return True
        return False
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)