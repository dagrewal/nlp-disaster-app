from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import re

def tokenize(text):
    """
    Tokenize messages using TweetTokenizer

    Args:
        text: (str) containing the message to be tokenized
    Returns:
        clean_tokens: (list) containing the clean tokens of the input message (text)
    """
    clean_tokens = []
    try:

        stops = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tweet_tokenizer = TweetTokenizer()

        # keep only lowercase letters
        text = re.sub("[^a-z]", " ", text.lower())

        # initialise empty list to hold tokens
        clean_tokens = []

        # tokenize text
        tokens = tweet_tokenizer.tokenize(text)

        # lemmatize each tokenized term in tokens and append to clean_tokens
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

        # keep only tokens that are not stopwords
        clean_tokens = [w for w in clean_tokens if w not in stops]
    except:
        raise Exception("Could not tokenize text.")
    finally:
        return clean_tokens