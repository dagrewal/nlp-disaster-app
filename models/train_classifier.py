import sys
import multiprocessing

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import coverage_error, classification_report

from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

import re
import joblib

from StartingVerbExtractor import StartingVerbExtractor

# initialise stopword, tokenizer and lemmatizer to be used for feature engineering of messages
stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tweet_tokenizer = TweetTokenizer()

# specify number of available coures -- needed for parallel model training
n_cores = multiprocessing.cpu_count()

def load_data(database_path):
    """
    Load the data from the database 

    Args:
        database_path: (str) representing the path to the database where the data is stored
    Returns:
        X: (pandas.DataFrame) containing the features of the data
        y: (pandas.DataFrame) containing the multilabel target data
        category_names: (list) containing the name of the target labels
    """

    try:
        engine = create_engine(f"sqlite:///{database_path}")
        df = pd.read_sql_table(table_name="disaster_data", con=engine)

        # seperate features and targets
        X = df.iloc[:,:3]
        y = df.iloc[:,3:]
        y = y.replace(2, 1)
        category_names = y.columns.tolist()
    except:
        raise Exception("Could not load data.")
    finally:
        return X, y, category_names

def tokenize(text):
    """
    Tokenize messages using TweetTokenizer

    Args:
        text: (str) containing the message to be tokenized
    Returns:
        clean_tokens: (list) containing the clean tokens of the input message (text)
    """
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

    return clean_tokens

def build_model():
    """
    Builds a multilabel classifier using X and y from load_data()

    Args:
        None
    Returns:
        cv: (sklearn.model_selection.GridSearchCV) estimator containing the model and the parameters to be trained
    """
    # initialise columns to be used for data preparation purposes in the model pipeline
    message_col = 'message'
    genre_col = 'genre'
    id_col = 'id'

    # build a pipeline containing the feature transformations and estimator
    pipeline = Pipeline([

        ('features', ColumnTransformer([
            # apply message transformations
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]), message_col),

            ('starting_verb', StartingVerbExtractor(), message_col),

            #('genre', GenreLabelEncoder(), genre_col),

        ], remainder='drop')),
        # specify the estimator
        ('clf', MultiOutputClassifier(ExtraTreesClassifier()))
    ])

    # parameter grid to be used for grid search
    parameters = {
        'features__text_pipeline__vect__max_features': [10000],
        'features__text_pipeline__tfidf__sublinear_tf': [True],
        'features__text_pipeline__vect__ngram_range': [(1,1), (1,2)],
        'features__text_pipeline__vect__min_df': [1],
        'features__text_pipeline__vect__max_df': [.95],
        'features__text_pipeline__tfidf__smooth_idf': [True],
        'features__text_pipeline__tfidf__norm': ['l2'],
        'clf__estimator__n_estimators': [100, 300, 500],
        'clf__estimator__max_features': ['auto'],
        'clf__estimator__min_samples_leaf': [10],
        'clf__estimator__max_depth': [.7, .9]
    }

    # perform cross validation using grid search on the pipeline described above
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=1)

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate the performance of the model on test data

    Args:
        model: (sklearn.model_selectionGridSearchCV) estimator created from build_model()
        X_test: (pandas.DataFrame) containing the features for the test data
        y_test: (pandas.DataFrame) containing the multilabel targets for the test data
        category_names: (list) containing the target names from load_data()
    Returns:
        (sklearn.metrics.multilabel_confusion_matrix) showing the performance statistics on the test data
    """
    y_pred = model.predict(X_test)
    y_test_avg_labels = round(np.mean(y_test.sum(axis=1)), 2)

    print("Printing classification report...\n")
    print(classification_report(y_test, y_pred, target_names=category_names, zero_division=0))
    print("\n Printing coverage error...\n")
    print(coverage_error(y_test, y_pred))
    print(f"\n Average number of true labels per sample in test sample: {y_test_avg_labels}")

def save_model(model, model_filepath):
    """
    Save the model that was created above

    Args:
        model: (sklearn.model_selection.GridSearchCV) estimator from build_model()
        model_filepath: (str) location of where to save the model
    Returns:
        None
    """
    filename = f"{model_filepath}.sav"
    joblib.dump(model, filename) 


def main():
    """
    This function is adapted from https://classroom.udacity.com/nanodegrees/nd025/parts/3f1cdf90-8133-4822-ba56-934933b6b4bb/modules/b46b8867-d211-4be9-88f9-2365a35874be/lessons/7a929d2c-6da9-49d4-9849-e725b8c6e7a2/concepts/94f3a9bf-52af-4c12-82e2-b6065716fa1f

    Args:
        None
    Returns:
        None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()