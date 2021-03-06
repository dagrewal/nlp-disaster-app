import sys
import os
from functools import partial

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import coverage_error, classification_report

from skmultilearn.problem_transform import LabelPowerset

import re
import joblib

from utils import tokenize
from StartingVerbExtractor import StartingVerbExtractor
from CategoryTermExtractor import CategoryTermExtractor


def load_data(database_path):
    """
    Load the data from the database 

    Args:
        database_path: (str) representing the path to the database where the data is stored
    Returns:
        X: (pandas.DataFrame) containing the features of the data
        yt: (pandas.DataFrame) containing the multilabel target data
        y_resampled:
        category_names: (list) containing the name of the target labels
    """

    try:
        engine = create_engine(f"sqlite:///{database_path}")
        df = pd.read_sql_table(table_name="disaster_data".lower(), con=engine)

        # seperate features and targets
        X = pd.DataFrame(df.loc[:,'message'])
        y = df.iloc[:,3:]
        category_names = y.columns.tolist()

        return X, y, category_names
    except:
        raise Exception("Could not load data.")
        
def build_model(category_names):
    """
    Builds a multilabel classifier using X and y from load_data()

    Args:
        None
    Returns:
        cv: (sklearn.model_selection.GridSearchCV) estimator containing the model and the parameters to be trained
    """
    try:
        # initialise columns to be used for data preparation purposes in the model pipeline
        message_col = 0

        # build a pipeline containing the feature transformations and estimator
        pipeline = Pipeline([

            ('features', ColumnTransformer([
                # apply message transformations
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=partial(tokenize))),
                    ('tfidf', TfidfTransformer())
                ]), message_col),

                ('starting_verb', StartingVerbExtractor(), message_col),

                ('category_terms', CategoryTermExtractor(category_names=category_names),
                message_col),

            ], remainder='drop')),

            # specify the estimator
            ('clf', LabelPowerset(MultinomialNB(fit_prior=True)))
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
            'clf__classifier__alpha': [0.01, 1.]
        }

        # perform cross validation using grid search on the pipeline described above
        cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, verbose=2)
        return cv
    except:
        raise Exception("Could not build model.")
    #finally:
    #    return cv
 
def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate the performance of the model on test data using coverage error and classification report

    Args:
        model: (sklearn.model_selectionGridSearchCV) estimator created from build_model()
        X_test: (pandas.DataFrame) containing the features for the test data
        y_test: (pandas.DataFrame) containing the multilabel targets for the test data
        category_names: (list) containing the target names from load_data()
    Returns:
        None
    """
    try:
        y_pred = model.predict(X_test)
        y_test_avg_labels = round(np.mean(y_test.sum(axis=1)), 2)
    
        print("Printing classification report...\n")
        y_pred = model.predict(X_test).todense()

        i = -1
        for col in category_names:
            i += 1
            ytrue = y_test[col]
            ypred = y_pred[:,i]
            print(col)
            print(classification_report(ytrue, ypred))        
            print('-' * 60)

        print("\n Printing coverage error...\n")
        print(round(coverage_error(y_test, y_pred), 2))
        print(f"\n Average number of true labels per sample in test sample: {y_test_avg_labels}")
    except:
        raise Exception("Could not evaluate model.")

def save_model(model, model_filepath):
    """
    Save the model that was created above

    Args:
        model: (sklearn.model_selection.GridSearchCV) estimator from build_model()
        model_filepath: (str) location of where to save the model
    Returns:
        None
    """
    try:
        filename = f'{model_filepath}.pkl'
        joblib.dump(model,
                    open(filename, 'wb'),
                    compress=3)
    except:
        raise Exception("Could not save model.")

def main():
    try:
        if len(sys.argv) == 3:
            database_filepath, model_filepath = sys.argv[1:]
            print('Loading data...\n    DATABASE: {}'.format(database_filepath))
            X, Y, category_names = load_data(database_filepath)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
            
            print('Building model...')
            model = build_model(category_names)
            
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
    except:
        raise Exception("Could not run train_classifier.py.")

if __name__ == '__main__':
    main()
