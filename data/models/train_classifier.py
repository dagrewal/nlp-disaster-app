import sys
import pandas
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize, TweetTokenizer

def load_data(database):
    """
    Load the data from the database

    Args:
        database: (str) representing the database where the data is stored
    Returns:
        X: (pandas.DataFrame) containing the features of the data
        y: (pandas.DataFrame) containing the target data
        category_names: (list) containing the name of the labels
    """

    try:
        engine = create_engine(f"sqlite:///{database}")
        df = pd.read_sql_table(table_name="disaster_data", con=engine)

        # seperate features and targets
        X = df.iloc[:,:3]
        y = df.iloc[:,3:].values
        category_names = df.columns[3:].tolist()
    except:
        raise Exception("Could not load data.")
    finally:
        return X, y, category_names

class MessageEngineer(BaseEstimator, TransformerMixin):
    """
    Engineers features from the categorical features.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self):


def build_model():
    """
    Builds a classifier using X and y from load_data()
    """



def evaluate_model():

def save_model():

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