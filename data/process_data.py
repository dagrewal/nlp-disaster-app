import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine

def load_data(feature_file, target_file):
    """
    Load dataset(s) into memory

    Args:
        feature_file: (str) location of the file to be read -> file must be .csv format
                   header on the first line and data on the second line until the end of file.
        target_file: (str) location of the file to be read -> file must be .csv format
                   header on the first line and data on the second line until the end of file. 
    Returns:
        X: (pandas.DataFrame) containing the features from the data
        y: (pandas.DataFrame) containing the target labels from the data
    """

    # read in files
    try:
        X = pd.read_csv(feature_file)
        y = pd.read_csv(target_file)
    except:
        raise Exception("Unable to read in data file.")
    finally:
        return X, y

def merge_data(X, y):
    """
    Merges X and y together

    Args:
        X: (pandas.DataFrame) containing message data
        y: (pandas.DataFrame) containing target label data
    Returns:
        df: (pandas.DataFrame) containing the merged data.
    """

    # merge data
    try:
        if X.shape[0] == y.shape[0]:
            df = X.merge(y, left_on='id', right_on='id')
    except:
        raise Exception("Could not merge datasets.")
    finally:
        return df

def clean_data(df):
    """
    Clean the dataset

    Args:
        df: (pandas.DatFrame) containing data to be cleaned
    Returns:
        df: (pandas.DataFrame) containing the cleaned dataset
    """

    try:
        # clean target labels
        categories = df.categories.str.split(";", expand=True)
        cat_names = categories.applymap(lambda x: x[:-2]).values
        categories = categories.applymap(lambda x: x[-1])
        categories.columns = cat_names.tolist()[0]
        single_cardinality_classes = categories.columns[categories.nunique() == 1]
        categories.drop(single_cardinality_classes, axis=1, inplace=True)

        # there are some errors in the data where related = 2, change this to one
        categories = categories.replace(2, 1)

        # drop original categories column
        df.drop(['categories'], axis=1, inplace=True)

        # convert categories columns to int
        for col in categories.columns:
            categories[col] = categories[col].astype(int)

        # merge categories with df
        df = df.merge(categories, left_index=True, right_index=True)

        # drop duplicates
        df = df.drop_duplicates().reset_index(drop=True)

        # remove original columns as it is not needed for modelling
        df.drop(['original'], axis=1, inplace=True)
    except:
        raise Exception("Could not clean dataset.")
    finally:
        return df

def store_data(df, database):
    """
    Store cleaned dataset into database

    Args:
        df: (pandas.DataFrame) containing dataset to be stored
        database: (str) representing name of database to store dataset into
    Returns:
        None
    """
    try:
        engine = create_engine(f'sqlite:///{database}')
        df.to_sql("disaster_data", engine, index=False, if_exists='replace')
    except:
        raise Exception("Could not store data.")

def main():
    """
    Run process_data.py pipeline
    This function is adapted from https://classroom.udacity.com/nanodegrees/nd025/parts/3f1cdf90-8133-4822-ba56-934933b6b4bb/modules/b46b8867-d211-4be9-88f9-2365a35874be/lessons/7a929d2c-6da9-49d4-9849-e725b8c6e7a2/concepts/94f3a9bf-52af-4c12-82e2-b6065716fa1f

    Args:
        None
    Returns:
        None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        X, y = load_data(messages_filepath, categories_filepath)

        print("Merging message and category data together...")
        df = merge_data(X,y)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Storing data...\n    DATABASE: {}'.format(database_filepath))
        store_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()

