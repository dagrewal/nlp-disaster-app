from . import app
import sys
import os
from flask import Flask, render_template, url_for, jsonify, request
import json, plotly
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import pandas as pd
import numpy as np

#app = Flask(__name__)

# load data from sql
database_path = os.path.abspath(os.getcwd()) + "/data/disasters.db"
engine = create_engine(f'sqlite:///{database_path}')
df = pd.read_sql_table(table_name='disaster_data', con=engine)

targets = df.iloc[:,3:].copy()
df = df.iloc[:,:3]
labels = targets.columns.tolist()

# load the pre-trained model
def tokenize(text):
    """
    Tokenize messages using TweetTokenizer

    Args:
        text: (str) containing the message to be tokenized
    Returns:
        clean_tokens: (list) containing the clean tokens of the input message (text)
    """
    try:
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

model_path = os.path.abspath(os.getcwd()) + "/models/etrees_260720.pkl"
sys.path.append(os.path.abspath(os.getcwd()) + '/models/')
model = joblib.load(model_path)

@app.route('/')
@app.route('/index')
def index():

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    top_targets = targets.sum().sort_values(ascending=False)[:10]
    target_names = top_targets.index
    target_counts = top_targets.values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker_color='turquoise',
                    marker_line_color='white'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'font': {
                    'color':'white'
                },
                'plot_bgcolor':'transparent',
                'paper_bgcolor':'transparent',
                'yaxis': {
                    'title': "Count",
                    'color':'white'
                },
                'xaxis': {
                    'title': "Genre",
                    'color':'white',
                    'showgrid':False
                }
            }
        },
        {
            'data': [
                Bar(
                    x=target_names,
                    y=target_counts,
                    marker_color='turquoise',
                    marker_line_color='white'
                )
            ],

            'layout': {
                'title': 'Distribution of Most Common Target Labels',
                'font': {
                    'color': 'white'
                },
                'plot_bgcolor':'transparent',
                'paper_bgcolor':'transparent',
                'yaxis': {
                    'title': "Count",
                    'color':'white'
                },
                'xaxis': {
                    'title': "Genre",
                    'color':'white',
                    'showgrid':False
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', ids=ids, graphJSON=graphJSON)

@app.route('/about')
def about():

    return render_template('about.html')

@app.route('/results')
def results():

    # get the message that the user entered to be predicted and store as df (required for columntransformer)
    query = request.args.get('query', '')
    query_df = pd.DataFrame({'message':query}, index=[0])

    # predict the classes for the message inside the query
    query_pred = model.predict(query_df)[0]
    model_results = dict(zip(labels, query_pred))

    return render_template('results.html', query=query, model_results=model_results,
                           enumerate=enumerate)

def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()