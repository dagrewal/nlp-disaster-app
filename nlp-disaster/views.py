from . import app
from flask import render_template, url_for
import json, plotly

@app.route('/')
@app.route('/index')
def index():

    return render_template('index.html')

@app.route('/about')
def about():

    return render_template('about.html')