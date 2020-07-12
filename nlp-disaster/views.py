from . import app
from flask import render_template
import json, plotly

@app.route('/')
@app.route('/index')
def index():

    return render_template('index.html')