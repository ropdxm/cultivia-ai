from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

import pickle
import pandas as pd
import os
import numpy as np


# Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/SVMClassifier.pkl'), 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index2.html')

@app.route('/results', methods=['POST'])
def predict():
	n = int(request.form['N'])
	p = int(request.form['P'])
	k = int(request.form['K'])
	temperature = float(request.form['temperature'])
	humidity = float(request.form['humidity'])
	ph = float(request.form['ph'])
	rainfall = float(request.form['rainfall'])

	input_data = [{'N': n, 'P': p, 'K': k, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}]
	data = pd.DataFrame(input_data)

	logreg = clf.predict(data)[0]
	return render_template('results.html', res=logreg)

if __name__ == '__main__':
	app.run(debug=False, host="0.0.0.0")