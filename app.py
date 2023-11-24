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
	n = n/140
	p = int(request.form['P'])
	p = (p-5)/140
	k = int(request.form['K'])
	k = (k-5)/200
	temperature = float(request.form['temperature'])
	temperature = (temperature-8.825675)/(43.675493-8.825675)
	humidity = float(request.form['humidity'])
	humidity = (humidity-14.25804)/(99.981876-14.25804)
	ph = float(request.form['ph'])
	ph = (ph-3.504752)/(9.935091-3.504752)
	rainfall = float(request.form['rainfall'])
	rainfall = (rainfall-20.211267)/(298.560117-20.211267)


	input_data = [{'N': n, 'P': p, 'K': k, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}]
	# input_data = np.array([n, p, k, temperature, humidity, ph, rainfall])
	data = pd.DataFrame(input_data)

	print(data)

	logreg = clf.predict(data)[0]
	return render_template('results.html', res=logreg)

if __name__ == '__main__':
	app.run(debug=False, host="0.0.0.0")