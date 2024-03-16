from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

print("Current working directory:", os.getcwd())
app = Flask(__name__, template_folder='templates')

model = pickle.load(open('../Model/model.pkl', 'rb'))

data = pd.read_csv('../CarsData.csv')
manufacturers = data['Manufacturer'].unique().tolist()
models = data['model'].unique().tolist()
min_year = data['year'].min()
max_year = data['year'].max()
engine_sizes = sorted(data['engineSize'].unique())


@app.route('/')
def home():
    print("Home route was accessed")
    return render_template('index.html', manufacturers=manufacturers, models=models, min_year=min_year,
                           max_year=max_year, engine_sizes=engine_sizes)


@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route was accessed")
    mileage = request.form.get('mileage', type=int)
    if mileage is None or not (0 <= mileage <= 300000):
        return "Invalid mileage. Please enter a value between 0 and 300,000."

    return "Prediction placeholder"


if __name__ == "__main__":
    app.run(debug=True)
