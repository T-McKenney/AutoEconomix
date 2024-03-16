from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

print("Current working directory:", os.getcwd())
app = Flask(__name__, template_folder='templates')


ml_model = pickle.load(open('/Users/tylermckenney/Desktop/AutoEconomix/Model/model.pkl', 'rb'))

data = pd.read_csv('../CarsData.csv')
manufacturers = data['Manufacturer'].unique().tolist()
models = data['model'].unique().tolist()
min_year = data['year'].min()
max_year = data['year'].max()
engine_sizes = sorted(data['engineSize'].unique())
mpg_values = sorted(data['mpg'].unique())


@app.route('/')
def home():
    print("Home route was accessed")
    return render_template('index.html', manufacturers=manufacturers, models=models, min_year=min_year,
                           max_year=max_year, engine_sizes=engine_sizes, mpg_values=mpg_values)


@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('/Users/tylermckenney/Desktop/AutoEconomix/Model/model.pkl', 'rb'))

    # Collect and prepare input data
    form_data = {
        'Manufacturer': [request.form['manufacturer']],
        'model': [request.form['model']],  # This needs to match your training data's feature name
        'year': [int(request.form['year'])],
        'transmission': [request.form['transmission']],
        'mileage': [int(request.form.get('mileage', 0))],
        'fuelType': [request.form['fuelType']],
        'engineSize': [float(request.form['engineSize'])],
        'mpg': [float(request.form['mpg'])]
    }
    input_df = pd.DataFrame(form_data)

    # Load the entire dataset to prepare for dummy variable encoding
    full_data = pd.read_csv('../CarsData.csv')

    # Append the input_df to full_data to ensure categorical variables are encoded correctly
    combined = pd.concat([full_data, input_df], sort=False).reset_index(drop=True)

    # Convert categorical variables using get_dummies and drop the first to avoid dummy variable trap
    combined_encoded = pd.get_dummies(combined, drop_first=True)

    # Separate the last row as input data for prediction
    input_encoded = combined_encoded.tail(1)

    # Explicitly making a copy of the DataFrame to avoid SettingWithCopyWarning (was experiencing this before this fix)
    input_encoded = combined_encoded.tail(1).copy()

    # Create a filtered DataFrame with only the required columns
    required_columns = [col for col in model.feature_names_in_ if col in input_encoded.columns]
    input_encoded_filtered = input_encoded[required_columns]

    # Adding missing columns with 0s to match the model's training features
    for column in model.feature_names_in_:
        if column not in input_encoded_filtered.columns:
            input_encoded_filtered[column] = 0

    # Reorder columns to match the training data
    input_encoded_final = input_encoded_filtered[model.feature_names_in_]

    # Predict
    predicted_price = model.predict(input_encoded_final)

    return render_template('index.html', manufacturers=manufacturers, models=models, min_year=min_year,
                           max_year=max_year, engine_sizes=engine_sizes, mpg_values=mpg_values, prediction=round(predicted_price[0], 2), year=request.form['year'], manufacturer=request.form['manufacturer'],
                       model=request.form['model'], mileage=request.form['mileage'])


if __name__ == "__main__":
    app.run(debug=True)
