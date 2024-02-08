import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Loading the trained Support Vector Machine (SVM) model
with open('creavins.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('creavins.html')

@app.route('/predict', methods=['POST'])
def predict():
    lender_count = int(request.form['lender_count'])
    repayment_term = int(request.form['repayment_term'])
    sector = request.form['sector']
    location_country_code = request.form['location_country_code']

    # Converting sector and location_country_code to one-hot encoded format
    sectors = ['Arts', 'Clothing', 'Construction', 'Education', 'Food', 'Health', 'Housing', 'Manufacturing', 'Personal Use', 'Retail', 'Services', 'Transportation', 'Wholesale']
    location_country_codes = ['BI', 'BJ', 'BW', 'CD', 'CG', 'CI', 'CM', 'EG', 'GH', 'KE', 'LR', 'LS', 'MG', 'ML', 'MR', 'MW', 'MZ', 'NG', 'RW', 'SL', 'SN', 'SO', 'SS', 'TG', 'TZ', 'UG', 'ZA', 'ZM', 'ZW']

    sector_values = [1 if sector == i else 0 for i in sectors]
    location_country_code_values = [1 if location_country_code == i else 0 for i in location_country_codes]

    # Combining the features into a single array
    my_features = [lender_count, repayment_term] + sector_values + location_country_code_values + [0, 0]

    features = [np.array(my_features)]

    # Making the prediction using the model
    prediction = model.predict(features)
    output = round(prediction[0], 2)

    return render_template('creavins.html', prediction_text=f'Predicted Loan Amount: ${output}')

if __name__ == "__main__":
    app.run(debug=True)



