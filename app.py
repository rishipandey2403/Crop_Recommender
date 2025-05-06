from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')

# Load model and scaler
model = joblib.load('crop_model.pkl')
scaler = joblib.load('scaler.pkl')

# Crop dictionary
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14,
    'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
reverse_crop_dict = {v: k for k, v in crop_dict.items()}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [
            float(request.form['Nitrogen']),
            float(request.form['Phosporus']),
            float(request.form['Potassium']),
            float(request.form['Temperature']),
            float(request.form['Humidity']),
            float(request.form['Ph']),
            float(request.form['Rainfall'])
        ]

        # Scale features and predict
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        crop_name = reverse_crop_dict.get(prediction, "Unknown Crop")

        return jsonify({
            'status': 'success',
            'result': crop_name
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
