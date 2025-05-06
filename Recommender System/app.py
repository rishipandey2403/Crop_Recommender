import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
print("Templates exists:", os.path.exists('templates/index.html'))

app = Flask(__name__)

# Load and prepare data
crop = pd.read_csv('.venv/Crop_recommendation.csv')

# Create label mapping (same as your original)
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14,
    'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Convert crop labels to numbers
crop['label_num'] = crop['label'].map(crop_dict)

# Split data (keeping your original column names)
x = crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = crop['label_num']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train scaler and model
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
model = RandomForestClassifier(random_state=42)
model.fit(x_train_scaled, y_train)

# Save model and scaler (creates .pkl files automatically)
joblib.dump(model, '.venv/crop_model.pkl')
joblib.dump(sc, '.venv/scaler.pkl')

# Reverse mapping for predictions
reverse_crop_dict = {v: k for k, v in crop_dict.items()}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data (matching your HTML names exactly)
        features = np.array([[
            float(request.form['Nitrogen']),
            float(request.form['Phosporus']),  # Note: Your HTML spells it with 'o'
            float(request.form['Potassium']),
            float(request.form['Temperature']),
            float(request.form['Humidity']),
            float(request.form['Ph']),  # Matches name="Ph" in HTML
            float(request.form['Rainfall'])
        ]])

        # Scale and predict
        scaled_features = sc.transform(features)
        prediction = model.predict(scaled_features)
        crop_name = reverse_crop_dict.get(prediction[0], "Unknown Crop")

        return render_template('index.html', result=crop_name)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)