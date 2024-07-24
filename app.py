from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the trained model
model = load_model('lstm_model.keras')

# Load the scalers
with open('input_scaler.pkl', 'rb') as f:
    input_scaler = pickle.load(f)

with open('target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)

# Function to preprocess input data
def preprocess_input(data):
    data = np.array(data).reshape(1, -1)
    data_scaled = input_scaler.transform(data)
    data_reshaped = data_scaled.reshape((data_scaled.shape[0], data_scaled.shape[1], 1))
    return data_reshaped

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = [float(x) for x in request.form.values()]
        
        # Preprocess the input data
        preprocessed_data = preprocess_input(data)
        
        # Predict using the model
        prediction = model.predict(preprocessed_data)
        
        # Inverse transform the prediction
        prediction_original = target_scaler.inverse_transform(prediction)
        
        # Return the result
        return render_template('index.html', prediction_text=f'Predicted Production: {prediction_original[0][0]:.2f}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)