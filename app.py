from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Muat model dan scaler
model = load_model('lstm_model.keras')

with open('scaler.pkl', 'rb') as file:
    target_scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    tahun = float(data['tahun'])
    luas_panen = float(data['luas_panen'])
    curah_hujan = float(data['curah_hujan'])
    kelembapan = float(data['kelembapan'])
    suhu_rata_rata = float(data['suhu_rata_rata'])

    # Preprocessing input
    input_data = np.array([[tahun, luas_panen, curah_hujan, kelembapan, suhu_rata_rata]])

    # Periksa bentuk data sebelum transformasi
    print(f"Bentuk data sebelum transformasi: {input_data.shape}")

    input_data_scaled = target_scaler.transform(input_data)

    # Periksa bentuk data setelah transformasi
    print(f"Bentuk data setelah transformasi: {input_data_scaled.shape}")

    # Reshape untuk model
    input_data_reshaped = input_data_scaled.reshape((input_data_scaled.shape[0], input_data_scaled.shape[1], 1))

    # Prediksi
    prediction = model.predict(input_data_reshaped)

    # Inverse transform hasil prediksi
    prediction_inverse = target_scaler.inverse_transform(prediction)

    return jsonify({'prediction': prediction_inverse[0, 0]})

if __name__ == '__main__':
    app.run(debug=True)
