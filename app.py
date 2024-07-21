from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Muat model dan scaler
model_path = 'lstm_model.keras'
input_scaler_path = 'input_scaler.pkl'
target_scaler_path = 'target_scaler.pkl'

try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open(input_scaler_path, 'rb') as file:
        input_scaler = pickle.load(file)
    print("Input scaler loaded successfully.")
except Exception as e:
    print(f"Error loading input scaler: {e}")

try:
    with open(target_scaler_path, 'rb') as file:
        target_scaler = pickle.load(file)
    print("Target scaler loaded successfully.")
except Exception as e:
    print(f"Error loading target scaler: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        # Validasi input
        required_fields = ['tahun', 'luas_panen', 'curah_hujan', 'kelembapan', 'suhu_rata_rata']
        for field in required_fields:
            if field not in data or data[field].strip() == '':
                return jsonify({'error': f'Field {field} is required and cannot be empty'}), 400

        tahun = float(data['tahun'])
        luas_panen = float(data['luas_panen'])
        curah_hujan = float(data['curah_hujan'])
        kelembapan = float(data['kelembapan'])
        suhu_rata_rata = float(data['suhu_rata_rata'])

        # Preprocessing input
        input_data = np.array([[tahun, luas_panen, curah_hujan, kelembapan, suhu_rata_rata]])
        print(f"Original input data: {input_data}")

        # Periksa bentuk data sebelum transformasi
        print(f"Bentuk data sebelum transformasi: {input_data.shape}")

        input_data_scaled = input_scaler.transform(input_data)
        print(f"Scaled input data: {input_data_scaled}")

        # Periksa bentuk data setelah transformasi
        print(f"Bentuk data setelah transformasi: {input_data_scaled.shape}")

        # Reshape untuk model
        input_data_reshaped = input_data_scaled.reshape((input_data_scaled.shape[0], input_data_scaled.shape[1], 1))
        print(f"Reshaped input data: {input_data_reshaped}")

        # Prediksi
        prediction = model.predict(input_data_reshaped)
        print(f"Raw prediction: {prediction}")

        # Inverse transform hasil prediksi
        prediction_inverse = target_scaler.inverse_transform(prediction)
        print(f"Inversed prediction: {prediction_inverse}")

        # Konversi hasil prediksi ke tipe float untuk JSON
        prediction_result = float(prediction_inverse[0, 0])
        print(f"Prediction Result: {prediction_result}")

        return jsonify({'prediction': prediction_result})
    
    except ValueError as ve:
        print(f"Value error: {ve}")
        return jsonify({'error': f"Value error: {ve}"}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
