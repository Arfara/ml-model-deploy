from flask import Flask, request, jsonify
import tensorflow as tf
from model2 import load_model, get_recommendations
from data import df

app = Flask(__name__)

#load model
interpreter=tf.lite.Interpreter(model_path='models/model.tflite')
interpreter.allocate_tensors()

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json

    # Validate data fields
    if 'usia' not in data or 'berat_badan' not in data or 'tinggi_badan' not in data:
        return jsonify({'error': 'Invalid data fields'}), 400

    try:
        # Extract data
        usia = float(data['usia'])
        berat_badan = float(data['berat_badan'])
        tinggi_badan = float(data['tinggi_badan'])

        # Perform prediction
        input_details = interpreter.get_input_details()
        input_data = [[usia, berat_badan, tinggi_badan]]
        input_data = tf.constant(input_data, dtype=tf.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        output = interpreter.get_tensor(output_details[0]['index'])
        result = {'prediction': int(output[0])}

        return jsonify(result)

    except Exception as e:
        print(f'Error during prediction: {e}')
        return jsonify({'error': 'Prediction error'}), 500

model = load_model()

@app.route('/recommend', methods=['POST'])
def recommend_food():
    # Mendapatkan data dari permintaan POST
    req_data = request.get_json()

    # Memeriksa apakah data yang diperlukan ada
    if 'name' not in req_data:
        return jsonify({'error': 'Data tidak lengkap'}), 400

    # Mengambil data makanan dari permintaan
    name = req_data['name']

    # Melakukan rekomendasi menggunakan fungsi dari file model
    food_indices = get_recommendations(name, model)

    # Mengambil nama makanan berdasarkan indeks yang direkomendasikan
    recommended_food = [df['Name'].iloc[idx] for idx in food_indices]

    # Mengirimkan rekomendasi makanan sebagai respons
    return jsonify({'recommendation': recommended_food})


if _name_ == '_main_':
    app.run(host='0.0.0.0', port=8080)
