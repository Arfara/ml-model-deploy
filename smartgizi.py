from flask import Flask, request, jsonify
import tensorflow as tf
from google.cloud import storage
#import os
from clustering_model import ClusteringModel
from recommendation_model import RecommendationModel

app = Flask(__name__)
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "submission-mgce-irfanal-bc0df925669e.json"

#load model
interpreter=tf.lite.Interpreter(model_path='models/model.tflite')
clustering_model = ClusteringModel('models/model_clustering.pickle')
recommendation_model = RecommendationModel('models/rekomendasi.pickle')
interpreter.allocate_tensors()

# Load model from Google Cloud Storage
# def load_model_from_gcs():
#     bucket_name = 'smartgizi_bucket_2'
#     model_filename = 'logistic_regression.tflite'

#     try:
#         client = storage.Client()
#         bucket = client.get_bucket(bucket_name)
#         blob = bucket.blob(model_filename)

#         # Save model to temporary local file
#         local_model_filename = '/tmp/logistic_regression.tflite'
#         blob.download_to_filename(local_model_filename)

#         # Load model from local file
#         interpreter = tf.lite.Interpreter(model_path=local_model_filename)
#         interpreter.allocate_tensors()

#         return interpreter

#     except Exception as e:
#         print(f'Error loading model from Google Cloud Storage: {e}')
#         return None

# interpreter = load_model_from_gcs()


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
    
@app.route('/clustering', methods=['POST'])
def clustering():
    data = request.json['data']
    # Lakukan inferensi clustering menggunakan model
    labels = clustering_model.perform_clustering(data)
    response = {'labels': labels}
    return jsonify(response)

@app.route('/recommendation', methods=['POST'])
def recommendation():
    data = request.json
    usia = float(data['usia'])
    berat_badan = float(data['berat_badan'])
    tinggi_badan = float(data['tinggi_badan'])
    trimester = float(data['trimester'])
    # Lakukan rekomendasi menggunakan model
    recommendations = recommendation_model.make_recommendations(tinggi_badan, berat_badan, usia, trimester)
    response = {'labels': data['labels'],'recommendations': recommendations}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
