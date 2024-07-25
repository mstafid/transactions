from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model terbaik
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return "Model Deployment using Flask and Vercel.com"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)  # Mengubah input menjadi array 2D
    prediction = model.predict(features)
    output = prediction[0]
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
