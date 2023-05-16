from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Python is running on Qoddi!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Preprocess the data and make your prediction
    # ...
    prediction = 0

    return jsonify({
        'prediction': prediction
    })

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
