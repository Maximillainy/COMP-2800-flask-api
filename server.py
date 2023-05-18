from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

price_model = joblib.load("./models/price_prediction.joblib")
manufacturer_encoder = joblib.load('./models/encoders/manufacturer_encoder.joblib')
model_encoder = joblib.load('./models/encoders/model_encoder.joblib')
condition_encoder = joblib.load('./models/encoders/condition_encoder.joblib')
title_status_encoder = joblib.load('./models/encoders/title_status_encoder.joblib')
paint_color_encoder = joblib.load('./models/encoders/paint_color_encoder.joblib')
imputer = joblib.load('./models/imputers/price_imputer.joblib')

app = Flask(__name__)


@app.route('/')
def home():
    return "Python is running!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_string = data['input']

    # Preprocess the data
    input_data = pd.DataFrame([input_string.split(",")],
                              columns=['year', 'manufacturer', 'model', 'condition', 'odometer', 'title_status',
                                       'paint_color', 'year_listed', 'month'])

    input_data['manufacturer'] = manufacturer_encoder.transform(input_data['manufacturer'].astype(str))
    input_data['model'] = model_encoder.transform(input_data['model'].astype(str))
    input_data['condition'] = condition_encoder.transform(input_data['condition'].astype(str))
    input_data['title_status'] = title_status_encoder.transform(input_data['title_status'].astype(str))
    input_data['paint_color'] = paint_color_encoder.transform(input_data['paint_color'].astype(str))

    data_i = imputer.transform(input_data)
    input_data = pd.DataFrame(data_i, columns=input_data.columns)

    prediction = price_model.predict(input_data)

    return jsonify({
        'prediction': prediction[0]
    })


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
