import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

"""### FLASK APP"""

# SERVER
app = Flask(__name__)
cors = CORS(app, resources={
    r"/*": {"origins": "*"}
})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    required_fields = [
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
        'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
        'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
    ]

    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Invalid input data'}), 400
    
    input_data = [[data[field] for field in required_fields]]
    input_df = pd.DataFrame(input_data, columns=required_fields)
    
    input_df = input_df.replace(',', '.', regex=True).astype(float)

    scaled_data = scaler.transform(input_df)

    prediction = model.predict(scaled_data)
    result = prediction[0]

    return jsonify({'result': result})


@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)