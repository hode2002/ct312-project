import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

from data_transformation import scaler

# Để tải lại mô hình
model = joblib.load('model.pkl')

"""### FLASK APP"""

# SERVER
app = Flask(__name__)
cors = CORS(app, resources={
    r"/*": {"origins": "*"}
})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    Area = data['Area']
    Perimeter = data['Perimeter']
    MajorAxisLength = data['MajorAxisLength']
    MinorAxisLength = data['MinorAxisLength']
    AspectRation = data['AspectRation']
    Eccentricity = data['Eccentricity']
    ConvexArea = data['ConvexArea']
    EquivDiameter = data['EquivDiameter']
    Extent = data['Extent']
    Solidity = data['Solidity']
    roundness = data['roundness']
    Compactness = data['Compactness']
    ShapeFactor1 = data['ShapeFactor1']
    ShapeFactor2 = data['ShapeFactor2']
    ShapeFactor3 = data['ShapeFactor3']
    ShapeFactor4 = data['ShapeFactor4']

    # Create a DataFrame for the input data
    input_data = [
        [Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation,
        Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity,
        roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4]
    ]

    columns = [
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
        'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
        'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
    ]

    # Convert input_data to a DataFrame
    input_df = pd.DataFrame(input_data, columns=columns)

    # Replace ',' with '.' in the DataFrame and convert to float
    input_df = input_df.replace(',', '.', regex=True).astype(float)

    # Scale the input data
    scaled_data = scaler.transform(input_df)

    np.set_printoptions(suppress=True)

    df = pd.DataFrame(scaled_data, columns=columns)

    # Make prediction
    prediction = model.predict(df)

    print(prediction)

    result = prediction[0]

    return jsonify({'result': result})

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)