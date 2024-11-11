import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

model = joblib.load('model.pkl')

"""### FLASK APP"""

# SERVER
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

from flask import Flask, request, jsonify
import pandas as pd

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print('data', data)
    required_fields = [
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
        'Eccentricity', 'Extent', 'Solidity', 'roundness', 'Compactness', 
        'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor4'
    ]

    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Invalid input data, missing required fields'}), 400

    input_data = [[data[field] for field in required_fields]]
    input_df = pd.DataFrame(input_data, columns=required_fields)

    input_df = input_df.replace(',', '.', regex=True).astype(float)

    try:
        prediction = model.predict(input_df)
        result = prediction[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'result': result})

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
