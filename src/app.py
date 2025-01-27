from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/model.pkl')


def model_predict(features: np.array) -> np.array:
    prediction = model.predict(features)
    return prediction


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model_predict(features=features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
