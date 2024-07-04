import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Load model and scaler (replace with your actual file paths)
logging.info("importing pickle file")
ridge_model = pickle.load(open('model/linreg.pkl', 'rb'))
standard_scaler = pickle.load(open('model/scaler.pkl', 'rb'))

logging.info("imported pickle file done")
# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST', 'OPTIONS'])  # Allow both POST and OPTIONS requests
def predict_insurance_cost():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'CORS pre-flight request successful'}), 200

    try:
        # Data extraction and validation
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data found in request'}), 400

        age = int(data.get('age'))  # Handle potential missing values
        bmi = float(data.get('bmi'))  # Assign default 0.0 for missing BMI
        children = int(data.get('children'))
        smoker = data.get('smoker').lower()  # Convert smoker to lowercase
        print(age,bmi,children,smoker)
        if smoker not in ('yes', 'no'):
            return jsonify({'error': 'Invalid smoker value. Enter "Yes" or "No".'}), 400

        region = int(data.get('region'))

        logging.info(f"Received data: age={age}, bmi={bmi}, children={children}, smoker={smoker}, region={region}")

        # Preprocess data using the loaded scaler if necessary
        # ... (replace with your code for data preprocessing)

        # Make prediction
        predict = ridge_model.predict([[age, bmi, children,1 if smoker =='yes' else 0, region]]).tolist()

        response = jsonify({
            'estimated_cost': predict[0]
        })
        return response

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(debug=True)
