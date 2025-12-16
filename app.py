import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# --- Configuration ---
# Your uploaded model file name
MODEL_PATH = 'placementmodel_file.pkl'

# The features required by the model (derived from the file metadata)
FEATURE_COLUMNS = [
    'IQ',
    'Prev_Sem_Result',
    'CGPA',
    'Academic_Performance',
    'Internship_Experience',
    'Extra_Curricular_Score',
    'Communication_Skills',
    'Projects_Completed'
]

# --- Load Model ---
# Load the pickled model file when the application starts
model = None
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file '{MODEL_PATH}' not found. Ensure it is in the same directory as app.py.")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")

# --- Flask App Initialization ---
app = Flask(__name__)

@app.route('/')
def home():
    """Simple status page."""
    return (
        "<h1>Placement Model Prediction API</h1>"
        "<p>Send a POST request to <code>/predict</code> with JSON data to get a placement prediction.</p>"
        f"<p><strong>Required features:</strong> {', '.join(FEATURE_COLUMNS)}</p>"
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handles POST requests for prediction."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs for details.'}), 500

    try:
        # Get JSON data from the request body
        data = request.get_json(force=True)

        # Check if all required features are present
        if not all(col in data for col in FEATURE_COLUMNS):
            return jsonify({
                'error': 'Missing one or more required features in the request data.',
                'required_features': FEATURE_COLUMNS,
                'received_keys': list(data.keys())
            }), 400

        # Prepare data for prediction
        # Create a dictionary where keys are feature names and values are the received data
        input_data = {col: [data[col]] for col in FEATURE_COLUMNS}
        
        # Convert to a pandas DataFrame, which scikit-learn models expect
        input_df = pd.DataFrame(input_data, columns=FEATURE_COLUMNS)

        # Make prediction and probability calculation
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df) # probability is an array like [[prob_0, prob_1]]

        # Assuming it's a binary classification (0: Not Placed, 1: Placed)
        result = {
            'prediction_value': int(prediction[0]), # The raw model output (0 or 1)
            'prediction_label': 'Placed' if prediction[0] == 1 else 'Not Placed',
            # Probabilities for class 0 (index 0) and class 1 (index 1)
            'probability_not_placed': round(probability[0][0], 4),
            'probability_placed': round(probability[0][1], 4)
        }

        return jsonify(result)

    except Exception as e:
        # Log the error and return a general error message
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the application
    # Set debug=False for production
    # Host '0.0.0.0' makes the server accessible externally (useful for containers/VMs)
    app.run(host='0.0.0.0', debug=True, port=5000)