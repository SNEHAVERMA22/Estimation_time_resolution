import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import xgboost as xgb

# Load trained XGBoost model
model_path = "xgboost_model.pkl"  # Update with actual model path
with open(model_path, "rb") as f:
    model = pickle.load(f)
    
# Load category encoders
with open("category_encoder.pkl", "rb") as f:
    category_encoder = pickle.load(f)

with open("subcategory_encoder.pkl", "rb") as f:
    subcategory_encoder = pickle.load(f)
    
with open("reported_year_encoder.pkl", "rb") as f:
    reported_year_encoder = pickle.load(f)

# Define Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Complaint Resolution Time Predictor API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Ensure required fields exist
        required_features = ["category", "subcategory", "reported_year"]  # Update with actual categorical feature names
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Convert input to DataFrame
        input_data = pd.DataFrame([data])
        
        print("Category:", data["category"])
        print("Subcategory:", data["subcategory"])


        # Convert categorical data to numeric if necessary
        try:
            category_num = category_encoder.transform([data["category"]])[0]
            subcategory_num = subcategory_encoder.transform([data["subcategory"]])[0]
            reported_year_num = reported_year_encoder.transform([data["reported_year"]])[0]
        except ValueError:
            return jsonify({"error": "Unknown category or sub-category"}), 400
        
        # Replace original values with encoded ones
        input_data["category"] = category_num
        input_data["subcategory"] = subcategory_num
        input_data["reported_year"] = reported_year_num
        # Predict resolution time
        prediction = model.predict(xgb.DMatrix(input_data))[0]  # XGBoost expects DMatrix

        return jsonify({"predicted_resolution_time": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    print(app.url_map)

    app.run(debug=True,port=5000)
