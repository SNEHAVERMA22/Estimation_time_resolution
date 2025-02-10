import pickle
import math
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



with open("pincode_encoder.pkl", "rb") as f:
    pincode_encoder = pickle.load(f)


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
        
        # Log input data for debugging
        print(f"Received data: {data}")

        # Convert input to DataFrame
        input_data = pd.DataFrame([data])
        
        print(f"Processed input data: {input_data}")
        
        # Ensure required fields exist
        required_features = ["category", "subcategory", "pincode"]
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400
            
        # Check available categories and subcategories from encoders
        print(f"Available categories in encoder: {category_encoder.classes_}")
        print(f"Available subcategories in encoder: {subcategory_encoder.classes_}")

        # Convert categorical data to numeric if necessary
        try:
            # Check and print if the category and subcategory exist in the encoder's classes
            if data["category"] not in category_encoder.classes_:
                raise ValueError(f"Category '{data['category']}' not found in encoder classes")
            if data["subcategory"] not in subcategory_encoder.classes_:
                raise ValueError(f"Subcategory '{data['subcategory']}' not found in encoder classes")

            category_num = category_encoder.transform([data["category"]])[0]
            subcategory_num = subcategory_encoder.transform([data["subcategory"]])[0]
        
            pincode_num = pincode_encoder.transform([data["pincode"]])[0]

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Replace original values with encoded ones
        input_data["category"] = category_num
        input_data["subcategory"] = subcategory_num
     
        input_data["pincode"] = pincode_num
        
        # Print the encoded values
        print(f"Category encoded: {category_num}, Subcategory encoded: {subcategory_num}")
        
        # Check if encoding has failed and return an error if so
        if category_num is None or subcategory_num is None  or pincode_num is None:
            return jsonify({"error": "Invalid category, subcategory or pincode"}), 400

        # Prepare the model input
        model_input = [[category_num, subcategory_num, int(data["pincode"])]]

        # Print the model input data for debugging
        print(f"Model input data: {model_input}")

        # Predict resolution time
        prediction = model.predict(np.array(input_data))[0]  
        
        predicted_days = math.ceil(prediction)  # Always rounds up


        return jsonify({"predicted_resolution_time": f"{predicted_days} days"})


    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    print(app.url_map)
    app.run(debug=True, port=5000)

