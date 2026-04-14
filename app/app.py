from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load artifacts
model  = joblib.load("../models/stacking_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
ohe    = joblib.load("../models/ohe_encoder.pkl")
le2    = joblib.load("../models/education_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Match the 11 numerical columns in model.py
        num_fields = [
            "Applicant_Income", "Coapplicant_Income", "Age", "Dependents", 
            "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings", 
            "Collateral_Value", "Loan_Amount", "Loan_Term"
        ]

        cat_cols = [
            "Employment_Status", "Marital_Status", "Loan_Purpose",
            "Property_Area", "Gender", "Employer_Category"
        ]

        # Process inputs
        num_values = [[float(data[f]) for f in num_fields]]
        edu_encoded = le2.transform([data["Education_Level"]])[0]
        ohe_values = ohe.transform([[data[c] for c in cat_cols]])

        # Combine in order: Numerical (11) + Education (1) + OHE (15) = 27
        final_input = np.hstack([num_values, [[edu_encoded]], ohe_values])
        
        # Predict
        final_scaled = scaler.transform(final_input)
        prediction = model.predict(final_scaled)[0]
        probability = model.predict_proba(final_scaled)[0][1]

        return jsonify({
            "approved": bool(prediction),
            "confidence": round(float(probability) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)