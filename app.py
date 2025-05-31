from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained model and metadata
model = joblib.load("trained_data/pass_fail_model.pkl")
feature_columns = joblib.load("trained_data/pass_fail_features.pkl")
categorical_cols = joblib.load("trained_data/categorical_cols.pkl")

# Expected raw input (before encoding)
expected_input_columns = [
    "Gender", "Age", "Department", "Attendance (%)", "Midterm_Score", "Final_Score",
    "Assignments_Avg", "Quizzes_Avg", "Participation_Score", "Projects_Score",
    "Total_Score", "Study_Hours_per_Week", "Extracurricular_Activities",
    "Internet_Access_at_Home", "Parent_Education_Level", "Family_Income_Level",
    "Stress_Level (1-10)", "Sleep_Hours_per_Night"
]

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Student Pass/Fail Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        input_data = request.get_json(force=True)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=expected_input_columns)

        # One-hot encode categorical features
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

        # Add any missing columns (from training)
        for col in feature_columns:
            if col not in input_encoded:
                input_encoded[col] = 0

        # Ensure correct column order
        input_encoded = input_encoded[feature_columns]

        # Predict
        prediction = model.predict(input_encoded)[0]
        prediction_label = "Pass" if prediction == 1 else "Fail"

        return jsonify({"prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)