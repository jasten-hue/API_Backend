import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

# Create output directory
os.makedirs("trained_data", exist_ok=True)

# Load dataset
df = pd.read_csv(r"C:\Users\marti\OneDrive\Desktop\New folder\API_Backend\csv\Students_Grading_Dataset.csv")

# Create Pass/Fail label from Grade
passing_grades = ["A", "B", "C"]
df['Pass_Fail'] = df['Grade'].apply(lambda x: 1 if x in passing_grades else 0)

# Columns used for prediction
features_to_use = [
    "Gender", "Age", "Department", "Attendance (%)", "Midterm_Score", "Final_Score",
    "Assignments_Avg", "Quizzes_Avg", "Participation_Score", "Projects_Score",
    "Total_Score", "Study_Hours_per_Week", "Extracurricular_Activities",
    "Internet_Access_at_Home", "Parent_Education_Level", "Family_Income_Level",
    "Stress_Level (1-10)", "Sleep_Hours_per_Night"
]

categorical_cols = [
    "Gender", "Department", "Extracurricular_Activities", "Internet_Access_at_Home",
    "Parent_Education_Level", "Family_Income_Level"
]

# Check for missing columns
missing_cols = [col for col in features_to_use + ['Pass_Fail'] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

# Balance the dataset
df_pass = df[df['Pass_Fail'] == 1]
df_fail = df[df['Pass_Fail'] == 0]
df_fail_upsampled = resample(df_fail, replace=True, n_samples=len(df_pass), random_state=42)
df_balanced = pd.concat([df_pass, df_fail_upsampled])

# Separate features and target
X = df_balanced[features_to_use]
y = df_balanced['Pass_Fail']

# Encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Save feature columns for prediction alignment
joblib.dump(X_encoded.columns.tolist(), "trained_data/pass_fail_features.pkl")
joblib.dump(categorical_cols, "trained_data/categorical_cols.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model with pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Save trained model
joblib.dump(pipeline, "trained_data/pass_fail_model.pkl")

print("✅ Model trained and saved successfully.")