import streamlit as st
import joblib
import numpy as np

# Load the trained model
model_filename = "RandomForestClassifier_best_model.joblib"
model = joblib.load(model_filename)

# Define the input fields based on the dataset (excluding removed features)
feature_names = [
    "Marital status", "Course", "Previous qualification", 
    "Previous qualification (grade)", "Nacionality", "Mother's qualification", "Father's qualification", "Admission grade", "Displaced", 
    "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", 
    "Age at enrollment", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", 
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)", 
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)", 
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)", 
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)", 
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)", 
    "Unemployment rate", "Inflation rate", "GDP"
]

# Streamlit UI
st.title("Student Dropout Prediction")
st.write("Enter the student details to predict the likelihood of dropout or graduation.")

# Collect user input
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    user_input.append(value)

# Convert input to NumPy array and reshape
input_data = np.array(user_input).reshape(1, -1)

# Predict on button click
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.write(f"**Predicted Outcome:** {prediction}")
