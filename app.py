import streamlit as st
import joblib
import numpy as np

# Load the trained model
model_filename = "RandomForestClassifier_best_model.joblib"
model = joblib.load(model_filename)

feature_names =  [
    "Admission grade", "Previous qualification (grade)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)", 
    "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (approved)", 
    "Curricular units 2nd sem (grade)", "Tuition fees up to date", "Scholarship holder", 
    "Age at enrollment"
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
