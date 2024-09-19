import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Data preprocessing
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# App layout
st.title('Diabetes Prediction App')

# Gender input
gender = st.selectbox('Gender', ['Male', 'Female'])

# Dynamic input fields based on gender
if gender == 'Female':
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
else:
    pregnancies = 0  # Set pregnancies to 0 for males

# Other common inputs
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=80)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=600, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input('Age', min_value=1, max_value=120, value=30)

# Prediction button
if st.button('Predict'):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.write("The patient is likely to have diabetes.")
    else:
        st.write("The patient is unlikely to have diabetes.")

# Model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
