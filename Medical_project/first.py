# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.utils import resample

# # Load the dataset
# data = pd.read_csv('diabetes.csv')
# df_minority=data[data['Outcome']==1]
# df_majority=data[data['Outcome']==0]

# minority_upsampled=resample(df_minority,replace=True, #Sample With replacement
#          n_samples=len(df_majority),
#          random_state=42
#         )

# # Data preprocessing
# X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
# y = data['Outcome']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # Scale the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Model training with increased iterations to prevent convergence warnings
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_scaled, y_train)

# # App layout
# st.title('Diabetes Prediction App')

# # Gender input
# gender = st.selectbox('Gender', ['Male', 'Female'])

# # Dynamic input fields based on gender
# if gender == 'Female':
#     pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
# else:
#     pregnancies = 0  # Set pregnancies to 0 for males

# # Other common inputs
# glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
# blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=80)
# skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
# insulin = st.number_input('Insulin', min_value=0, max_value=600, value=80)
# bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
# dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
# age = st.number_input('Age', min_value=1, max_value=120, value=30)

# # Prediction button
# if st.button('Predict'):
#     # Scale the input data
#     input_data = scaler.transform(np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]))
#     prediction = model.predict(input_data)
    
#     if prediction[0] == 1:
#         st.write("The patient is likely to have diabetes.")
#     else:
#         st.write("The patient is unlikely to have diabetes.")

# # Model accuracy
# accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
# st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Separate the majority and minority classes
df_minority = data[data['Outcome'] == 1]
df_majority = data[data['Outcome'] == 0]

# Upsample minority class
df_minority_upsampled = resample(df_minority, replace=True,  # Sample with replacement
                                 n_samples=len(df_majority),  # Match majority class
                                 random_state=42)

# Combine majority class with upsampled minority class
data_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Data preprocessing
X = data_upsampled[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data_upsampled['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM model with default parameters
model = SVC()  # Default parameters
model.fit(X_train_scaled, y_train)

# Model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

# Streamlit app layout
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
    # Scale the input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    
    if prediction[0] == 1:
        st.write("The patient is likely to have diabetes.")
    else:
        st.write("The patient is unlikely to have diabetes.")

# Display model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
