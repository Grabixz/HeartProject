import streamlit as st
import numpy as np
import joblib

st.title("Heart Disease Prediction App")
st.write("This app predicts whether a patient is likely to have heart disease based on certain attributes.")

model = joblib.load('heart_disease_model.pkl')

age = st.slider("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.slider("Resting Blood Pressure (in mm Hg on admission to the hospital)", min_value=90, max_value=200, value=120)
chol = st.slider("Serum Cholesterol in mg/dl", min_value=100, max_value=600, value=200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.selectbox("Resting Electrocardiographic Results",
                       ["Normal", "Abnormal", "Ventricular hypertrophy"])
thalach = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("ST Depression induced by exercise relative to rest", min_value=0.0, max_value=6.2, value=0.0)
slope = st.selectbox("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])
ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
thal = st.selectbox("Status of the heart", ["Normal", "Fixed Defect", "Reversible Defect"])

if st.button('Predict'):
    sex = 1 if sex == "Male" else 0
    cp_dict = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    cp = cp_dict[cp]
    fbs = 1 if fbs == "True" else 0
    restecg_dict = {"Normal": 0, "Abnormal": 1, "Ventricular hypertrophy": 2}
    restecg = restecg_dict[restecg]
    exang = 1 if exang == "Yes" else 0
    slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope = slope_dict[slope]
    thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    thal = thal_dict[thal]

    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(features)
    if prediction == 0:
        st.write("Prediction: Patient is likely to be healthy.")
    elif prediction == 1:
        st.write("Prediction: Patient is likely to have heart disease.")
    else:
        st.write("Prediction: Unknown")
