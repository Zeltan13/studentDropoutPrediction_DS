import streamlit as st
import numpy as np
import joblib

#Load model, label encoder, and top features
model = joblib.load('random_forest_model.pkl')
le_status = joblib.load('random_forest_label_encoder.pkl')
top_features = joblib.load('model_features.pkl')

#Page configuration
st.set_page_config(page_title="Student Dropout Prediction", layout="centered")
st.title("Student Dropout Prediction Prototype")
st.markdown("""
This prototype helps predict whether a student is likely to **drop out**, **stay enrolled**, or **graduate** 
based on their initial academic performance and enrollment profile.
Please fill in the fields below to get a prediction.
""")

#Input
with st.form("prediction_form"):
    st.subheader("Student Information")

    col1, col2 = st.columns(2)

    with col1:
        curr_2nd_grade = st.slider("2nd Semester Grade (0-20)", 0.0, 18.6, 10.2)
        curr_2nd_approved = st.slider("2nd Semester Approved Units", 0, 20, 4)
        curr_1st_grade = st.slider("1st Semester Grade (0-20)", 0.0, 18.9, 10.6)
        curr_1st_approved = st.slider("1st Semester Approved Units", 0, 26, 5)
        age = st.slider("Age at Enrollment", 17, 70, 23)

    with col2:
        tuition = st.radio("Tuition Fees Up-to-Date", ["Yes", "No"], index=0)
        scholarship = st.radio("Scholarship Holder", ["Yes", "No"], index=1)
        debtor = st.radio("Has Outstanding Debt", ["Yes", "No"], index=1)
        gender = st.radio("Gender", ["Male", "Female"], index=0)
        app_mode_dict = {
            "1st phase - general": 1,
            "2nd phase - general": 17,
            "3rd phase - general": 18,
            "Transfer": 42,
            "Change of course": 43,
            "International": 15,
            "Other": 99
        }
        app_mode_label = st.selectbox("Application Mode", list(app_mode_dict.keys()))

    submitted = st.form_submit_button("Predict")

#On submit
if submitted:
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    features = np.array([
        curr_2nd_grade,
        curr_2nd_approved,
        curr_1st_grade,
        curr_1st_approved,
        binary_map[tuition],
        age,
        binary_map[scholarship],
        binary_map[debtor],
        binary_map[gender],
        app_mode_dict[app_mode_label]
    ]).reshape(1, -1)

    #Prediction
    pred_encoded = model.predict(features)[0]
    pred_label = le_status.inverse_transform([pred_encoded])[0]
    st.success(f"Predicted Student Status: **{pred_label}**")
