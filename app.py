# -------------------- app.py --------------------

import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(page_title="Liver Disease Prediction", layout="centered")

# --- Asset Loading (Crucial for performance) ---
# Use st.cache_resource to load the large model/scaler only once
@st.cache_resource
def load_assets():
    try:
        model = load_model('liver_prediction_model.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or Scaler files not found. Please ensure 'liver_prediction_model.h5' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_assets()

# --- Streamlit Interface ---
st.title('🩺 Liver Disease Prediction System')
st.markdown('A Multilayer Perceptron (MLP) Neural Network is used to predict liver disease risk based on patient clinical parameters.')

if model is None:
    st.stop()

st.header('Enter Patient Clinical Parameters:')

# Define the columns that need input based on the dataset
feature_names = [
    'Age of the patient', 'Gender of the patient', 'Total Bilirubin', 'Direct Bilirubin',
    'Alkphos Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase',
    'Sgot Aspartate Aminotransferase', 'Total Protiens', 'ALB Albumin',
    'A/G Ratio Albumin and Globulin Ratio'
]

# Create input widgets
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider('Age', 1, 90, 40)
        gender = st.selectbox('Gender', ('Male', 'Female'))
        total_bilirubin = st.number_input('Total Bilirubin (TB)', min_value=0.1, max_value=75.0, value=1.0, step=0.1)
        direct_bilirubin = st.number_input('Direct Bilirubin (DB)', min_value=0.0, max_value=35.0, value=0.4, step=0.1)
        alp = st.number_input('Alkphos Alkaline Phosphotase (ALP)', min_value=50, max_value=5000, value=180)
    
    with col2:
        sgpt = st.number_input('Sgpt Alamine Aminotransferase (ALT)', min_value=5, max_value=2000, value=25)
        sgot = st.number_input('Sgot Aspartate Aminotransferase (AST)', min_value=5, max_value=2000, value=25)
        total_proteins = st.number_input('Total Protiens', min_value=2.0, max_value=10.0, value=7.0, step=0.1)
        albumin = st.number_input('ALB Albumin', min_value=1.0, max_value=6.0, value=3.5, step=0.1)
        ag_ratio = st.number_input('A/G Ratio', min_value=0.1, max_value=4.0, value=1.0, step=0.01)

    submitted = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submitted:
    # 1. Prepare Input Data
    gender_encoded = 1 if gender == 'Male' else 0
    
    input_data = np.array([[
        age, gender_encoded, total_bilirubin, direct_bilirubin, alp, sgpt, 
        sgot, total_proteins, albumin, ag_ratio
    ]])

    # 2. Scale the input data using the saved scaler
    scaled_input = scaler.transform(input_data)

    # 3. Make the prediction
    prediction_proba = model.predict(scaled_input)[0][0]
    
    # Set a threshold (e.g., 0.5) to classify
    if prediction_proba >= 0.5:
        risk = 'High Risk'
        color = 'red'
        icon = '⚠️'
    else:
        risk = 'Low Risk'
        color = 'green'
        icon = '✅'

    # 4. Display Results
    st.markdown("---")
    st.subheader(f"{icon} Prediction Result: {risk}")
    st.markdown(f"**Probability of Liver Disease:** <span style='color:{color}'>{prediction_proba*100:.2f}%</span>", unsafe_allow_html=True)
    
    if risk == 'High Risk':
        st.warning('This indicates the model suggests a higher likelihood of liver disease. Please consult a medical professional.')
    else:
        st.success('The model suggests a lower likelihood of liver disease. Always follow up with a doctor for diagnosis.')

# ------------------------------------------------
