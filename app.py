# -------------------- app.py (Keras/TensorFlow Version) --------------------

import streamlit as st
import joblib
import numpy as np
# Note: TensorFlow/Keras must be installed correctly for load_model to work
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(page_title="Liver Disease Prediction", layout="centered")

# --- Asset Loading (Crucial for performance) ---
# Use st.cache_resource to load the large model/scaler only once
@st.cache_resource
def load_assets():
    try:
        # Load the Keras model (H5 format)
        model = load_model('liver_prediction_model.h5')
        # Load the saved StandardScaler
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or Scaler files not found. Please ensure 'liver_prediction_model.h5' and 'scaler.pkl' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading assets: {e}. Check TensorFlow installation or file integrity.")
        return None, None

model, scaler = load_assets()

# --- Streamlit Interface ---
st.title('Liver Disease Prediction System')
st.markdown('A Multilayer Perceptron (MLP) Neural Network predicts liver disease risk based on 10 clinical parameters.')

if model is None:
    st.stop()

st.header('Enter Patient Clinical Parameters:')

# Create input widgets
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    # ------------------------------------------------
    # Column 1 Inputs (Inputs 1-5)
    # The titles here are user-friendly, but the variables are strictly ordered below.
    with col1:
        age = st.slider('1. Age', 1, 90, 40)
        gender = st.selectbox('2. Gender', ('Male', 'Female'))
        total_bilirubin = st.number_input('3. Total Bilirubin (TB)', min_value=0.1, max_value=75.0, value=1.0, step=0.1)
        direct_bilirubin = st.number_input('4. Direct Bilirubin (DB)', min_value=0.0, max_value=35.0, value=0.4, step=0.1)
        alp = st.number_input('5. Alkphos Alkaline Phosphotase (ALP)', min_value=50, max_value=5000, value=180)
    
    # Column 2 Inputs (Inputs 6-10)
    with col2:
        sgpt = st.number_input('6. Sgpt Alamine Aminotransferase (ALT)', min_value=5, max_value=2000, value=25)
        sgot = st.number_input('7. Sgot Aspartate Aminotransferase (AST)', min_value=5, max_value=2000, value=25)
        total_proteins = st.number_input('8. Total Protiens', min_value=2.0, max_value=10.0, value=7.0, step=0.1)
        albumin = st.number_input('9. ALB Albumin', min_value=1.0, max_value=6.0, value=3.5, step=0.1)
        ag_ratio = st.number_input('10. A/G Ratio Albumin and Globulin Ratio', min_value=0.1, max_value=4.0, value=1.0, step=0.01)

    submitted = st.form_submit_button("Predict Liver Disease Risk")

# --- Prediction Logic ---
if submitted:
    
    # 1. Prepare Input Data: MUST MATCH THE 10 FEATURE ORDER USED DURING TRAINING
    
    # Encode Gender (Male=1, Female=0, based on training)
    gender_encoded = 1 if gender == 'Male' else 0

    # Create the list of inputs in the EXACT ORDER used by the scaler/model:
    input_data_list = [
        age,                 # 1. Age of the patient
        gender_encoded,      # 2. Gender of the patient (encoded)
        total_bilirubin,     # 3. Total Bilirubin
        direct_bilirubin,    # 4. Direct Bilirubin
        alp,                 # 5. Alkphos Alkaline Phosphotase
        sgpt,                # 6. Sgpt Alamine Aminotransferase
        sgot,                # 7. Sgot Aspartate Aminotransferase
        total_proteins,      # 8. Total Protiens
        albumin,             # 9. ALB Albumin
        ag_ratio             # 10. A/G Ratio Albumin and Globulin Ratio
    ]
    
    # Convert list to a NumPy array for the scaler (shape: 1 row, 10 columns)
    input_data = np.array([input_data_list])

    # 2. Scale the input data
    scaled_input = scaler.transform(input_data)

    # 3. Make the prediction
    # Keras predict returns a 2D array: [[probability]]
    prediction_proba = model.predict(scaled_input)[0][0]
    
    # 4. Interpret the result
    if prediction_proba >= 0.5:
        risk = 'High Risk'
        color = 'red'
        icon = '⚠️'
    else:
        risk = 'Low Risk'
        color = 'green'
        icon = '✅'

    # 5. Display Results
    st.markdown("---")
    st.subheader(f"{icon} Prediction Result: {risk}")
    
    # Display percentage
    st.markdown(f"**Probability of Liver Disease:** <span style='color:{color}'>{prediction_proba*100:.2f}%</span>", unsafe_allow_html=True)
    
    if risk == 'High Risk':
        st.warning('This model indicates a higher risk. Consult a medical professional for diagnosis.')
    else:
        st.success('The model indicates a lower risk. Always follow up with a doctor.')

# ------------------------------------------------

