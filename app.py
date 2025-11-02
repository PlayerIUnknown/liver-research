# -------------------- app.py (Final .PKL Version) --------------------

import streamlit as st
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Liver Disease Prediction", layout="centered")

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        # Load the stable Scikit-learn model
        model = joblib.load('liver_prediction_model.pkl') 
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading assets: {e}. Ensure model.pkl and scaler.pkl are correct.")
        return None, None

model, scaler = load_assets()

st.title('🩺 Liver Disease Prediction System (MLP)')

if model is None:
    st.stop()

st.header('Enter Patient Clinical Parameters:')

# --- Input Widgets inside a Form with unique keys for stability ---
with st.form("prediction_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    # Inputs are named for readability but ordered in the list below.
    with col1:
        age = st.slider('1. Age of the patient', 1, 90, 40, key='input_age')
        gender = st.selectbox('2. Gender of the patient', ('Male', 'Female'), key='input_gender')
        total_bilirubin = st.number_input('3. Total Bilirubin (TB)', min_value=0.1, max_value=75.0, value=10.0, step=0.1, key='input_tb') # Changed default for testing
        direct_bilirubin = st.number_input('4. Direct Bilirubin (DB)', min_value=0.0, max_value=35.0, value=5.0, step=0.1, key='input_db') # Changed default for testing
        alp = st.number_input('5. Alkphos Alkaline Phosphotase (ALP)', min_value=50, max_value=5000, value=400, key='input_alp') # Changed default for testing
    
    with col2:
        sgpt = st.number_input('6. Sgpt Alamine Aminotransferase (ALT)', min_value=5, max_value=2000, value=150, key='input_sgpt') # Changed default for testing
        sgot = st.number_input('7. Sgot Aspartate Aminotransferase (AST)', min_value=5, max_value=2000, value=250, key='input_sgot') # Changed default for testing
        total_proteins = st.number_input('8. Total Protiens', min_value=2.0, max_value=10.0, value=6.0, step=0.1, key='input_tp') # Changed default for testing
        albumin = st.number_input('9. ALB Albumin', min_value=1.0, max_value=6.0, value=2.5, step=0.1, key='input_alb') # Changed default for testing
        ag_ratio = st.number_input('10. A/G Ratio Albumin and Globulin Ratio', min_value=0.1, max_value=4.0, value=0.7, step=0.01, key='input_agr') # Changed default for testing

    submitted = st.form_submit_button("Predict Liver Disease Risk")

# --- Prediction Logic ---
if submitted:
    
    gender_encoded = 1 if gender == 'Male' else 0

    # CRUCIAL: Order MUST MATCH the FEATURE_ORDER list used in training:
    input_data_list = [
        age, gender_encoded, total_bilirubin, direct_bilirubin, alp, sgpt,
        sgot, total_proteins, albumin, ag_ratio
    ]
    
    input_data = np.array([input_data_list])

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Get probability for class 1 (High Risk)
    prediction_proba = model.predict_proba(scaled_input)[0][1]
    
    if prediction_proba >= 0.5:
        risk = 'High Risk'
        color = 'red'
        icon = '⚠️'
        
    else:
        risk = 'Low Risk'
        color = 'green'
        icon = '✅'

    # Display Results
    st.markdown("---")
    st.subheader(f"{icon} Prediction Result: {risk}")
    
    st.markdown(f"**Probability of Liver Disease:** <span style='color:{color}'>{prediction_proba*100:.2f}%</span>", unsafe_allow_html=True)
    
    if risk == 'High Risk':
        st.warning('This model indicates a higher risk. Consult a medical professional for diagnosis.')
    else:
        st.success('The model indicates a lower risk. Always follow up with a doctor.')
