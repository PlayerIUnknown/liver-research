# -------------------- app.py (FINAL VERSION with Bulk Upload) --------------------

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from sklearn.impute import SimpleImputer # Needed for bulk data cleaning

# Set page configuration
st.set_page_config(page_title="Liver Disease Prediction", layout="wide")

# --- Define Feature Order (MUST match the Colab training script) ---
FEATURE_ORDER = [
    'Age_of_the_patient', 'Gender_of_the_patient', 'Total_Bilirubin', 'Direct_Bilirubin',
    'Alkphos_Alkaline_Phosphotase', 'Sgpt_Alamine_Aminotransferase',
    'Sgot_Aspartate_Aminotransferase', 'Total_Protiens', 'ALB_Albumin',
    'AG_Ratio_Albumin_and_Globulin_Ratio'
]
# Mapping from raw column names (or approximate names) to the clean names
RAW_TO_CLEAN_MAP = {
    'Age of the patient': 'Age_of_the_patient',
    'Gender of the patient': 'Gender_of_the_patient',
    'Total Bilirubin': 'Total_Bilirubin',
    'Direct Bilirubin': 'Direct_Bilirubin',
    'Alkphos Alkaline Phosphotase': 'Alkphos_Alkaline_Phosphotase',
    'Sgpt Alamine Aminotransferase': 'Sgpt_Alamine_Aminotransferase',
    'Sgot Aspartate Aminotransferase': 'Sgot_Aspartate_Aminotransferase',
    'Total Protiens': 'Total_Protiens',
    'ALB Albumin': 'ALB_Albumin',
    'A/G Ratio Albumin and Globulin Ratio': 'AG_Ratio_Albumin_and_Globulin_Ratio',
    # Include possible corrupted names from the raw data as keys
    'Alkphos_Alkaline_Phosphotase': 'Alkphos_Alkaline_Phosphotase', # Handles cleanup from training script
    'Sgpt_Alamine_Aminotransferase': 'Sgpt_Alamine_Aminotransferase',
}

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        # Load the stable Scikit-learn model
        model = joblib.load('liver_prediction_model.pkl') 
        scaler = joblib.load('scaler.pkl')
        # Create a basic imputer for bulk processing (using mean strategy)
        imputer = SimpleImputer(strategy='mean')
        return model, scaler, imputer
    except Exception as e:
        st.error(f"Error loading assets: {e}. Ensure model.pkl and scaler.pkl are correct.")
        return None, None, None

model, scaler, imputer = load_assets()

# --- Prediction Function for Bulk Data ---
def bulk_predict(df_input, model, scaler, imputer):
    df = df_input.copy()
    
    # 1. Clean Column Names to match training order (CRITICAL STEP)
    # Use the known feature order for the columns, assuming the uploaded file has 10 columns in that order
    if len(df.columns) == 10:
        df.columns = FEATURE_ORDER
    else:
        st.error(f"Uploaded file must contain exactly 10 feature columns. Found {len(df.columns)}.")
        return None

    # 2. Preprocessing (Must match Colab steps)
    
    # Handle Gender Encoding and Missing values (Mode imputation for Gender)
    gender_col = 'Gender_of_the_patient'
    df[gender_col] = df[gender_col].fillna(df[gender_col].mode()[0])
    df[gender_col] = df[gender_col].map({'Female': 0, 'Male': 1})
    
    # Coerce to numeric and impute NaNs (Mean imputation for numerics)
    for col in FEATURE_ORDER:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df[FEATURE_ORDER] = imputer.fit_transform(df[FEATURE_ORDER])

    # 3. Scaling
    X_scaled = scaler.transform(df[FEATURE_ORDER])

    # 4. Prediction
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # 5. Output Creation
    df['Risk_Probability (%)'] = (y_pred_proba * 100).round(2)
    df['Prediction_Result'] = np.where(y_pred_proba >= 0.5, 'High Risk', 'Low Risk')
    
    return df[['Prediction_Result', 'Risk_Probability (%)'] + FEATURE_ORDER]


# --- Streamlit Interface ---

st.title('🩺 Liver Disease Prediction System')
st.markdown('A Multi-layer Perceptron (MLP) Neural Network predicts liver disease risk.')

if model is None:
    st.stop()

# --- Tabs for Interface Organization ---
tab1, tab2 = st.tabs(["Single Patient Prediction", "Bulk Data Prediction (.csv)"])

# ====================================================================
# TAB 1: Single Patient Prediction
# ====================================================================
with tab1:
    st.header('Enter Patient Clinical Parameters:')

    with st.form("single_prediction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider('1. Age of the patient', 1, 90, 40, key='input_age')
            gender = st.selectbox('2. Gender of the patient', ('Male', 'Female'), key='input_gender')
            total_bilirubin = st.number_input('3. Total Bilirubin (TB)', min_value=0.1, max_value=75.0, value=1.0, step=0.1, key='input_tb')
            direct_bilirubin = st.number_input('4. Direct Bilirubin (DB)', min_value=0.0, max_value=35.0, value=0.4, step=0.1, key='input_db')
            alp = st.number_input('5. Alkphos Alkaline Phosphotase (ALP)', min_value=50, max_value=5000, value=180, key='input_alp')
        
        with col2:
            sgpt = st.number_input('6. Sgpt Alamine Aminotransferase (ALT)', min_value=5, max_value=2000, value=25, key='input_sgpt')
            sgot = st.number_input('7. Sgot Aspartate Aminotransferase (AST)', min_value=5, max_value=2000, value=25, key='input_sgot')
            total_proteins = st.number_input('8. Total Protiens', min_value=2.0, max_value=10.0, value=7.0, step=0.1, key='input_tp')
            albumin = st.number_input('9. ALB Albumin', min_value=1.0, max_value=6.0, value=3.5, step=0.1, key='input_alb')
            ag_ratio = st.number_input('10. A/G Ratio Albumin and Globulin Ratio', min_value=0.1, max_value=4.0, value=1.0, step=0.01, key='input_agr')

        submitted = st.form_submit_button("Predict Liver Disease Risk")

    if submitted:
        gender_encoded = 1 if gender == 'Male' else 0
        
        input_data_list = [
            age, gender_encoded, total_bilirubin, direct_bilirubin, alp, sgpt,
            sgot, total_proteins, albumin, ag_ratio
        ]
        
        input_data = np.array([input_data_list])

        # Scale and Predict
        scaled_input = scaler.transform(input_data)
        prediction_proba = model.predict_proba(scaled_input)[0][1]
        
        if prediction_proba >= 0.5:
            risk, color, icon = 'High Risk', 'red', '⚠️'
        else:
            risk, color, icon = 'Low Risk', 'green', '✅'

        st.markdown("---")
        st.subheader(f"{icon} Prediction Result: {risk}")
        st.markdown(f"**Probability of Liver Disease:** <span style='color:{color}'>{prediction_proba*100:.2f}%</span>", unsafe_allow_html=True)
        st.info('Consult a medical professional for diagnosis.')

# ====================================================================
# TAB 2: Bulk Data Prediction
# ====================================================================
with tab2:
    st.header('Upload Test Data for Batch Analysis')
    st.warning("⚠️ **Crucial:** The CSV file must contain exactly 10 columns of features in the **same order** as the training data (Age, Gender, Total Bilirubin, etc.), and **must NOT** contain the 'Result' column.")
    
    uploaded_file = st.file_uploader("Upload your CSV file (e.g., test.csv)", type="csv")

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file, encoding='latin1', header=None)
            
            # If the header wasn't read correctly, the first row is data, so we proceed without header
            if len(df_uploaded.columns) == 10 or len(df_uploaded.columns) == 11:
                 # If 11 columns, assume header was read and drop it
                 df_uploaded = pd.read_csv(uploaded_file, encoding='latin1', header=None, skiprows=1) 
            
            # Run the prediction function
            df_results = bulk_predict(df_uploaded, model, scaler, imputer)
            
            if df_results is not None:
                st.success(f"Analysis complete! {len(df_results)} patient records processed.")
                
                # Display results table
                st.subheader("Prediction Results Table")
                st.dataframe(df_results)

                # Provide a download link for the results
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_data = convert_df_to_csv(df_results)
                st.download_button(
                    label="Download Full Prediction Results",
                    data=csv_data,
                    file_name='bulk_liver_predictions.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"An error occurred during file processing: {e}")
            st.code("Check if your CSV columns match the expected format (10 columns, no header row/no target column).")
