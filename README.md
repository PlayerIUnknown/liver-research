
-----

#  Liver Disease Prediction using Neural Network (MLP)

##  Project Overview

This project implements a binary classification model to predict the presence of liver disease in a patient based on clinical laboratory parameters. We use a Multi-Layer Perceptron (MLP) Neural Network, implemented via the stable `scikit-learn` library, and deploy the interactive prediction tool using **Streamlit**.

###  Live Application

**(Add your Streamlit Cloud URL here after deployment)**

-----

##  Project Structure

The deployment requires exactly these four files in the root directory of your repository:

| File Name | Description | Source |
| :--- | :--- | :--- |
| `app.py` | The main Streamlit application code for the web interface and prediction logic. | Local creation (provided) |
| `requirements.txt` | Lists all necessary Python dependencies (Streamlit, scikit-learn, joblib). | Local creation (provided) |
| `liver_prediction_model.pkl` | **Trained MLP Model** (includes weights and architecture). | Generated and downloaded from Colab. |
| `scaler.pkl` | **Trained StandardScaler** (necessary for normalizing new input data). | Generated and downloaded from Colab. |

-----

##  Phase 1: Model Training and Asset Creation (Colab Workflow)

The training and saving process was executed in a Google Colab notebook to ensure consistent environments and easy file management.

### 1\. Data Acquisition and Cleaning

  * **Dataset Source:** Indian Liver Patient Dataset (ILPD) via the uploaded CSV file (`Liver Patient Dataset (LPD)_train.csv`).
  * **Cleaning:** The dataset contained multiple issues that required robust preprocessing:
      * **Encoding Fix:** The file was loaded with `encoding='latin1'` to resolve `UnicodeDecodeError`.
      * **Column Mismatch:** Hidden characters (`Â\xa0`) and inconsistent spacing in column headers were removed.
      * **Missing Values:** Categorical missing values (Gender) were imputed using the **mode**, and numerical missing values were imputed using the **mean** via `SimpleImputer`.
      * **Target Encoding:** The target column (`Result`/`Is_Patient`) was mapped from `(1, 2)` to standard binary labels `(1, 0)`.

### 2\. Feature Scaling

  * **Method:** `StandardScaler` was fitted exclusively on the training data.
  * **Output:** The fitted `scaler.pkl` object was saved. This is **critical** because all new data (from the Streamlit form) must be scaled using the same mean and standard deviation learned from the training set.

### 3\. Model Creation

  * **Algorithm:** Scikit-learn's **MLPClassifier** (Multi-Layer Perceptron), chosen for its stability and functional equivalence to a simple feed-forward Neural Network without the large overhead of TensorFlow.
  * **Architecture:** A simple architecture was used: `(64, 32)` hidden layers with a ReLU activation function.
  * **Training:** The model was trained on the scaled training data.
  * **Output:** The trained model was saved as **`liver_prediction_model.pkl`** using the `joblib` library.

-----

##  Phase 2: Deployment and Local Running

### 1\. Local Setup

To run this application on your local machine, follow these terminal steps:

1.  **Clone the Repository:**

    ```bash
    git clone [YOUR_REPO_URL]
    cd [YOUR_REPO_NAME]
    ```

2.  **Create and Activate Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    # On Mac/Linux:
    source venv/bin/activate
    # On Windows (PowerShell):
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

    The application will open automatically in your browser at `http://localhost:8501`.

### 2\. Prediction Logic

The core prediction logic in `app.py` is structured to prevent the `nan%` error and stabilize the output:

1.  **Input Collection:** 10 clinical features are collected via Streamlit input widgets.
2.  **Order Enforcement:** The inputs are compiled into a NumPy array in the **exact same order** (`Age`, `Gender_encoded`, `Total_Bilirubin`, etc.) as the model was trained.
3.  **Scaling:** The input array is transformed using the loaded `scaler.pkl`.
4.  **Prediction:** The scaled input is passed to the `model.predict_proba()`, which returns the probability of belonging to the "High Risk" class (class 1).

### 3\. Streamlit Cloud Deployment

The application is hosted on Streamlit Community Cloud (or a similar PaaS like Heroku) for global access:

1.  **Commit:** Ensure the four final files (`app.py`, `requirements.txt`, `*.pkl` files) are pushed to your GitHub repository.
2.  **Connect:** Link the repository to your Streamlit Cloud account.
3.  **Deploy:** Streamlit Cloud reads the `requirements.txt`, installs the dependencies, and executes `app.py`.
