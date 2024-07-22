import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import base64
import plotly.express as px

# Load the trained model and scaler
model = pickle.load(open("models_kidney.pkl", "rb"))
sc = pickle.load(open("scaler.pkl", "rb"))

# Initialize session state
if "doctor_logged_in" not in st.session_state:
    st.session_state["doctor_logged_in"] = False
if "doctor_mail_id" not in st.session_state:
    st.session_state["doctor_mail_id"] = None

# Function for doctor login
def doctor_login():
    st.sidebar.subheader("Doctor Login")
    doctor_mail_id = st.sidebar.text_input("Doctor Mail ID", key="doctor_mail_id_login")
    doctor_name = st.sidebar.text_input("Doctor Name", key="doctor_name_login")
    if st.sidebar.button("Login"):
        conn = sqlite3.connect('kidney_disease.db')
        c = conn.cursor()
        c.execute('SELECT * FROM doctor WHERE doctor_mail_id=?', (doctor_mail_id,))
        data = c.fetchone()
        conn.close()
        if data and data[1] == doctor_name:
            st.sidebar.success(f"Welcome back, {data[1]}")
            st.session_state["doctor_logged_in"] = True
            st.session_state["doctor_mail_id"] = doctor_mail_id
        else:
            st.sidebar.error("Invalid doctor credentials")

# Function to add patient data to the database
def add_patient_data(patient_data):
    conn = sqlite3.connect('kidney_disease.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO patient (
            patient_mail_id, patient_name, doctor_mail_id, age, blood_pressure, specific_gravity,
            albumin, sugar, red_blood_cells, pus_cell, pus_cell_clumps, bacteria, blood_glucose_random,
            blood_urea, serum_creatinine, sodium, potassium, haemoglobin, packed_cell_volume,
            white_blood_cell_count, red_blood_cell_count, hypertension, diabetes_mellitus,
            coronary_artery_disease, appetite, peda_edema, aanemia
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', patient_data)
    conn.commit()
    conn.close()

# Main application logic
def main():
    # Doctor login logic
    if not st.session_state["doctor_logged_in"]:
        doctor_login()
    else:
        doctor_mail_id = st.session_state["doctor_mail_id"]
        st.sidebar.success(f"Logged in as {doctor_mail_id}")

        st.title("Kidney Disease Prediction")

        if st.sidebar.checkbox("Add New Patient"):
            # Collecting patient details with unique keys
            patient_mail_id = st.text_input("Patient Mail ID", key="patient_mail_id")
            patient_name = st.text_input("Patient Name", key="patient_name")
            age = st.number_input("Age", min_value=0, max_value=120, key="age_input")
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, key="blood_pressure_input")
            specific_gravity = st.number_input("Specific Gravity", min_value=1.000, max_value=1.040, step=0.001, key="specific_gravity_input")
            albumin = st.number_input("Albumin", min_value=0, max_value=5, key="albumin_input")
            sugar = st.number_input("Sugar", min_value=0, max_value=5, key="sugar_input")
            red_blood_cells = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"], key="red_blood_cells_input")
            pus_cell = st.selectbox("Pus Cell", ["Normal", "Abnormal"], key="pus_cell_input")
            pus_cell_clumps = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"], key="pus_cell_clumps_input")
            bacteria = st.selectbox("Bacteria", ["Not Present", "Present"], key="bacteria_input")
            blood_glucose_random = st.number_input("Blood Glucose Random", min_value=0, max_value=500, key="blood_glucose_random_input")
            blood_urea = st.number_input("Blood Urea", min_value=0, max_value=300, key="blood_urea_input")
            serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, max_value=15.0, step=0.1, key="serum_creatinine_input")
            sodium = st.number_input("Sodium", min_value=0, max_value=200, key="sodium_input")
            potassium = st.number_input("Potassium", min_value=0.0, max_value=10.0, step=0.1, key="potassium_input")
            haemoglobin = st.number_input("Haemoglobin", min_value=0.0, max_value=20.0, step=0.1, key="haemoglobin_input")
            packed_cell_volume = st.number_input("Packed Cell Volume", min_value=0, max_value=100, key="packed_cell_volume_input")
            white_blood_cell_count = st.number_input("White Blood Cell Count", min_value=0, max_value=20000, key="white_blood_cell_count_input")
            red_blood_cell_count = st.number_input("Red Blood Cell Count", min_value=0.0, max_value=10.0, step=0.1, key="red_blood_cell_count_input")
            hypertension = st.selectbox("Hypertension", ["No", "Yes"], key="hypertension_input")
            diabetes_mellitus = st.selectbox("Diabetes Mellitus", ["No", "Yes"], key="diabetes_mellitus_input")
            coronary_artery_disease = st.selectbox("Coronary Artery Disease", ["No", "Yes"], key="coronary_artery_disease_input")
            appetite = st.selectbox("Appetite", ["Good", "Poor"], key="appetite_input")
            peda_edema = st.selectbox("Pedal Edema", ["No", "Yes"], key="peda_edema_input")
            aanemia = st.selectbox("Anemia", ["No", "Yes"], key="aanemia_input")

            if st.button("Predict"):
                # Convert inputs to appropriate format
                red_blood_cells_input = red_blood_cells == "Abnormal"
                pus_cell_input = pus_cell == "Abnormal"
                pus_cell_clumps_input = pus_cell_clumps == "Present"
                bacteria_input = bacteria == "Present"
                hypertension_input = hypertension == "Yes"
                diabetes_mellitus_input = diabetes_mellitus == "Yes"
                coronary_artery_disease_input = coronary_artery_disease == "Yes"
                appetite_input = appetite == "Poor"
                peda_edema_input = peda_edema == "Yes"
                aanemia_input = aanemia == "Yes"

                inputs = np.array([[
                    age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells_input,
                    pus_cell_input, pus_cell_clumps_input, bacteria_input, blood_glucose_random,
                    blood_urea, serum_creatinine, sodium, potassium, haemoglobin, packed_cell_volume,
                    white_blood_cell_count, red_blood_cell_count, hypertension_input, diabetes_mellitus_input,
                    coronary_artery_disease_input, appetite_input, peda_edema_input, aanemia_input
                ]])

                # Scale inputs
                inputs = sc.transform(inputs)

                # Predict
                prediction = model.predict(inputs)
                prediction_proba = model.predict_proba(inputs)

                if prediction[0] == 0:
                    st.success(f"The patient is predicted to have chronic kidney disease with a probability of {prediction_proba[0][0]*100:.2f}%")
                else:
                    st.success(f"The patient is predicted to NOT have chronic kidney disease with a probability of {prediction_proba[0][1]*100:.2f}%")

                # Save patient data
                patient_data = (
                    patient_mail_id, patient_name, doctor_mail_id, age, blood_pressure, specific_gravity, albumin, sugar,
                    red_blood_cells_input, pus_cell_input, pus_cell_clumps_input, bacteria_input,
                    blood_glucose_random, blood_urea, serum_creatinine, sodium, potassium, haemoglobin, packed_cell_volume,
                    white_blood_cell_count, red_blood_cell_count, hypertension_input, diabetes_mellitus_input,
                    coronary_artery_disease_input, appetite_input, peda_edema_input, aanemia_input
                )
                add_patient_data(patient_data)

        # Display patient data
        if st.sidebar.checkbox("View My Patients"):
            conn = sqlite3.connect('kidney_disease.db')
            c = conn.cursor()
            c.execute('''
                SELECT p.patient_mail_id, p.patient_name, p.age, p.blood_pressure, p.specific_gravity, p.albumin,
                       p.sugar, p.red_blood_cells, p.pus_cell, p.pus_cell_clumps, p.bacteria, p.blood_glucose_random,
                       p.blood_urea, p.serum_creatinine, p.sodium, p.potassium, p.haemoglobin,
                       p.packed_cell_volume, p.white_blood_cell_count, p.red_blood_cell_count,
                       p.hypertension, p.diabetes_mellitus, p.coronary_artery_disease,
                       p.appetite, p.peda_edema, p.aanemia
                FROM patient p JOIN doctor d ON p.doctor_mail_id = d.doctor_mail_id
                WHERE p.doctor_mail_id = ?
            ''', (doctor_mail_id,))
            patients = c.fetchall()
            conn.close()

            if patients:
                st.subheader("Patient Data")
                df_patients = pd.DataFrame(patients, columns=['patient_mail_id', 'patient_name', 'age', 'blood_pressure',
                                                              'specific_gravity', 'albumin', 'sugar', 'red_blood_cells',
                                                              'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random',
                                                              'blood_urea', 'serum_creatinine', 'sodium', 'potassium', 'haemoglobin',
                                                              'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                                                              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
                                                              'appetite', 'peda_edema', 'aanemia'])
                st.dataframe(df_patients)
            else:
                st.write("No patient data available.")
        # Functionality to view decision tree plot
        if st.sidebar.checkbox("View Decision Tree Plot"):
            try:
                with open("decision_tree_plot.bin", "rb") as f:
                    image_base64 = f.read().decode('utf-8')
                image_data = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_data))
                st.subheader("Decision Tree Plot")
                st.image(image, caption='Decision Tree')
            except Exception as e:
                st.error(f"Error opening decision tree plot: {e}")

        # Functionality to view feature importances
        if st.sidebar.checkbox("Feature Importance"):
            st.subheader("Feature Importances")

            # Assuming input_df contains the feature names used during model training
            feature_names = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells',
                             'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea',
                             'serum_creatinine', 'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume',
                             'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
                             'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia']

            importance_values = model.feature_importances_

            # Create a DataFrame for feature importances
            importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance_values})
            importance_df = importance_df.sort_values(by="Importance", ascending=False)

            # Plot using Plotly Express
            fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importances')
            st.plotly_chart(fig)

    # Functionality to log out
    if st.sidebar.button("Logout"):
        st.sidebar.success("Logged out successfully.")
        st.session_state["doctor_logged_in"] = False
        st.session_state["doctor_mail_id"] = None
        st.experimental_rerun()

if __name__ == "__main__":
    main()

