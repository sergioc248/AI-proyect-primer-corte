import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Diagnóstico de PCOS", page_icon="🩺", layout="centered")

# CSS para un estilo elegante y médico
st.markdown("""
    <style>
    .big-font {
        font-size: 36px !important;
        text-align: center;
        color: #2c3e50;
    }
    .sub-font {
        font-size: 20px !important;
        text-align: center;
    }
    .result-green {
        font-size: 24px !important;
        color: green;
        text-align: center;
    }
    .result-red {
        font-size: 24px !important;
        color: red;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Título principal
st.markdown('<p class="big-font">Diagnóstico de PCOS</p>', unsafe_allow_html=True)

# Mostrar imagen centrada desde el enlace proporcionado
image_url = "https://ferticity.com/wp-content/uploads/2025/01/pcos_symtoms3-webp.webp"
st.image(image_url, use_container_width=True)

st.markdown('<p class="sub-font">Complete los siguientes campos para obtener el diagnóstico:</p>', unsafe_allow_html=True)

# Distribuir los inputs en dos columnas
col1, col2 = st.columns(2)

with col1:
    edad = st.slider("Edad (años)", min_value=18, max_value=50, value=30, step=1)
    bmi = st.slider("Índice de Masa Corporal (BMI)", min_value=18.0, max_value=40.0, value=25.0, step=0.1)
    irregularidad_menstrual = st.radio("¿Presenta irregularidad menstrual?", options=["Sí", "No"])
    
with col2:
    testosterona = st.slider("Nivel de Testosterona", min_value=20.0, max_value=100.0, value=50.0, step=0.1)
    foliculos = st.slider("Recuento de folículos antrales", min_value=5, max_value=30, value=15, step=1)

# Convertir la respuesta de irregularidad menstrual a valor numérico (1: Sí, 0: No)
irregularidad_valor = 1 if irregularidad_menstrual == "Sí" else 0

# Botón para predecir el diagnóstico
if st.button("Predecir Diagnóstico"):
    # Cargar el modelo y el escalador
    model = joblib.load("svm_selected_model.pkl")
    scaler = joblib.load("scaler.bin")
    
    input_df = pd.DataFrame({
        "Age": [edad],
        "BMI": [bmi],
        "Menstrual_Irregularity": [irregularidad_valor],
        "Testosterone_Level": [testosterona],
        "Antral_Follicle_Count": [foliculos]
    })
    
    # Escalar los datos de entrada
    input_data_scaled = scaler.transform(input_df)
    
    # Realizar la predicción
    prediction = model.predict(input_data_scaled)
    
    # Mostrar el resultado: verde si no se diagnostica PCOS, rojo si se diagnostica PCOS
    if prediction[0] == 0:
        st.markdown('<p class="result-green">Diagnóstico: No PCOS</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="result-red">Diagnóstico: PCOS</p>', unsafe_allow_html=True)
