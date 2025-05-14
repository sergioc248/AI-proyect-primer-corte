import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Diagnóstico de PCOS", page_icon="🩺", layout="centered")

# CSS para un estilo elegante y médico
st.markdown("""
    <style>
    body {
        background-color: #e9e9e9; /* Page background */
        font-family: 'Montserrat', sans-serif !important; /* Modern font, strong override */
        font-size: 17px; /* Increased base font size for the whole app */
    }

    /* Styles Streamlit's main content area to look like a card */
    .block-container { 
        background-color: #fdfaf6; /* Creamy white card background */
        padding: 30px !important;
        margin: 30px auto !important;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        max-width: 800px; 
    }

    /* Styling for widget labels (e.g., for sliders, radio buttons) */
    .block-container label {
        font-family: 'Montserrat', sans-serif !important; /* Ensure Montserrat for labels */
        font-size: 18px !important; /* Make widget labels clearly legible */
        color: #333; /* Slightly darker color for better contrast if needed */
        margin-bottom: 8px !important; /* Add a bit of space below labels */
        display: block; /* Ensure proper spacing */
    }

    .big-font {
        font-size: 40px !important; /* Increased size */
        text-align: center;
        color: #1abc9c; /* Teal color */
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700; /* Bold */
        margin-bottom: 25px; /* Space below title */
    }

    .sub-font {
        font-size: 20px !important; 
        text-align: center;
        color: #16a085; /* Darker teal */
        font-family: 'Montserrat', sans-serif !important;
        margin-bottom: 30px; 
    }

    .pcos-recommendation-text {
        font-family: 'Montserrat', sans-serif !important;
        font-size: 20px !important; /* Consistent with sub-font */
        text-align: center;
        color: #e74c3c !important; /* Red color, same as .result-red text */
        margin-bottom: 30px; /* Consistent with sub-font */
    }

    /* Styling for the Streamlit button */
    .stButton button {
        background-color: #1abc9c; /* Teal */
        color: white;
        padding: 12px 25px;
        border: none;
        border-radius: 8px;
        font-size: 20px; 
        font-family: 'Montserrat', sans-serif !important;
        font-weight: bold;
        display: block; 
        margin: 30px auto; 
        cursor: pointer;
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out, box-shadow 0.2s ease-in-out;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton button:hover {
        background-color: #17a790; /* Slightly darker teal for hover */
        transform: translateY(-1px); 
        box-shadow: 0 4px 8px rgba(0,0,0,0.15); /* Enhanced shadow on hover */
    }

    .stButton button:active {
        background-color: #148f77; /* Darker teal for active state */
        transform: translateY(0px); /* Reset transform for active state */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Reset shadow for active state */
    }

    /* Adjustments for input widgets */
    .stSlider, .stRadio { 
        margin-bottom: 20px; /* Increased spacing for sliders and radio buttons */
        font-family: 'Montserrat', sans-serif !important;
    }
    
    /* Styling for result messages (diagnosis) */
    .result-green {
        font-size: 24px !important; /* Increased size */
        color: #27ae60; 
        text-align: center;
        border: 1px solid #2ecc71; 
        font-family: 'Montserrat', sans-serif !important;
        padding: 15px;
        border-radius: 10px;
        background-color: #eafaf1;
        margin-top: 20px;
    }
    .result-red {
        font-size: 24px !important; /* Increased size */
        color: #e74c3c; 
        text-align: center;
        border: 1px solid #c0392b; 
        font-family: 'Montserrat', sans-serif !important;
        padding: 15px;
        border-radius: 10px;
        background-color: #fdecea;
        margin-top: 20px;
    }

    /* Recommendation text (which uses .sub-font) will inherit its centered alignment. 
       If left-alignment is preferred for recommendations within result boxes, specific CSS would be needed.
       For now, it will use the .sub-font styling. */

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
    
    # URLs de las imágenes (RECUERDA REEMPLAZAR pcos_image_url con una URL real DE ACUERDO A LAS SUGERENCIAS)
    no_pcos_image_url = "https://images.pexels.com/photos/3076509/pexels-photo-3076509.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"  # wellness/health
    pcos_image_url = "https://images.pexels.com/photos/4021779/pexels-photo-4021779.jpeg"      # Placeholder: medical support/consultation

    # Mostrar el resultado y recomendaciones
    if prediction[0] == 0:
        st.markdown('<p class="result-green">Diagnóstico: No PCOS</p>', unsafe_allow_html=True)
        # Solo mostrar la imagen si la URL no es el placeholder o la URL actualizada
        if no_pcos_image_url and no_pcos_image_url not in ["URL_FOR_NO_PCOS_IMAGE", "https://images.pexels.com/photos/3985062/pexels-photo-3985062.jpeg"]:
            st.image(no_pcos_image_url, use_container_width=True, caption="Mantén un estilo de vida saludable")
        
        recommendation_no_pcos = (
            "Aunque el diagnóstico no indica PCOS, es importante mantener un estilo de vida saludable.<br><br>"
            "Continúe con una dieta equilibrada y ejercicio regular.<br><br>"
            "Si tiene alguna preocupación sobre su salud, consulte a un profesional médico."
        )
        st.markdown(f'<div style="text-align: center; margin-top: 15px;"><p class="sub-font">{recommendation_no_pcos}</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="result-red">Diagnóstico: PCOS</p>', unsafe_allow_html=True)
        # Solo mostrar la imagen si la URL no es el placeholder
        if pcos_image_url and pcos_image_url != "URL_FOR_PCOS_IMAGE":
            st.image(pcos_image_url, use_container_width=True, caption="Consulte a un especialista para un seguimiento adecuado")
            
        recommendation_pcos = (
            "El diagnóstico indica la posibilidad de PCOS.<br><br>"
            "Se recomienda consultar a un endocrinólogo o ginecólogo para una evaluación más detallada.<br><br>"
            "El tratamiento puede incluir cambios en el estilo de vida, medicamentos o terapia hormonal.<br><br>"
            "Es importante seguir las recomendaciones médicas para manejar los síntomas de manera efectiva."
        )
        st.markdown(f'<div style="text-align: center; margin-top: 15px;"><p class="pcos-recommendation-text">{recommendation_pcos}</p></div>', unsafe_allow_html=True)
