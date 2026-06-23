import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)


@st.cache_resource
def load_models():
    model = load_model("models/heart_disease_model.keras")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


model, scaler = load_models()

st.markdown("""
    <h1 style='text-align: center; color: #e63946; font-size: 42px;'>
        ❤️ Heart Disease Risk Predictor
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center; font-size:18px; color: gray;'>
        AI-powered medical risk analysis system
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 50, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])


st.markdown("<br>", unsafe_allow_html=True)

predict = st.button("🔍 Predict Heart Disease Risk", use_container_width=True)


if predict:

    input_data = np.array([[
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ]])

    input_scaled = scaler.transform(input_data)

    prob = model.predict(input_scaled)[0][0]

    st.markdown("---")
    st.subheader("🧠 Prediction Result")

    
    if prob < 0.3:
        st.success("🟢 LOW RISK")
    elif prob < 0.6:
        st.warning("🟡 MEDIUM RISK")
    else:
        st.error("🔴 HIGH RISK")

    
    st.markdown(f"""
        <div style="
            padding:20px;
            border-radius:10px;
            background-color:#f1f1f1;
            text-align:center;
            font-size:20px;
        ">
        <b>Probability:</b> {prob:.2%}
        </div>
    """, unsafe_allow_html=True)

    st.progress(float(prob))


st.markdown("---")
st.write("""
📌 Model Insight:
- Neural Network trained on clinical features
- Key indicators: chest pain, heart rate, oldpeak, vessels
- Output is probability-based risk prediction
""")