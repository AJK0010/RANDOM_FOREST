import streamlit as st
import pickle
import numpy as np
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

# =========================
# LOAD MODEL (FIXED PATH)
# =========================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "placement_model.pkl")

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()

# =========================
# UI
# =========================
st.title("🎓 Placement Prediction App")
st.write("Enter student details to predict placement status")

cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("IQ", min_value=0, max_value=300, step=1)
profile_score = st.number_input("Profile Score", min_value=0, max_value=100, step=1)

if st.button("Predict"):
    input_data = np.array([[cgpa, iq, profile_score]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student will be placed")
    else:
        st.error("❌ Student will not be placed")
