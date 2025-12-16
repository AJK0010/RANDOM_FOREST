import streamlit as st
import pickle
import numpy as np
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Random Forest Classifier",
    page_icon="🌲",
    layout="centered"
)

st.title("🌲 Random Forest Prediction App")
st.write("Enter feature values to get a prediction.")

# =========================
# LOAD MODEL SAFELY
# =========================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "random_forest_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.info("Make sure **random_forest_model.pkl** is in the same folder as app.py")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.success("✅ Model loaded successfully!")

# =========================
# INPUTS (CHANGE COUNT IF NEEDED)
# =========================
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    X = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(X)

    st.subheader("🔮 Prediction Result")
    st.success(f"Predicted Class: {prediction[0]}")
