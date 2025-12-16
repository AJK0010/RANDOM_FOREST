import streamlit as st
import pickle
import numpy as np

# Load model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Random Forest Prediction", layout="centered")

st.title("🌲 Random Forest Classifier")

st.write("Enter feature values below:")

# Example inputs (change according to your dataset)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

if st.button("Predict"):
    X = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(X)

    st.success(f"Prediction: {prediction[0]}")
