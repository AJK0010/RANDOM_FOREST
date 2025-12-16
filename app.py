import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Placement Prediction App",
    page_icon="🎓",
    layout="centered"
)

# Load model safely
@st.cache_resource
def load_model():
    with open("placement_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# App title
st.title("🎓 Placement Prediction System")
st.subheader("AI-powered Placement Chance Predictor 🚀")

st.markdown("---")

# User inputs
cgpa = st.number_input("📘 CGPA", min_value=0.0, max_value=10.0, value=7.5)
internships = st.number_input("💼 Internships Completed", min_value=0, value=1)
projects = st.number_input("🛠️ Projects Done", min_value=0, value=2)
certifications = st.number_input("📜 Certifications", min_value=0, value=1)
aptitude_score = st.number_input("🧠 Aptitude Score", min_value=0, max_value=100, value=70)

st.markdown("---")

# Predict button
if st.button("🔍 Predict Placement"):
    try:
        input_data = np.array([[cgpa, internships, projects, certifications, aptitude_score]])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("🎉 Congratulations! You are likely to get PLACED!")
        else:
            st.warning("📉 Placement chances are low. Keep improving your skills!")

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

st.markdown("---")

# Footer
st.markdown(
    """
    🔹 **Model:** Pickle (.pkl)  
    🔹 **Framework:** Scikit-Learn  
    🔹 **Deployment:** Streamlit Cloud  
    """
)
