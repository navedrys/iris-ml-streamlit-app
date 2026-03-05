import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load the trained model
# -----------------------------
model = pickle.load(open("iris.pkl", "rb"))

# -----------------------------
# Page Title
# -----------------------------
st.set_page_config(page_title="Iris ML Predictor", page_icon="🌸")

st.title("🌸 Iris Flower Prediction App")
st.write("Predict the species of an Iris flower using Machine Learning.")

st.markdown("---")

# -----------------------------
# User Input Section
# -----------------------------
st.header("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2
    )

st.markdown("---")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Species"):

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(features)

    species = ["Setosa", "Versicolor", "Virginica"]

    result = species[prediction[0]]

    st.success(f"Predicted Iris Species: **{result}**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Built with Python and Streamlit")
