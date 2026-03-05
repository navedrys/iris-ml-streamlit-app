import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load the trained model
# -----------------------------
model = pickle.load(open("iris.pkl", "rb"))

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")

st.title("🌸 Iris Flower Species Prediction")
st.write("Enter the flower measurements to predict the species.")

st.markdown("---")

# -----------------------------
# Input Fields
# -----------------------------
st.header("Input Flower Measurements")

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
# Prediction
# -----------------------------
if st.button("Predict Species"):

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(features)

    # Mapping for safety (if model returns numbers)
    species_map = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    pred = prediction[0]

    # If prediction is number
    if isinstance(pred, (int, np.integer)):
        result = species_map[pred]
    else:
        # If prediction is already text
        result = str(pred).replace("Iris-", "").capitalize()

    st.success(f"Predicted Iris Species: {result}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Built using Python and Streamlit")
