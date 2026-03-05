import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("iris.pkl", "rb"))

st.set_page_config(page_title="Iris Predictor", page_icon="🌸")

st.title("🌸 Iris Flower Species Predictor")

st.write("Enter the flower measurements below:")

st.markdown("---")

# Input fields
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
    petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

st.markdown("---")

# Prediction button
if st.button("Predict Species"):

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(features)

    # Convert prediction to integer
    pred = int(prediction[0])

    # Map numbers to species
    species = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    result = species[pred]

    st.success(f"Predicted Iris Species: {result}")
