import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ------------------------
# Load model & scaler
# ------------------------
model = load_model("iris_model.h5")
scaler = joblib.load("scaler.pkl")

# ------------------------
# Page configuration
# ------------------------
st.set_page_config(
    page_title="Iris Species Predictor 🌸",
    page_icon="🌼",
    layout="wide"
)

# ------------------------
# Sidebar - Model Info
# ------------------------
st.sidebar.title("ℹ️ Model Information")
st.sidebar.markdown("""
**Model:** ANN (Artificial Neural Network)  
**Dataset:** Iris Dataset  
**Accuracy:** 89%  
**Purpose:** Predict the species of Iris flower based on sepal and petal measurements.  
**Species:** Setosa, Versicolor, Virginica  
**Instructions:** Move the sliders to set the flower measurements and click **Predict**.
""")

# ------------------------
# Main Title
# ------------------------
st.title("🌸 Iris Species Prediction App")
st.markdown("Predict the Iris flower species by entering its measurements below:")

# ------------------------
# Input sliders
# ------------------------
st.subheader("Enter Flower Measurements:")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)

with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# ------------------------
# Predict Button
# ------------------------
if st.button("Predict 🌼"):
    # Prepare input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction)
    predicted_species = ["Setosa", "Versicolor", "Virginica"][predicted_class]
    confidence = prediction[0][predicted_class] * 100

    # ------------------------
    # Display prediction in a card-like format
    # ------------------------
    st.markdown("---")
    st.markdown("### 🌸 Prediction Result")
    st.success(f"**Predicted Species:** {predicted_species}")
    st.info(f"**Prediction Confidence:** {confidence:.2f}%")
    st.write(f"**Sepal Length:** {sepal_length} cm")
    st.write(f"**Sepal Width:** {sepal_width} cm")
    st.write(f"**Petal Length:** {petal_length} cm")
    st.write(f"**Petal Width:** {petal_width} cm")