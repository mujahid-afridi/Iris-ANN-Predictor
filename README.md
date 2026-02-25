# 🌸 Iris ANN Predictor

A Streamlit web app that predicts **Iris flower species** using an Artificial Neural Network (ANN) model. Users can input sepal and petal measurements and get predictions along with confidence scores.

---

## 🧠 Model Details

- **Dataset:** Iris Dataset (150 samples, 3 species: Setosa, Versicolor, Virginica)  
- **Model Type:** Artificial Neural Network (ANN)  
- **Accuracy:** ~89%  
- **Input Features:**  
  - Sepal Length (cm)  
  - Sepal Width (cm)  
  - Petal Length (cm)  
  - Petal Width (cm)  

---

## 💻 Features

- Interactive **Streamlit UI**  
- Input flower measurements using sliders  
- Shows **predicted species** and **prediction confidence**  
- Sidebar with **model info and instructions**  
- Handles **scaled inputs** with saved scaler (`scaler.pkl`)  

---

## 📦 Project Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit app code for user interface |
| `iris_model.h5` | Trained ANN model |
| `scaler.pkl` | Scaler used for feature normalization |
| `README.md` | Project documentation |
| `phase2.ipynb` | Model code |
| `Iris.csv` | Iris Dataset |