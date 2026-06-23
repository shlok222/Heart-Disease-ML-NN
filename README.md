# ❤️ Heart Disease Prediction System (Neural Network)

## 📌 Overview
This project is a Machine Learning web application that predicts the risk of heart disease using a Neural Network model.  
It provides a real-time risk assessment based on patient medical parameters.

The model is deployed using **Streamlit** for an interactive and user-friendly interface.

---

## 🚀 Live Demo
(After deployment, add your Streamlit Cloud link here)

Example:
https://your-app-name.streamlit.app

---

## 🧠 Problem Statement
Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve patient outcomes.  
This project aims to assist in early prediction using machine learning.

---

## 📊 Dataset
- UCI Heart Disease Dataset
- Features include:
  - Age, Sex, Chest Pain Type
  - Blood Pressure, Cholesterol
  - Heart Rate, ECG results, etc.

---

## 🤖 Model Architecture
- Neural Network (Keras / TensorFlow)
- Input Layer: 13 medical features
- Hidden Layers: Dense layers with activation functions
- Output Layer: Sigmoid (binary classification)

---

## ⚙️ Features
- Real-time prediction
- Risk classification:
  - 🟢 Low Risk
  - 🟡 Medium Risk
  - 🔴 High Risk
- Probability score output
- Clean Streamlit UI

---

## 📈 Evaluation Metrics
- Accuracy: ~84%
- Precision / Recall / F1-score
- Confusion Matrix analysis
- Threshold tuning for reducing False Negatives (important in healthcare)

---

## 🧠 Key Insight
In healthcare applications, **False Negatives are more critical than False Positives**, so the model threshold was tuned to improve recall.

---

## 🏗️ Project Structure