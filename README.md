# 🩺 Disease Prediction System

An end-to-end Machine Learning application that predicts possible diseases based on user-selected symptoms and provides medical descriptions, precautions, and severity analysis using a Streamlit web interface.

---

## 📌 Project Overview

The Disease Prediction System uses a supervised machine learning model to analyze symptoms entered by users and predict the most likely diseases.  
It is designed for **early-stage health awareness** and **educational purposes**, not as a replacement for professional medical diagnosis.

---

## 🚀 Features

- 🔍 Symptom-based disease prediction  
- 📊 Top-3 disease predictions with confidence scores  
- 🧠 Machine Learning model using K-Nearest Neighbors (KNN)  
- 📝 Disease descriptions and recommended precautions  
- ⚖️ Symptom severity analysis  
- 🛡️ Defensive handling for missing dataset information  
- 🌐 Interactive Streamlit web application  

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend / ML:** Python, Scikit-learn  
- **Data Handling:** Pandas, NumPy  
- **Model:** Weighted KNN Classifier  
- **Dataset Source:** Kaggle (Disease–Symptom Dataset)  

---

## 📂 Project Structure
Disease-Prediction/
- │── app.py
- │── model.pkl
- │── columns.pkl
- │── label_encoder.pkl
- │── requirements.t
