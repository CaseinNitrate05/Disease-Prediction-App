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
Disease-Prediction/<br>

│── app.py <br>

│── model.pkl <br>

│── columns.pkl <br>

│── label_encoder.pkl <br>

│── requirements.t

---

## ⚙️ How the System Works

1. User selects one or more symptoms from the UI  
2. Symptoms are converted into a numerical feature vector  
3. The trained ML model predicts disease probabilities  
4. The top 3 most likely diseases are displayed  
5. Disease description, precautions, and severity analysis are shown  

---

## 🧠 Machine Learning Details

- **Algorithm:** K-Nearest Neighbors (KNN)  
- **Reason for Choosing KNN:**
  - Works well with symptom similarity
  - Easy to interpret
  - Suitable for multi-class classification  
- **Encoding:** One-hot encoding for symptom features  
- **Evaluation:** Train–test split with accuracy-based evaluation  

---

## ⚠️ Disclaimer

This application is intended for **educational and early-warning purposes only**.  
It does **not** replace professional medical diagnosis or treatment.

---

## ▶️ How to Run Locally

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 🌍 Deployment

- The application is compatible with Streamlit Cloud and can be deployed directly using GitHub by including:
  - app.py
  - model files (.pkl)
  - requirements.txt

---
