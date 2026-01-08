import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import kagglehub

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Disease Prediction App", layout="centered")

# ===============================
# LOAD MODEL ARTIFACTS
# ===============================
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ===============================
# LOAD DATASETS (CACHED)
# ===============================
@st.cache_data
def load_aux_data():
    path = kagglehub.dataset_download(
        "itachi9604/disease-symptom-description-dataset"
    )
    desc = pd.read_csv(os.path.join(path, "symptom_Description.csv"))
    prec = pd.read_csv(os.path.join(path, "symptom_precaution.csv"))
    severity = pd.read_csv(os.path.join(path, "Symptom-severity.csv"))
    return desc, prec, severity

desc_df, prec_df, severity_df = load_aux_data()

# ===============================
# SAFE DATA ACCESS HELPERS (FIX)
# ===============================
def get_description(disease):
    row = desc_df[desc_df["Disease"] == disease]
    if not row.empty:
        return row["Description"].values[0]
    return "No detailed description available for this condition."

def get_precautions(disease):
    row = prec_df[prec_df["Disease"] == disease]
    if not row.empty:
        return row.iloc[0, 1:].values
    return ["Consult a medical professional."]

# ===============================
# UI HEADER
# ===============================
st.title("🩺 Intelligent Disease Prediction System")
st.caption("ML-based early disease prediction using symptom analysis")

# ===============================
# SYMPTOM SELECTION
# ===============================
symptoms = sorted(list(set(col.split("_")[-1] for col in columns)))

selected_symptoms = st.multiselect(
    "Select your symptoms",
    symptoms,
    help="Choose all symptoms you are experiencing"
)

# ===============================
# PREDICTION LOGIC
# ===============================
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Create input vector
        input_data = pd.DataFrame(0, index=[0], columns=columns)

        for symptom in selected_symptoms:
            for col in columns:
                if symptom in col:
                    input_data[col] = 1

        # Predict probabilities
        probabilities = model.predict_proba(input_data)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]

        st.subheader("🧬 Prediction Results")

        for i, idx in enumerate(top_indices):
            disease = label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx] * 100

            description = get_description(disease)
            precautions = get_precautions(disease)

            st.markdown(f"### {i+1}. {disease}")
            st.progress(confidence / 100)
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.info(description)

            with st.expander("🛡️ Precautions"):
                for p in precautions:
                    st.write("✔️", p)

        # ===============================
        # SEVERITY SCORE
        # ===============================
        severity_map = dict(
            zip(severity_df["Symptom"], severity_df["weight"])
        )

        score = sum(severity_map.get(sym, 1) for sym in selected_symptoms)

        st.subheader("📊 Symptom Severity Analysis")
        st.write(f"**Severity Score:** {score}")

        if score < 10:
            st.success("🟢 Low severity – monitor symptoms.")
        elif score < 20:
            st.warning("🟠 Moderate severity – consult a doctor.")
        else:
            st.error("🔴 High severity – seek medical attention immediately.")

# ===============================
# DISCLAIMER
# ===============================
st.divider()
st.caption(
    "⚠️ This system is for educational and early-warning purposes only. "
    "It does not replace professional medical diagnosis."
)
