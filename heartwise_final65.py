
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import random
import string
import sqlite3
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# DATABASE SETUP (SQLite)
# ----------------------------
conn = sqlite3.connect("heartwise.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS patients (
    patient_code TEXT PRIMARY KEY,
    age_diagnosed INTEGER,
    stage INTEGER
)
""")
conn.commit()

def save_patient(patient_code, age_diagnosed, stage):
    c.execute("REPLACE INTO patients (patient_code, age_diagnosed, stage) VALUES (?, ?, ?)",
              (patient_code, age_diagnosed, stage))
    conn.commit()

def load_patient(patient_code):
    c.execute("SELECT age_diagnosed, stage FROM patients WHERE patient_code = ?", (patient_code,))
    return c.fetchone()

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(page_title="HeartWise", layout="centered")
st.title("🫀 HEARTWISE")
st.markdown("### Stage Prediction Tool for Coronary Artery Disease")

# ----------------------------
# LOAD TRAINED MODEL
# ----------------------------
rf_model = joblib.load("rf_model_final_web.pkl")

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def generate_code():
    return "HW" + ''.join(random.choices(string.digits, k=4))

def calculate_map_pp(sys, dia):
    map_val = (2 * dia + sys) / 3
    pp_val = sys - dia
    return round(map_val, 1), round(pp_val, 1)

def lifestyle_recommendations(systolic=None, diastolic=None, ldl=None, hdl=None):
    """Lifestyle Interventions."""
    st.subheader("💡 Lifestyle Interventions")

    bp_status = ""
    chol_status = ""

    # --- Blood Pressure Conditions ---
    if systolic is not None and diastolic is not None:
        if systolic < 90 and diastolic < 60:
            bp_status = "⚠️ Blood Pressure - Consult a doctor"
        elif 90 <= systolic <= 120 and 60 <= diastolic <= 80:
            bp_status = "✅ Blood Pressure - Safe with Lifestyle Interventions"
        elif systolic >= 130 or diastolic >= 80:
            bp_status = "⚠️ Blood Pressure - Consult a doctor"
        else:
            bp_status = "⚠️ Blood Pressure - Needs monitoring"

    # --- Cholesterol Conditions ---
    if ldl is not None and hdl is not None:
        if ldl < 100 and hdl >= 60:
            chol_status = "✅ Cholesterol - Safe with Lifestyle Interventions"
        elif ldl > 100 and hdl < 60:
            chol_status = "⚠️ Cholesterol - Consult a doctor"
        else:
            chol_status = "⚠️ Cholesterol - Needs monitoring"

    # --- Display Results ---
    if bp_status:
        st.markdown(bp_status)
    if chol_status:
        st.markdown(chol_status)

    # --- Recommendations Based on Status ---
    if systolic is not None and diastolic is not None and systolic < 90 and diastolic < 60:
        st.markdown("""
        **You may include this:**
        - Drink plenty of water
        """)
    else:
        st.markdown("""
        **You may include these:**
        - Apply the Dietary Approaches to Stop Hypertension (DASH) Diet.  
        - Try to get at least 150 minutes of moderate to intensity aerobic activity each week.  
        - Manage stress by having at least 7 hours of sleep.  
        - Quit smoking and avoid alcohol.
        """)

def get_stage_description(stage):
    descriptions = {
        1: "💛 **Stage 1:** Mild progression",
        2: "🧡 **Stage 2:** Moderate progression",
        3: "💗 **Stage 3:** Severe progression",
        4: "❤️ **Stage 4:** Advanced progression"
    }
    return descriptions.get(stage, "Unknown Stage")

def get_symptoms(stage):
    symptoms = {
        1: "💢 **Symptoms:** Chest pain only after unusual or prolonged physical activity, but not during basic everyday activities like walking or climbing stairs.",
        2: "💢 **Symptoms:** Chest pain during more intense activities like walking quickly, walking uphill and climbing stairs after eating, when it's cold or when also under emotional stress.",
        3: "💢 **Symptoms:** Chest pain even during low-intensity physical activities like walking or getting dressed.",
        4: "💢 **Symptoms:** Chest pain when at rest or during the gentlest physical activities. This is then referred to as unstable angina."
    }
    return symptoms.get(stage, "N/A")

def biomarker_info(map_val, pp_val):
    st.subheader("🧪 Biomarker Analysis")
    st.markdown(f"**Mean Arterial Pressure (MAP):** {map_val} mmHg")
    st.caption("🔹 **Normal range:** 70–100 mmHg")
    st.markdown(f"**Pulse Pressure (PP):** {pp_val} mmHg")
    st.caption("🔹 **Normal range:** 40–60 mmHg")

# ----------------------------
# PATIENT DATA STORAGE
# ----------------------------
if "patient_data" not in st.session_state:
    st.session_state["patient_data"] = {}

# ----------------------------
# MAIN INTERFACE - BUTTONS FIRST
# ----------------------------
st.markdown("Select your user type to continue:")

col1, col2 = st.columns(2)
new_patient_clicked = col1.button("🧍 New Patient")
returning_patient_clicked = col2.button("🔁 Returning Patient")

if new_patient_clicked:
    st.session_state["page_choice"] = "new"
elif returning_patient_clicked:
    st.session_state["page_choice"] = "returning"

page_choice = st.session_state.get("page_choice", None)

# ============================
# NEW PATIENT SECTION
# ============================
if page_choice == "new":
    st.header("🧍 New Patient Registration")

    age_diagnosed = st.number_input("Age Diagnosed with CAD", value=None)
    current_age = st.number_input("Current Age", value=None)
    systolic = st.number_input("Systolic BP (mmHg)", value=None)
    diastolic = st.number_input("Diastolic BP (mmHg)", value=None)
    ldl = st.number_input("LDL Cholesterol (mg/dL)", value=None)
    hdl = st.number_input("HDL Cholesterol (mg/dL)", value=None)

    if st.button("🔍 Predict CAD Stage"):
        start = time.time()
        years_since = current_age - age_diagnosed

        X_input = np.array([[age_diagnosed, current_age, systolic, diastolic, ldl, hdl, years_since]])
        preds = np.array([tree.predict(X_input)[0] for tree in rf_model.estimators_])
        mean_pred = np.mean(preds)
        std_pred = np.std(preds)
        confidence = max(0, 100 - (std_pred / (mean_pred + 1e-6)) * 100)
        stage = int(round(mean_pred))

        # --- Display prediction ---
        st.success(f"🩺 Predicted CAD Stage: **Stage {stage}**")
        st.markdown(get_stage_description(stage))
        st.markdown(get_symptoms(stage))
        st.caption(f"Prediction Confidence: **{confidence:.1f}%**")

        # --- Compute biomarkers after prediction ---
        map_val, pp_val = calculate_map_pp(systolic, diastolic)
        biomarker_info(map_val, pp_val)

        # --- Lifestyle Recommendations ---
        lifestyle_recommendations(systolic, diastolic, ldl, hdl)

        # Generate and save patient code
        patient_code = generate_code()
        save_patient(patient_code, age_diagnosed, stage)

        st.markdown(f"### 🔑 Your Patient Code: **{patient_code}**")
        st.caption("Save this code to monitor your future CAD status.")

        st.caption(f"⏱️ Web Response time: {time.time() - start:.3f} seconds")

# ============================
# RETURNING PATIENT SECTION
# ============================
elif page_choice == "returning":
    st.header("🔁 Returning Patient Login")

    patient_code = st.text_input("Enter your patient code (e.g., HW1234):")

    if patient_code:
        data = load_patient(patient_code)

        if not data:
            st.error("❌ Invalid code. Please check your entry or register as a new patient.")
        else:
            stored_age_diagnosed, prev_stage = data
            st.success(f"Welcome back! Your previous CAD stage was **Stage {prev_stage}**.")

            # AUTO-FILL AGE DIAGNOSED
            age_diagnosed = st.number_input("Age Diagnosed with CAD", value=stored_age_diagnosed)
            current_age = st.number_input("Current Age", value=None)
            systolic = st.number_input("Systolic BP (mmHg)", value=None)
            diastolic = st.number_input("Diastolic BP (mmHg)", value=None)
            ldl = st.number_input("LDL Cholesterol (mg/dL)", value=None)
            hdl = st.number_input("HDL Cholesterol (mg/dL)", value=None)

            if st.button("🔍 Update Prediction"):
                start = time.time()
                years_since = current_age - age_diagnosed

                X_input = np.array([[age_diagnosed, current_age, systolic, diastolic, ldl, hdl, years_since]])
                preds = np.array([tree.predict(X_input)[0] for tree in rf_model.estimators_])
                mean_pred = np.mean(preds)
                stage = int(round(mean_pred))

                map_val, pp_val = calculate_map_pp(systolic, diastolic)

                # ---- STAGE BEHAVIOR ----
                if stage > prev_stage:
                    st.error(f"⚠️ Condition has worsened. New stage: **Stage {stage}**")
                    st.markdown(get_stage_description(stage))
                    st.markdown(get_symptoms(stage))
                else:
                    st.success("✅ Condition has stabilized since last assessment.")

                biomarker_info(map_val, pp_val)
                lifestyle_recommendations(systolic, diastolic, ldl, hdl)

                # UPDATE DATABASE
                save_patient(patient_code, age_diagnosed, stage)

                st.caption(f"⏱️ Web Response time: {time.time() - start:.3f} seconds")
