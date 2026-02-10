import streamlit as st
import joblib
import pandas as pd
import numpy as np

# CONFIG FIRST
st.set_page_config(
    page_title="ICU Mortality Risk Predictor",
    layout="wide"
)

# LOAD MODEL SAFELY 
@st.cache_resource
def load_model():
    return joblib.load("logreg_pipeline.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found! Please upload 'final_model_logreg_balanced.pkl' to the same folder.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Info"])

# PAGE: HOME
if page == "Home":
    st.title("ICU Mortality Risk Predictor")
    st.markdown("""
    This clinical decision-support tool predicts the probability of **in-hospital mortality** for ICU patients using data from the first 24 hours.
    
    **Model Performance:**
    - **Sensitivity (Recall):** 73% (High safety)
    - **Algorithm:** Logistic Regression (Balanced/Upscaled)
    """)
    st.info("👈 Select **'Prediction'** in the sidebar to start.")

#PAGE: PREDICTION
elif page == "Prediction":
    st.title("Patient Assessment")
    
    # Input Form
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vitals & CNS")
            gcs_total = st.number_input("GCS Total (3-15)", 3, 15, 15)
            d1_sysbp_min = st.number_input("Min Systolic BP (mmHg)", 0, 300, 120)
            d1_heartrate_max = st.number_input("Max Heart Rate (bpm)", 0, 300, 80)
            d1_temp_min = st.number_input("Min Temperature (°C)", 30.0, 45.0, 37.0)
            ventilated_apache = st.selectbox("Ventilated?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        with col2:
            st.subheader("Labs")
            d1_lactate_max = st.number_input("Max Lactate (mmol/L)", 0.0, 30.0, 1.0)
            d1_bun_max = st.number_input("Max BUN (mg/dL)", 0.0, 200.0, 20.0)
            d1_inr_max = st.number_input("Max INR", 0.0, 10.0, 1.0)
            d1_spo2_min = st.number_input("Min SpO2 (%)", 0, 100, 98)
            d1_arterial_ph_min = st.number_input("Min Arterial pH", 6.5, 8.0, 7.35)

        submit = st.form_submit_button("Calculated Risk")

    if submit:
        # Create DataFrame (Must match training order!)
        input_data = pd.DataFrame([[
            d1_lactate_max, gcs_total, ventilated_apache, d1_sysbp_min, 
            d1_spo2_min, d1_temp_min, d1_arterial_ph_min, d1_bun_max, 
            d1_heartrate_max, d1_inr_max
        ]], columns=[
            'd1_lactate_max', 'gcs_total', 'ventilated_apache', 'd1_sysbp_min', 
            'd1_spo2_min', 'd1_temp_min', 'd1_arterial_ph_min', 'd1_bun_max', 
            'd1_heartrate_max', 'd1_inr_max'
        ])
        
        # Predict
        probability = model.predict_proba(input_data)[0][1]
        risk_percent = probability * 100
        
        st.divider()
        st.subheader("Result")
        
        # Color-coded Risk Levels
        if risk_percent < 20:
            st.success(f"**Low Risk ({risk_percent:.1f}%)**\n\nPatient is likely stable.")
        elif risk_percent < 50:
            st.warning(f"**Moderate Risk ({risk_percent:.1f}%)**\n\n⚠️ Consider closer monitoring.")
        else:
            st.error(f"**High Risk ({risk_percent:.1f}%)**\n\n🚨 Urgent intervention may be required.")

# PAGE: MODEL INFO
elif page == "Model Info":
    st.title("Model Internals")
    st.write("This model was trained using **Class Balancing** to prioritize patient safety.")
    
    # Feature Importance Visualization
    st.subheader("What drives the risk?")
    st.write("Positive values increase risk (Red), Negative values decrease risk (Blue).")
    
    
    try:
        # 1. Get Coefficients
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['model']
        else:
            classifier = model
            
        coefficients = classifier.coef_[0]
        
        feature_names = [
            'd1_lactate_max', 'gcs_total', 'ventilated_apache', 'd1_sysbp_min', 
            'd1_spo2_min', 'd1_temp_min', 'd1_arterial_ph_min', 'd1_bun_max', 
            'd1_heartrate_max', 'd1_inr_max'
        ]

        # 2. Create Dataframe with Color Logic
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients
        })
        
        # Define Color: Red for Risk (Positive), Blue for Protective (Negative)
        importance_df["Color"] = importance_df["Coefficient"].apply(lambda x: "red" if x > 0 else "blue")
        importance_df["Absolute Impact"] = np.abs(importance_df["Coefficient"])
        importance_df = importance_df.sort_values(by="Absolute Impact", ascending=False)

        # 3. Create the Colorful Chart using Altair
        import altair as alt

        chart = alt.Chart(importance_df).mark_bar().encode(
            x=alt.X('Coefficient', title='Impact on Mortality Risk'),
            y=alt.Y('Feature', sort='-x', title='Clinical Feature'),
            color=alt.Color('Color', scale=None), # Use our manual colors
            tooltip=['Feature', 'Coefficient']
        ).properties(
            title="Feature Importance (Red = Increases Risk, Blue = Decreases Risk)"
        )

        st.altair_chart(chart, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")