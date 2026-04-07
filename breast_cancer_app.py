import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Breast Cancer Diagnosis",
    page_icon="🔬",
    layout="centered"
)

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('Breast Cancer Wisconsin (Diagnostic).csv')

    # Drop unnecessary columns
    df = df.drop(['id'], axis=1)
    if 'Unnamed: 32' in df.columns:
        df = df.drop(['Unnamed: 32'], axis=1)

    # Encode target
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])  # B=0, M=1

    # Features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    feature_names = X.columns.tolist()
    return model, scaler, feature_names

# Load model
with st.spinner("Loading model... please wait"):
    model, scaler, feature_names = load_and_train_model()

# UI
st.title("🔬 Breast Cancer Diagnosis Predictor")
st.markdown("Enter tumour measurements to predict whether the tumour is **Benign** or **Malignant**.")
st.divider()

st.subheader("Mean Measurements")
col1, col2, col3 = st.columns(3)

with col1:
    radius_mean = st.number_input("Radius Mean", min_value=0.0, value=14.0, step=0.1)
    texture_mean = st.number_input("Texture Mean", min_value=0.0, value=19.0, step=0.1)
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=92.0, step=0.1)
    area_mean = st.number_input("Area Mean", min_value=0.0, value=655.0, step=1.0)

with col2:
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.096, step=0.001, format="%.3f")
    compactness_mean = st.number_input("Compactness Mean", min_value=0.0, value=0.104, step=0.001, format="%.3f")
    concavity_mean = st.number_input("Concavity Mean", min_value=0.0, value=0.089, step=0.001, format="%.3f")
    concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, value=0.049, step=0.001, format="%.3f")

with col3:
    symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.181, step=0.001, format="%.3f")
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, value=0.063, step=0.001, format="%.3f")

st.divider()
st.subheader("Standard Error Measurements")
col4, col5, col6 = st.columns(3)

with col4:
    radius_se = st.number_input("Radius SE", min_value=0.0, value=0.405, step=0.001, format="%.3f")
    texture_se = st.number_input("Texture SE", min_value=0.0, value=1.216, step=0.001, format="%.3f")
    perimeter_se = st.number_input("Perimeter SE", min_value=0.0, value=2.866, step=0.001, format="%.3f")
    area_se = st.number_input("Area SE", min_value=0.0, value=40.0, step=0.1)

with col5:
    smoothness_se = st.number_input("Smoothness SE", min_value=0.0, value=0.007, step=0.001, format="%.3f")
    compactness_se = st.number_input("Compactness SE", min_value=0.0, value=0.025, step=0.001, format="%.3f")
    concavity_se = st.number_input("Concavity SE", min_value=0.0, value=0.032, step=0.001, format="%.3f")
    concave_points_se = st.number_input("Concave Points SE", min_value=0.0, value=0.012, step=0.001, format="%.3f")

with col6:
    symmetry_se = st.number_input("Symmetry SE", min_value=0.0, value=0.021, step=0.001, format="%.3f")
    fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, value=0.004, step=0.001, format="%.3f")

st.divider()
st.subheader("Worst Measurements")
col7, col8, col9 = st.columns(3)

with col7:
    radius_worst = st.number_input("Radius Worst", min_value=0.0, value=16.0, step=0.1)
    texture_worst = st.number_input("Texture Worst", min_value=0.0, value=25.0, step=0.1)
    perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, value=107.0, step=0.1)
    area_worst = st.number_input("Area Worst", min_value=0.0, value=880.0, step=1.0)

with col8:
    smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, value=0.132, step=0.001, format="%.3f")
    compactness_worst = st.number_input("Compactness Worst", min_value=0.0, value=0.254, step=0.001, format="%.3f")
    concavity_worst = st.number_input("Concavity Worst", min_value=0.0, value=0.272, step=0.001, format="%.3f")
    concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, value=0.115, step=0.001, format="%.3f")

with col9:
    symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, value=0.290, step=0.001, format="%.3f")
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, value=0.084, step=0.001, format="%.3f")

st.divider()

if st.button("🔬 Diagnose Tumour", use_container_width=True, type="primary"):

    input_data = np.array([[
        radius_mean, texture_mean, perimeter_mean, area_mean,
        smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
        symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se,
        smoothness_se, compactness_se, concavity_se, concave_points_se,
        symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst,
        smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
        symmetry_worst, fractal_dimension_worst
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()

    if prediction == 1:
        malignant_prob = round(probability[1] * 100, 1)
        st.error("⚠️ Tumour is likely MALIGNANT")
        st.metric("Malignant Probability", f"{malignant_prob:.1f}%")
        st.markdown("**Recommendation:** Immediate medical consultation advised. This prediction is for reference only and should not replace professional medical diagnosis.")
    else:
        benign_prob = round(probability[0] * 100, 1)
        st.success("✅ Tumour is likely BENIGN")
        st.metric("Benign Probability", f"{benign_prob:.1f}%")
        st.markdown("**Status:** Low malignancy risk detected. Regular monitoring recommended. This prediction is for reference only.")

st.divider()
st.caption("Built by Shiny Madona Arockiasamy | IBM Certified Data Scientist | github.com/shinymadona")
st.caption("⚠️ This app is for educational purposes only and should not be used as a substitute for professional medical advice.")
