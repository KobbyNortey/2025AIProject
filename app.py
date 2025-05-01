# app.py ────────────────────────────────────────────────────────────────
import pathlib, joblib, pandas as pd, streamlit as st
from tensorflow.keras.models import load_model


@st.cache_resource
def _load():
    art = pathlib.Path("artefacts")
    pre = joblib.load(art / "preprocessor.joblib")
    net = load_model(art / "energy_net.keras", compile=False)
    return pre, net


preprocessor, net = _load()

st.sidebar.header("Input parameters (Per Semester)")
bt = st.sidebar.selectbox("Building Type", ["Residential", "Commercial", "Industrial"])
sf = st.sidebar.number_input("Square Footage", 100.0, 1_000_000.0, 5_000.0, 100.0)
occ = st.sidebar.number_input("Average Number of Occupants", 0, 10_000, 50, 1)
app = st.sidebar.number_input("Average Daily Appliances Used", 0, 5_000, 100, 1)
tmp = st.sidebar.number_input("Average Temp (°C)", 20.0, 50.0, 25.0, 0.1)
dow = st.sidebar.selectbox("Day of Week",
                           ["Weekday", "Weekend"], )

st.title("Energy-Consumption Predictor")

if st.sidebar.button("Predict"):
    row = pd.DataFrame([{
        "Building Type": bt,
        "Square Footage": sf,
        "Number of Occupants": occ,
        "Appliances Used": app,
        "Average Temperature": tmp,
        "Day of Week": dow,
    }])

    X = preprocessor.transform(row)
    kwh = float(net.predict(X, verbose=0)[0, 0])

    st.success(f"Estimated consumption: **{kwh:,.0f} kWh**")

st.caption("Model : 2-layer NN • Features pre-processed via saved ColumnTransformer")
