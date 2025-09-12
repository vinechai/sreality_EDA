import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ---- Load artifacts ----
artifact_dir = Path("deployable")
prep = joblib.load(artifact_dir / "preprocessing.joblib")
model_obj = joblib.load(artifact_dir / "model.joblib")

sanit_map = prep["sanit_map"]
label_encoders = prep["label_encoders"]
tgt = prep["target_transform"]

# ---- Helpers ----
def sanitize_columns(df):
    missing = set(sanit_map.keys()) - set(df.columns)
    for m in missing:
        df[m] = np.nan
    return df.rename(columns=sanit_map)

def apply_label_encoders(df, label_encoders):
    for c, le in label_encoders.items():
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA_LE")
            df[c] = df[c].apply(lambda x: x if x in le.classes_ else "NA_LE")
            df[c] = le.transform(df[c])
    return df

def inv_y_local(y):
    if tgt == "log1p": return np.expm1(y)
    if tgt == "log": return np.exp(y)
    return y

def predict_price(df_raw: pd.DataFrame):
    df_s = sanitize_columns(df_raw.copy())
    df_s = apply_label_encoders(df_s, label_encoders)
    yhat_log = model_obj.predict(df_s)
    return inv_y_local(yhat_log)

# ---- Streamlit UI ----
st.title("🏠 Flat Price Predictor – Prague")

inputs = {}

# Numeric features (sliders/number inputs)
numeric_features = [
    "usable_area", "floorage", "balcony", "terrace", "cellar", "parking",
    "garden_area", "garage", "loggia", "final_building_approval_year",
    "basin_area", "floor_number", "total_floors", "square_meters",
    "building_age", "total_area"
]
for col in numeric_features:
    inputs[col] = st.number_input(col, value=0.0)

# Categorical features (dropdowns)
for col, le in label_encoders.items():
    options = le.classes_
    inputs[col] = st.selectbox(col, options)

# Predict button
if st.button("Predict Price"):
    incoming = pd.DataFrame([inputs])
    pred = predict_price(incoming)
    st.success(f"💰 Estimated price: {pred[0]:,.0f} CZK")
