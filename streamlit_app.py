# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ---------------------
# Load artifacts
# ---------------------
artifact_dir = Path("deployable")
prep = joblib.load(artifact_dir / "preprocessing.joblib")
model_obj = joblib.load(artifact_dir / "model.joblib")

sanit_map = prep["sanit_map"]
label_encoders = prep["label_encoders"]
reduced_feature_names = prep["reduced_feature_names"]
tgt = prep["target_transform"]

winner_name = "Blend" if isinstance(model_obj, dict) else "Single"

def inv_y(y):
    if tgt == "log1p":
        return np.expm1(y)
    elif tgt == "log":
        return np.exp(y)
    return y

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

def predict_price(df_raw: pd.DataFrame):
    df_s = sanitize_columns(df_raw.copy())
    df_s = apply_label_encoders(df_s)

    if winner_name == "Blend":
        order = model_obj["order"]
        weights = model_obj["weights"]
        base_models = model_obj["base_models"]
        preds = []
        for mname in order:
            if mname in reduced_feature_names:
                preds.append(base_models[mname].predict(df_s[reduced_feature_names]))
            else:
                preds.append(base_models[mname].predict(df_s))
        yhat_log = np.vstack(preds).T @ weights
    else:
        yhat_log = model_obj.predict(df_s if "reduced_feature_names" not in prep else df_s[reduced_feature_names])

    return inv_y(yhat_log)

# ---------------------
# Streamlit UI
# ---------------------
st.title("🏠 Flat Price Predictor - Prague")

st.write("Enter flat details to predict price (CZK).")

rooms = st.number_input("Number of rooms", min_value=1, max_value=10, value=2)
size = st.number_input("Size (m²)", min_value=15, max_value=300, value=60)
options = prep["label_encoders"]["district"].classes_
district = st.selectbox("District", options)

if st.button("Predict price"):
    incoming = pd.DataFrame([{
        "rooms": rooms,
        "size": size,
        "district": district
    }])
    pred = predict_price(incoming)
    st.success(f"Estimated price: {pred[0]:,.0f} CZK")
