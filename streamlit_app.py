import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

# ---- Load artifacts ----
artifact_dir = Path("deployable")
prep = joblib.load(artifact_dir / "preprocessing.joblib")
model_obj = joblib.load(artifact_dir / "model.joblib")

sanit_map = prep["sanit_map"]
label_encoders = prep["label_encoders"]
reduced_feature_names = prep["reduced_feature_names"]
tgt = prep["target_transform"]

# ---- Preprocessing helpers ----
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

    if isinstance(model_obj, dict) and "weights" in model_obj:  # Blend
        order = model_obj["order"]
        weights = model_obj["weights"]
        base_models = model_obj["base_models"]
        preds = []
        for mname in order:
            preds.append(base_models[mname].predict(df_s[reduced_feature_names]))
        yhat_log = np.vstack(preds).T @ weights
    else:
        yhat_log = model_obj.predict(df_s[reduced_feature_names])

    return inv_y_local(yhat_log)

# ---- Streamlit UI ----
st.title("🏠 Flat Price Predictor – Prague")

# Build inputs ONLY from features known by the model
inputs = {}
for col in reduced_feature_names:
    if col in label_encoders:  # categorical
        options = label_encoders[col].classes_
        inputs[col] = st.selectbox(col, options)
    else:
        # numeric columns
        inputs[col] = st.number_input(col, value=0.0)

if st.button("Predict Price"):
    incoming = pd.DataFrame([inputs])
    pred = predict_price(incoming)
    st.success(f"💰 Estimated price: {pred[0]:,.0f} CZK")
