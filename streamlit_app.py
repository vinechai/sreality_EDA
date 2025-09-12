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
            if mname in reduced_feature_names:  
                cols = reduced_feature_names
                preds.append(base_models[mname].predict(df_s[cols]))
            else:
                preds.append(base_models[mname].predict(df_s))
        yhat_log = np.vstack(preds).T @ weights
    else:
        yhat_log = model_obj.predict(df_s)

    return inv_y_local(yhat_log)

# ---- Streamlit UI ----
st.title("🏠 Flat Price Predictor – Prague")

# Generate inputs dynamically from encoders
inputs = {}
for col, le in label_encoders.items():
    options = le.classes_
    choice = st.selectbox(f"{col}", options)
    inputs[col] = choice

# Add numerical fields (customize according to your dataset)
size = st.number_input("Size (m²)", min_value=10, max_value=300, value=50)
rooms = st.number_input("Rooms", min_value=1, max_value=10, value=2)
inputs["size"] = size
inputs["rooms"] = rooms

if st.button("Predict Price"):
    incoming = pd.DataFrame([inputs])
    pred = predict_price(incoming)
    st.success(f"💰 Estimated price: {pred[0]:,.0f} CZK")
