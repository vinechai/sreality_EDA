import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---- Load artifacts ----
@st.cache_resource
def load_artifacts():
    model_obj = joblib.load("deployable/model.pkl")
    label_encoders = joblib.load("deployable/label_encoders.pkl")
    return model_obj, label_encoders

model_obj, label_encoders = load_artifacts()

# ---- Helpers ----
def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure input DataFrame has all expected columns, fill missing with NaN."""
    expected_cols = model_obj["models"]["CatBoost"].feature_names_ \
        if isinstance(model_obj, dict) else model_obj.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[expected_cols]

def apply_label_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply label encoders to categorical columns, fall back to 'NA_LE' if unseen."""
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .apply(lambda x: x if x in le.classes_ else "NA_LE")
            )
            df[col] = le.transform(df[col])
    return df

def inv_y_local(y_log):
    """Inverse of log target transform (exp)."""
    return np.expm1(y_log)

def predict_price(df_raw: pd.DataFrame):
    """Predict housing price from raw user input."""
    df_s = sanitize_columns(df_raw.copy())
    df_s = apply_label_encoders(df_s, label_encoders)

    if isinstance(model_obj, dict) and "weights" in model_obj:
        preds = []
        weights = []
        for name in model_obj["order"]:
            m = model_obj["models"][name]
            w = model_obj["weights"][name]
            yhat = m.predict(df_s)
            preds.append(yhat * w)
            weights.append(w)
        yhat_log = sum(preds) / sum(weights)
    else:
        yhat_log = model_obj.predict(df_s)

    return inv_y_local(yhat_log)

# ---- Streamlit UI ----
st.title("🏠 Housing Price Prediction")

st.write("Provide a few details about the property to estimate its price.")

inputs = {}

# Numerical input
inputs["square_meters"] = st.number_input("Size (m²)", min_value=10, max_value=500, value=50)

# Categorical inputs (from encoders)
for col in ["district", "ownership", "layout"]:
    if col in label_encoders:
        options = list(label_encoders[col].classes_)
        choice = st.selectbox(f"{col}", options)
        inputs[col] = choice
    else:
        st.warning(f"⚠️ Missing encoder for {col}")
        inputs[col] = "NA_LE"

# Predict button
if st.button("Predict Price"):
    incoming = pd.DataFrame([inputs])
    pred = predict_price(incoming)
    st.success(f"💰 Estimated price: {pred[0]:,.0f} CZK")
