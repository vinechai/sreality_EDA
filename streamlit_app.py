import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ======================
# Load Artifacts
# ======================
artifact_dir = Path(__file__).resolve().parent / "deployable"

@st.cache_resource
def load_artifacts():
    prep = joblib.load(artifact_dir / "preprocessing.joblib")
    model_obj = joblib.load(artifact_dir / "model.joblib")
    return prep, model_obj

prep, model_obj = load_artifacts()

sanit_map = prep["sanit_map"]
label_encoders = prep["label_encoders"]
reduced_feature_names = prep["reduced_feature_names"]
target_transform = prep["target_transform"]
feature_defaults = prep.get("feature_defaults", {})

# ======================
# Helpers
# ======================
def inv_y(y):
    if target_transform == "log1p":
        return np.expm1(y)
    elif target_transform == "log":
        return np.exp(y)
    return y

def sanitize_columns(df, sanit_map):
    """Ensure all training-time columns exist in input and rename consistently."""
    for col in sanit_map.keys():
        if col not in df.columns:
            df[col] = np.nan
    return df.rename(columns=sanit_map)

def apply_label_encoders(df, label_encoders):
    """Apply label encoders safely with fallback for unseen values."""
    for c, le in label_encoders.items():
        if c in df.columns:
            vals = df[c].astype("string").fillna("__MISSING__")
            safe_vals = []
            for v in vals:
                if v in le.classes_:
                    safe_vals.append(v)
                else:
                    safe_vals.append(le.classes_[0])  # fallback to first class
            df[c] = le.transform(safe_vals)
    return df

def safe_default(col, fallback=1):
    """Ensure defaults are valid for Streamlit inputs (no negatives/zeros where invalid)."""
    val = feature_defaults.get(col, fallback)
    try:
        if np.isnan(val):
            return fallback
        if isinstance(val, (int, float)):
            return max(fallback, int(val))
    except Exception:
        return fallback
    return val

def predict_price(df_raw: pd.DataFrame):
    df_s = sanitize_columns(df_raw.copy(), sanit_map)
    df_s = apply_label_encoders(df_s, label_encoders)

    # Reorder columns exactly as in training
    X = df_s.reindex(columns=reduced_feature_names, fill_value=0)

    # Handle blend model vs single model
    if isinstance(model_obj, dict) and "weights" in model_obj:
        preds = []
        for mname, base_model in model_obj["base_models"].items():
            preds.append(base_model.predict(X))
        yhat_log = np.vstack(preds).T @ model_obj["weights"]
    else:
        yhat_log = model_obj.predict(X)

    return inv_y(yhat_log)

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="🏠 Real Estate Price Predictor", layout="wide")
st.title("🏠 Real Estate Price Predictor")
st.markdown("Provide property details to estimate its price.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        usable_area = st.number_input(
            "Usable area (m²)", min_value=1, max_value=5000,
            value=safe_default("usable_area", 50)
        )
        total_area = st.number_input(
            "Total area (m²)", min_value=1, max_value=5000,
            value=safe_default("total_area", 50)
        )
        square_meters = st.number_input(
            "Square meters (m²)", min_value=1, max_value=5000,
            value=safe_default("square_meters", 50)
        )
        floor_number = st.number_input(
            "Floor number", min_value=0, max_value=50,
            value=safe_default("floor_number", 0)
        )
        total_floors = st.number_input(
            "Total floors", min_value=1, max_value=50,
            value=safe_default("total_floors", 1)
        )

    with col2:
        building = st.selectbox(
            "Building type",
            options=label_encoders["building"].classes_,
            index=0
        )
        layout = st.selectbox(
            "Layout",
            options=label_encoders["layout"].classes_,
            index=0
        )
        district = st.selectbox(
            "District",
            options=label_encoders["district"].classes_,
            index=0
        )
        property_status = st.selectbox(
            "Property status",
            options=label_encoders["property_status"].classes_,
            index=0
        )

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_data = pd.DataFrame([{
        "usable_area": usable_area,
        "total_area": total_area,
        "square_meters": square_meters,
        "floor_number": floor_number,
        "total_floors": total_floors,
        "building": building,
        "layout": layout,
        "district": district,
        "property_status": property_status
    }])

    try:
        prediction = predict_price(input_data)
        st.success(f"💰 Predicted Price: {prediction[0]:,.0f} CZK")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
