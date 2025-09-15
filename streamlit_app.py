# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any

# -----------------------
# CONFIG
# -----------------------
ARTIFACT_DIR = Path("deployable")
MODEL_FILENAMES = ["model.joblib", "model.pkl", "model.pkl.joblib"]
PREP_FILENAME = "preprocessing.joblib"

st.set_page_config(layout="wide", page_title="🏠 Flat Price Predictor")

# -----------------------
# Helpers
# -----------------------
@st.cache_resource
def load_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> Tuple[Any, Dict[str, Any]]:
    # 1) model
    model_path = None
    for fn in MODEL_FILENAMES:
        p = artifact_dir / fn
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"No model found in {artifact_dir}")

    model_obj = joblib.load(model_path)

    # 2) preprocessing
    prep_path = artifact_dir / PREP_FILENAME
    if not prep_path.exists():
        raise FileNotFoundError(f"No preprocessing file at {prep_path}")

    prep = joblib.load(prep_path)
    if not isinstance(prep, dict):
        raise ValueError("preprocessing.joblib should contain a dict")

    return model_obj, prep


def safe_label_encode(val, col, label_encoders, defaults):
    if col not in label_encoders:
        return val
    le = label_encoders[col]
    classes = list(map(str, le.classes_))
    if str(val) not in classes:
        if "NA_LE" in classes:
            val = "NA_LE"
        else:
            val = defaults.get(col, classes[0])
    return le.transform([str(val)])[0]


def build_input_dataframe(provided: dict, expected_cols: list, prep: dict) -> pd.DataFrame:
    label_encoders = prep.get("label_encoders", {})
    defaults = prep.get("feature_defaults", {})
    df = pd.DataFrame(index=[0], columns=expected_cols)

    # fill defaults
    for c in expected_cols:
        if c in label_encoders:
            df.at[0, c] = defaults.get(c, "NA_LE")
        else:
            df.at[0, c] = defaults.get(c, 0)

    # override with provided
    for k, v in provided.items():
        if k in df.columns:
            df.at[0, k] = v

    # encode categoricals
    for c in label_encoders:
        if c in df.columns:
            df[c] = safe_label_encode(df[c].iloc[0], c, label_encoders, defaults)

    # numerics
    for c in df.columns:
        if c not in label_encoders:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


def inv_target_transform(y, tgt):
    if tgt == "log1p": return np.expm1(y)
    if tgt == "log": return np.exp(y)
    return y

# -----------------------
# Load artifacts
# -----------------------
try:
    model_obj, prep = load_artifacts()
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

label_encoders = prep.get("label_encoders", {})
defaults = prep.get("feature_defaults", {})
target_transform = prep.get("target_transform", "log")

# 🔑 Use the model’s own feature names if available
if hasattr(model_obj, "feature_names_"):
    model_features = model_obj.feature_names_
else:
    model_features = prep.get("feature_names", list(prep.get("sanit_map", {}).values()))

# -----------------------
# UI
# -----------------------
st.title("🏠 Flat Price Predictor — Prague")

col1, col2, col3 = st.columns(3)
with col1:
    usable_area = st.number_input("Size (m²)", 5, 2000, 50)
with col2:
    district = st.selectbox("District", label_encoders.get("district").classes_
                            if "district" in label_encoders else ["NA_LE"])
with col3:
    ownership = st.selectbox("Ownership", label_encoders.get("ownership").classes_
                             if "ownership" in label_encoders else ["NA_LE"])

layout = st.selectbox("Layout", label_encoders.get("layout").classes_
                      if "layout" in label_encoders else ["NA_LE"])

# advanced section (as flags for simplicity)
with st.expander("Advanced options"):
    terrace = st.checkbox("Has Terrace?")
    garage = st.checkbox("Has Garage?")
    cellar = st.checkbox("Has Cellar?")

# -----------------------
# Prediction
# -----------------------
provided = {
    "usable_area": usable_area,
    "district": district,
    "ownership": ownership,
    "layout": layout,
    "terrace": int(terrace),
    "garage": int(garage),
    "cellar": int(cellar),
}

if st.button("Predict price"):
    df_in = build_input_dataframe(provided, model_features, prep)
    df_in = df_in[model_features]   # ensure correct order

    # sanity check: unrealistic combos
    if str(layout).startswith("6") and usable_area < 60:
        st.warning("⚠️ Unusual combo: very small area with 6+ rooms")

    try:
        if hasattr(model_obj, "predict"):
            preds = model_obj.predict(df_in)
        else:
            # blend dict, simple fallback
            preds = np.mean([m.predict(df_in) for m in model_obj.get("base_models", {}).values()], axis=0)

        yhat = inv_target_transform(preds, target_transform)
        price = float(yhat.ravel()[0])
        st.success(f"💰 Estimated price: {price:,.0f} CZK")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # map (dummy coords per district)
    coords = {
        "Prague 1": (50.088, 14.420),
        "Prague 2": (50.071, 14.437),
        "Prague 5": (50.067, 14.395),
    }
    if str(district) in coords:
        st.map(pd.DataFrame([{"lat": coords[district][0], "lon": coords[district][1]}]))

    # scatter plot example
    st.subheader("📊 Sensitivity: Price vs. Size (fixed district/layout)")
    sizes = np.linspace(20, 200, 20)
    preds_curve = []
    for s in sizes:
        prov = provided.copy()
        prov["usable_area"] = s
        df_tmp = build_input_dataframe(prov, model_features, prep)
        df_tmp = df_tmp[model_features]
        preds_curve.append(inv_target_transform(model_obj.predict(df_tmp), target_transform))
    st.line_chart(pd.DataFrame({"Size": sizes, "Price": np.ravel(preds_curve)}).set_index("Size"))

# -----------------------
# Preprocessing summary
# -----------------------
with st.expander("Preprocessing summary"):
    st.write("Encoders:", list(label_encoders.keys()))
    st.write("Expected features (first 10):", model_features[:10])
    st.write("Defaults (sample):", {k: defaults[k] for k in list(defaults)[:5]})
