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
    # Load model
    model_path = None
    for fn in MODEL_FILENAMES:
        p = artifact_dir / fn
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"No model found in {artifact_dir}")

    model_obj = joblib.load(model_path)

    # Load preprocessing
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
model_features = prep.get("reduced_feature_names", list(prep.get("sanit_map", {}).values()))

# -----------------------
# UI
# -----------------------
st.title("🏠 Flat Price Predictor — Prague")

col1, col2, col3 = st.columns(3)
with col1:
    usable_area = st.number_input("Usable area (m²)", 5, 2000, int(defaults.get("usable_area", 50)))
with col2:
    total_area = st.number_input("Total area (m²)", 5, 2000, int(defaults.get("total_area", 50)))
with col3:
    floorage = st.number_input("Floorage (floor number)", 0, 50, int(defaults.get("floorage", 1)))

col4, col5 = st.columns(2)
with col4:
    district = st.selectbox("District", label_encoders.get("district").classes_
                            if "district" in label_encoders else ["NA_LE"])
with col5:
    building = st.selectbox("Building type", label_encoders.get("building").classes_
                            if "building" in label_encoders else ["NA_LE"])

ownership = st.selectbox("Ownership", label_encoders.get("ownership").classes_
                         if "ownership" in label_encoders else ["NA_LE"])
layout = st.selectbox("Layout", label_encoders.get("layout").classes_
                      if "layout" in label_encoders else ["NA_LE"])
cadastral_area = st.selectbox("Cadastral area", label_encoders.get("cadastral_area").classes_
                              if "cadastral_area" in label_encoders else ["NA_LE"])

# advanced section
with st.expander("Advanced options"):
    terrace = st.checkbox("Terrace")
    garage = st.checkbox("Garage")
    cellar = st.checkbox("Cellar")

# -----------------------
# Prediction
# -----------------------
provided = {
    "usable_area": usable_area,
    "total_area": total_area,
    "floorage": floorage,
    "district": district,
    "building": building,
    "ownership": ownership,
    "layout": layout,
    "cadastral_area": cadastral_area,
    "terrace": int(terrace),
    "garage": int(garage),
    "cellar": int(cellar),
}

if st.button("Predict price"):
    df_in = build_input_dataframe(provided, model_features, prep)

    try:
        if hasattr(model_obj, "predict"):
            preds = model_obj.predict(df_in[model_features])
        else:
            preds = np.mean([m.predict(df_in[model_features])
                             for m in model_obj.get("base_models", {}).values()], axis=0)
        yhat = inv_target_transform(preds, target_transform)
        price = float(yhat.ravel()[0])
        st.success(f"💰 Estimated price: {price:,.0f} CZK")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # Map (approx coords for districts)
    coords = {
        "Prague 1": (50.088, 14.420),
        "Prague 2": (50.071, 14.437),
        "Prague 5": (50.067, 14.395),
    }
    if str(district) in coords:
        st.map(pd.DataFrame([{"lat": coords[district][0], "lon": coords[district][1]}]))

    # Sensitivity curve
    st.subheader("📊 Sensitivity: Price vs. Usable area")
    sizes = np.linspace(20, 200, 20)
    preds_curve = []
    for s in sizes:
        prov = provided.copy()
        prov["usable_area"] = s
        df_tmp = build_input_dataframe(prov, model_features, prep)
        try:
            preds_curve.append(inv_target_transform(model_obj.predict(df_tmp[model_features]), target_transform))
        except Exception:
            preds_curve.append(np.nan)
    st.line_chart(pd.DataFrame({"Usable area": sizes, "Price": np.ravel(preds_curve)}).set_index("Usable area"))

# -----------------------
# Preprocessing summary
# -----------------------
with st.expander("Preprocessing summary"):
    st.write("Encoders:", list(label_encoders.keys()))
    st.write("Expected features (first 10):", model_features[:10])

    if defaults:
        df_defaults = pd.DataFrame([
            {"Feature": k, "Default": v, "Type": "categorical" if k in label_encoders else "numeric"}
            for k, v in list(defaults.items())[:10]
        ])
        st.dataframe(df_defaults)
    else:
        st.warning("⚠️ No defaults found in preprocessing.joblib")
