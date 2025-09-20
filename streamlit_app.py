# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Any, Dict, List
import pydeck as pdk

# -----------------------
# Config
# -----------------------
ARTIFACT_DIR = Path("deployable")
PREP_FILENAME = "preprocessing.joblib"
MODEL_FILENAMES = ["model.joblib", "model.pkl", "model.pkl.joblib", "model.joblib"]

st.set_page_config(layout="wide", page_title="🏠 Flat Price Predictor")

# -----------------------
# Utility helpers
# -----------------------
def load_artifacts(artifact_dir: Path = ARTIFACT_DIR):
    model_path = None
    for fn in MODEL_FILENAMES:
        p = artifact_dir / fn
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {artifact_dir}. Expected one of {MODEL_FILENAMES}")

    model_obj = joblib.load(model_path)

    prep_path = artifact_dir / PREP_FILENAME
    if not prep_path.exists():
        raise FileNotFoundError(f"No preprocessing file at {prep_path}. Please save preprocessing.joblib in {artifact_dir}")

    prep = joblib.load(prep_path)
    if not isinstance(prep, dict):
        raise ValueError("preprocessing.joblib must contain a dict (sanit_map, label_encoders, reduced_feature_names/feature_names, target_transform, feature_defaults)")

    return model_obj, prep

def get_model_feature_names(model_obj: Any, prep: Dict[str, Any]) -> List[str]:
    try:
        if hasattr(model_obj, "feature_names_"):
            return list(map(str, model_obj.feature_names_))
        if hasattr(model_obj, "get_feature_names_out"):
            return list(map(str, model_obj.get_feature_names_out()))
    except Exception:
        pass

    if isinstance(model_obj, dict):
        for candidate in ("base_models", "models", "base_fitted"):
            if candidate in model_obj and isinstance(model_obj[candidate], dict):
                for m in model_obj[candidate].values():
                    if hasattr(m, "feature_names_"):
                        return list(map(str, m.feature_names_))
                    if hasattr(m, "get_feature_names_out"):
                        try:
                            return list(map(str, m.get_feature_names_out()))
                        except Exception:
                            pass

    if "reduced_feature_names" in prep:
        return list(map(str, prep["reduced_feature_names"]))
    if "feature_names" in prep:
        return list(map(str, prep["feature_names"]))
    if "sanit_map" in prep:
        return list(map(str, prep["sanit_map"].values()))

    raise RuntimeError("Cannot determine model feature names.")

def safe_label_encode_scalar(val: Any, le) -> int:
    sval = "" if val is None else str(val)
    classes = list(map(str, getattr(le, "classes_", [])))
    if sval in classes:
        return int(le.transform([sval])[0])
    if "NA_LE" in classes:
        return int(le.transform(["NA_LE"])[0])
    if len(classes) > 0:
        return int(le.transform([classes[0]])[0])
    return int(le.transform([sval])[0])

def build_input_row(provided: Dict[str, Any], expected_cols: List[str], prep: Dict[str, Any]) -> pd.DataFrame:
    label_encoders: Dict[str, Any] = prep.get("label_encoders", {})
    defaults: Dict[str, Any] = prep.get("feature_defaults", {})

    row: Dict[str, Any] = {}
    for c in expected_cols:
        if c in label_encoders:
            if c in defaults:
                row[c] = defaults[c]
            else:
                classes = list(map(str, getattr(label_encoders[c], "classes_", [])))
                if "NA_LE" in classes:
                    row[c] = "NA_LE"
                elif len(classes) > 0:
                    row[c] = classes[0]
                else:
                    row[c] = "NA_LE"
        else:
            row[c] = defaults.get(c, 0)

    for k, v in provided.items():
        if k in expected_cols:
            row[k] = v

    for c, le in label_encoders.items():
        if c in expected_cols:
            row[c] = safe_label_encode_scalar(row[c], le)

    df = pd.DataFrame([row], columns=expected_cols)
    for c in df.columns:
        if c not in label_encoders:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def inv_target_transform(arr: np.ndarray, tgt: str):
    if tgt == "log1p":
        return np.expm1(arr)
    if tgt == "log":
        return np.exp(arr)
    return arr

# -----------------------
# Load artifacts
# -----------------------
model_obj, prep = load_artifacts()
sanit_map = prep.get("sanit_map", {})
label_encoders = prep.get("label_encoders", {})
feature_defaults = prep.get("feature_defaults", {}) or {}
target_transform = prep.get("target_transform", "log")
model_feature_names = get_model_feature_names(model_obj, prep)

# -----------------------
# Layout ↔ size ranges
# -----------------------
LAYOUT_SIZE_RANGES = {
    "1+kk": (20, 50),
    "1+1": (25, 55),
    "2+kk": (35, 70),
    "2+1": (40, 75),
    "3+kk": (60, 100),
    "3+1": (65, 110),
    "4+kk": (90, 150),
    "4+1": (95, 160),
    "5+kk": (110, 180),
    "5+1": (120, 200),
}

# -----------------------
# UI
# -----------------------
st.title("🏠 Flat Price Predictor — Prague")

col1, col2, col3 = st.columns(3)
with col1:
    usable_area = st.number_input(
        "Usable area (m²)", min_value=10, max_value=500,
        value=int(max(20, int(round(feature_defaults.get("usable_area", 50)))))
    )
    # sync automatically
    square_meters = usable_area
    total_area = usable_area

with col2:
    district_opts = list(map(str, label_encoders.get("district", []).classes_)) if "district" in label_encoders else ["Praha 1"]
    district = st.selectbox("District", district_opts, index=0)

with col3:
    layout_opts = list(map(str, label_encoders.get("layout", []).classes_)) if "layout" in label_encoders else ["1+kk", "2+kk", "3+kk"]
    layout = st.selectbox("Layout", layout_opts, index=0)

with st.expander("Additional features"):
    terrace = st.number_input("Terrace size (m²)", min_value=0, max_value=200, value=0)
    garage = st.number_input("Garage size (m²)", min_value=0, max_value=100, value=0)
    cellar = st.number_input("Cellar size (m²)", min_value=0, max_value=50, value=0)

# -----------------------
# Prediction
# -----------------------
provided = {
    "usable_area": usable_area,
    "square_meters": square_meters,
    "total_area": total_area,
    "district": district,
    "layout": layout,
    "terrace": terrace,
    "garage": garage,
    "cellar": cellar,
}

if st.button("Predict price"):
    try:
        df_in = build_input_row(provided, model_feature_names, prep)
        df_in = df_in.reindex(columns=model_feature_names, fill_value=0)

        if not isinstance(model_obj, dict):
            preds_log = model_obj.predict(df_in)
        else:
            base_dict = model_obj.get("base_models") or model_obj.get("models") or {}
            preds_components = []
            for m in base_dict.values():
                cols_for_m = list(map(str, getattr(m, "feature_names_", model_feature_names)))
                preds_components.append(np.asarray(m.predict(df_in.reindex(columns=cols_for_m, fill_value=0))).ravel())
            preds_log = np.mean(np.vstack(preds_components), axis=0)

        preds = inv_target_transform(np.asarray(preds_log).ravel(), target_transform)
        price = float(preds[0])
        st.write("Model input row:", df_in)
        st.write("Model features:", model_feature_names)
        st.success(f"💰 Estimated price: {price:,.0f} CZK")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # -----------------------
    # Map: baseline prices by district
    # -----------------------
    st.subheader("🗺 District price map")
    
    DISTRICT_COORDS = {
        "Praha 1": (50.087, 14.421),
        "Praha 2": (50.071, 14.436),
        "Praha 3": (50.082, 14.454),
        "Praha 4": (50.036, 14.428),
        "Praha 5": (50.067, 14.389),
        "Praha 6": (50.099, 14.366),
        "Praha 7": (50.107, 14.449),
        "Praha 8": (50.108, 14.474),
        "Praha 9": (50.112, 14.514),
        "Praha 10": (50.070, 14.488),
    }
    
    district_baselines = []
    flat_size = 60  # reference size
    for d in district_opts:
        prov = provided.copy()
        prov["district"] = d
        prov["usable_area"] = prov["square_meters"] = prov["total_area"] = flat_size
        df_tmp = build_input_row(prov, model_feature_names, prep)
        df_tmp = df_tmp.reindex(columns=model_feature_names, fill_value=0)
        try:
            plog = model_obj.predict(df_tmp) if not isinstance(model_obj, dict) else np.mean(
                [m.predict(df_tmp) for m in model_obj.get("base_models", {}).values()], axis=0
            )
            price = inv_target_transform(np.asarray(plog).ravel(), target_transform)[0]
            price_m2 = price / flat_size
            coords = DISTRICT_COORDS.get(d, (50.08, 14.42))
            district_baselines.append({"district": d, "lat": coords[0], "lon": coords[1],
                                       "price": price, "price_m2": price_m2})
        except Exception:
            continue
    
    df_map = pd.DataFrame(district_baselines)
        
    # Normalize price per m²
    min_p, max_p = df_map["price_m2"].min(), df_map["price_m2"].max()
    df_map["price_norm"] = (
        (df_map["price_m2"] - min_p) / (max_p - min_p) if max_p > min_p else 0.5
    )
    
    # Colors & sizes
    df_map["color_r"] = (50 + df_map["price_norm"] * 200).clip(0, 255)
    df_map["color_g"] = (200 - df_map["price_norm"] * 150).clip(0, 255)
    df_map["color_b"] = 80
    df_map["color_a"] = 180
    df_map["radius"] = df_map["price_norm"] * 300 + 100
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["lon", "lat"],
        get_fill_color=["color_r", "color_g", "color_b", "color_a"],
        get_radius="radius",
        pickable=True,
    )
    
    view_state = pdk.ViewState(latitude=50.08, longitude=14.42, zoom=11)
    
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{district}\n{price_m2} CZK/m²"},
        )
    )


    # -----------------------
    # Sensitivity plot
    # -----------------------
    st.subheader("📊 Sensitivity: Price vs. Usable area")
    min_size, max_size = LAYOUT_SIZE_RANGES.get(layout, (20, 120))
    sizes = np.linspace(min_size, max_size, 15)
    prices = []
    for s in sizes:
        prov = provided.copy()
        prov["usable_area"] = prov["square_meters"] = prov["total_area"] = s
        df_tmp = build_input_row(prov, model_feature_names, prep)
        df_tmp = df_tmp.reindex(columns=model_feature_names, fill_value=0)
        try:
            plog = model_obj.predict(df_tmp) if not isinstance(model_obj, dict) else np.mean(
                [m.predict(df_tmp) for m in model_obj.get("base_models", {}).values()], axis=0
            )
            p = inv_target_transform(np.asarray(plog).ravel(), target_transform)[0]
            prices.append(p)
        except Exception:
            prices.append(None)
    st.line_chart(pd.DataFrame({"Price": prices}, index=sizes))
