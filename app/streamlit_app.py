# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Any, Dict, List
import pydeck as pdk

# Config
ARTIFACT_DIR = Path("models")
MODEL_PATH = ARTIFACT_DIR / "final_model.joblib"
PREP_PATH = ARTIFACT_DIR / "preprocessing.joblib"

model_obj = joblib.load(MODEL_PATH)
prep = joblib.load(PREP_PATH)

"""
ARTIFACT_DIR = Path("models")
PREP_FILENAME = "preprocessing.joblib"
MODEL_FILENAMES = ["model.joblib", "model.pkl", "model.pkl.joblib", "model.joblib"]

st.set_page_config(layout="wide", page_title="🏠 Flat Price Predictor")


# Utility helpers
def load_artifacts(artifact_dir: Path = ARTIFACT_DIR):
    model_path = None
    for fn in MODEL_FILENAMES:
        p = artifact_dir / fn
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(
            f"No model file found in {artifact_dir}. Expected one of {MODEL_FILENAMES}"
        )

    model_obj = joblib.load(model_path)

    prep_path = artifact_dir / PREP_FILENAME
    if not prep_path.exists():
        raise FileNotFoundError(
            f"No preprocessing file at {prep_path}. Please save preprocessing.joblib in {artifact_dir}"
        )

    prep = joblib.load(prep_path)
    if not isinstance(prep, dict):
        raise ValueError(
            "preprocessing.joblib must contain a dict (sanit_map, label_encoders, "
            "reduced_feature_names/feature_names, target_transform, feature_defaults)"
        )

    return model_obj, prep
"""

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

"""
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
"""

def build_input_row(
    provided: Dict[str, Any], expected_cols: List[str], prep: Dict[str, Any]
) -> pd.DataFrame:
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


def predict_with_blend(model_obj, df_in, model_feature_names):
    """Handle dict-style blended model with weights and order"""
    if not isinstance(model_obj, dict):
        return model_obj.predict(df_in)

    base_dict = model_obj.get("base_models") or model_obj.get("models") or {}
    order = model_obj.get("order", list(base_dict.keys()))
    weights = model_obj.get("weights", None)

    if isinstance(weights, dict):
        w = np.array([float(weights.get(n, 0.0)) for n in order], dtype=float)
    else:
        w = (
            np.asarray(weights, dtype=float)
            if weights is not None
            else np.ones(len(order), dtype=float)
        )

    if w.shape[0] != len(order):
        w = np.ones(len(order), dtype=float)
    w[w < 0] = 0.0
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()

    preds_components = []
    for i, name in enumerate(order):
        if name not in base_dict:
            continue
        mdl = base_dict[name]
        cols_for_base = list(
            map(str, getattr(mdl, "feature_names_", model_feature_names))
        )
        Xb = df_in.reindex(columns=cols_for_base, fill_value=0)
        p = np.asarray(mdl.predict(Xb)).ravel()
        preds_components.append(p * w[i])
    return np.sum(np.vstack(preds_components), axis=0)


# Load artifacts
#model_obj, prep = load_artifacts()
label_encoders = prep.get("label_encoders", {})
feature_defaults = prep.get("feature_defaults", {}) or {}
target_transform = prep.get("target_transform", "log")
model_feature_names = get_model_feature_names(model_obj, prep)

# Layout and size ranges
LAYOUT_SIZE_RANGES = {
    "1+kt": (20, 50),
    "1+1": (25, 55),
    "2+kt": (35, 70),
    "2+1": (40, 75),
    "3+kt": (60, 100),
    "3+1": (65, 110),
    "4+kt": (90, 150),
    "4+1": (95, 160),
    "5+kt": (110, 180),
    "5+1": (120, 200),
    "6+": (130, 200),
}

DISTRICT_COORDS = {
    "Praha 1": (50.087, 14.421),
    "Praha 2": (50.071, 14.434),
    "Praha 3": (50.084, 14.455),
    "Praha 4": (50.038, 14.437),
    "Praha 5": (50.065, 14.389),
    "Praha 6": (50.098, 14.363),
    "Praha 7": (50.107, 14.450),
    "Praha 8": (50.109, 14.474),
    "Praha 9": (50.111, 14.515),
    "Praha 10": (50.072, 14.490),
    "Praha 11": (50.029, 14.514),
    "Praha 12": (50.005, 14.451),
    "Praha 13": (50.051, 14.335),
    "Praha 14": (50.103, 14.551),
    "Praha 15": (50.080, 14.558),
    "Missing": (50.167, 14.400),
}

# UI
st.title("Prague Flat Price Predictor")

col1, col2, col3 = st.columns(3)
with col1:
    usable_area = st.number_input(
        "Usable area (m²)",
        min_value=10,
        max_value=500,
        value=int(max(20, int(round(feature_defaults.get("usable_area", 50))))),
    )
    square_meters = total_area = floorage = usable_area

with col2:
    district_opts = (
        list(map(str, label_encoders.get("district", []).classes_))
        if "district" in label_encoders
        else ["Praha 1"]
    )
    district = st.selectbox("District", district_opts, index=0)

with col3:
    layout_opts = (
        list(map(str, label_encoders.get("layout", []).classes_))
        if "layout" in label_encoders
        else ["1+kk", "2+kk", "3+kk"]
    )
    layout = st.selectbox("Layout", layout_opts, index=0)

col4, col5 = st.columns(2)
with col4:
    ownership_opts = (
        list(map(str, label_encoders.get("ownership", []).classes_))
        if "ownership" in label_encoders
        else ["Personal"]
    )
    ownership = st.selectbox("Ownership", ownership_opts, index=0)
with col5:
    building_opts = (
        list(map(str, label_encoders.get("building", []).classes_))
        if "building" in label_encoders
        else ["Brick", "Panel", "Other"]
    )
    building = st.selectbox("Building", building_opts, index=0)

with st.expander("Additional features"):
    terrace = st.number_input("Terrace size (m²)", min_value=0, max_value=200, value=0)
    garage = st.number_input("Garage size (m²)", min_value=0, max_value=100, value=0)
    cellar = st.number_input("Cellar size (m²)", min_value=0, max_value=50, value=0)

# Prediction
provided = {
    "usable_area": usable_area,
    "square_meters": square_meters,
    "total_area": total_area,
    "floorage": floorage,
    "district": district,
    "layout": layout,
    "ownership": ownership,
    "building": building,
    "terrace": terrace,
    "garage": garage,
    "cellar": cellar,
}

# Warn for unrealistic combos
if layout in LAYOUT_SIZE_RANGES:
    min_s, max_s = LAYOUT_SIZE_RANGES[layout]
    if usable_area < min_s or usable_area > max_s * 1.5:
        st.warning(
            f" The selected layout {layout} usually has between {min_s}–{max_s} m². "
            f"Your input ({usable_area} m²) looks unusual."
        )

if st.button("Predict price"):
    try:
        df_in = build_input_row(provided, model_feature_names, prep)
        df_in = df_in.reindex(columns=model_feature_names, fill_value=0)

        preds_log = predict_with_blend(model_obj, df_in, model_feature_names)
        preds = inv_target_transform(np.asarray(preds_log).ravel(), target_transform)

        price = float(preds[0])
        st.success(f" Estimated price: {price:,.0f} CZK")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # Map: baseline prices by district
    st.subheader("District price map")
    district_baselines = []
    flat_size = 60  # reference size
    reference_layouts = ["1+kt", "1+1", "2+kt", "2+1", "3+kt", "2+1", "4+kt", "4+1"]  # layouts to average over

    for d in district_opts:
        layout_prices = []
        for lay in reference_layouts:
            prov = provided.copy()
            prov.update({
                "district": d,
                "usable_area": flat_size,
                "square_meters": flat_size,
                "total_area": flat_size,
                "layout": lay,
                "terrace": 0,
                "garage": 0,
                "cellar": 0
            })
            df_tmp = build_input_row(prov, model_feature_names, prep)
            df_tmp = df_tmp.reindex(columns=model_feature_names, fill_value=0)
            try:
                plog = predict_with_blend(model_obj, df_tmp, model_feature_names)
                price = inv_target_transform(np.asarray(plog).ravel(), target_transform)[0]
                layout_prices.append(price / flat_size)
            except Exception:
                continue

        if layout_prices:  # take average across layouts
            avg_price_m2 = np.mean(layout_prices)
            coords = DISTRICT_COORDS.get(d, (50.08, 14.42))
            district_baselines.append({
                "district": d,
                "lat": coords[0],
                "lon": coords[1],
                "price_m2": avg_price_m2
            })


    df_map = pd.DataFrame(district_baselines)

    if not df_map.empty:
        min_p, max_p = df_map["price_m2"].min(), df_map["price_m2"].max()
        df_map["price_norm"] = (
            (df_map["price_m2"] - min_p) / (max_p - min_p) if max_p > min_p else 0.5
        )

        df_map["color_r"] = (50 + df_map["price_norm"] * 200).clip(0, 255)
        df_map["color_g"] = (200 - df_map["price_norm"] * 150).clip(0, 255)
        df_map["color_b"] = 80
        df_map["color_a"] = 180
        df_map["radius"] = df_map["price_norm"] * 600 + 200

        # Main layer (all districts, colored by price)
        layer_main = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_fill_color=["color_r", "color_g", "color_b", "color_a"],
            get_radius="radius",
            pickable=True,
        )

        # Halo layer (only selected district)
        df_selected = df_map[df_map["district"] == district]
        halo_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_selected,
            get_position=["lon", "lat"],
            get_fill_color=[255, 255, 255, 100],  # semi-transparent white
            get_radius="radius*1.6",              # a bit larger than main circle
        )

        view_state = pdk.ViewState(
            latitude=df_selected["lat"].iloc[0] if not df_selected.empty else 50.08,
            longitude=df_selected["lon"].iloc[0] if not df_selected.empty else 14.42,
            zoom=12,
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer_main, halo_layer],
            initial_view_state=view_state,
            tooltip={"text": "{district}\n{price_m2} CZK/m²"},
        ))

    # Sensitivity plot
    st.subheader("Sensitivity: Price vs. Usable area")
    min_size, max_size = LAYOUT_SIZE_RANGES.get(layout, (20, 120))
    sizes = np.linspace(min_size, max_size, 15)
    prices = []
    for s in sizes:
        prov = provided.copy()
        prov.update({"usable_area": s, "square_meters": s, "total_area": s})
        df_tmp = build_input_row(prov, model_feature_names, prep)
        df_tmp = df_tmp.reindex(columns=model_feature_names, fill_value=0)
        try:
            plog = predict_with_blend(model_obj, df_tmp, model_feature_names)
            p = inv_target_transform(np.asarray(plog).ravel(), target_transform)[0]
            prices.append(p)
        except Exception:
            prices.append(None)

    st.line_chart(pd.DataFrame({"Price": prices}, index=sizes))
