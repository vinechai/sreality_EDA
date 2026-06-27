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


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    p = joblib.load(PREP_PATH)
    return model, p


model_obj, prep = load_artifacts()



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

def encode_value(val, le):
    return int(le.transform([str(val)])[0])

def build_input_row(
    provided: Dict[str, Any], expected_cols: List[str], prep: Dict[str, Any],
    encode_cats: bool = True
) -> pd.DataFrame:
    label_encoders: Dict[str, Any] = prep.get("label_encoders", {})
    defaults: Dict[str, Any] = prep.get("feature_defaults", {}) or {}

    row: Dict[str, Any] = {}
    for c in expected_cols:
        if c in label_encoders:
            default_val = defaults.get(c)
            if default_val is not None:
                if not encode_cats and isinstance(default_val, (int, float)):
                    # reverse-decode integer default to string for native CatBoost
                    try:
                        row[c] = str(label_encoders[c].classes_[int(default_val)])
                    except (IndexError, TypeError):
                        row[c] = str(label_encoders[c].classes_[0])
                else:
                    row[c] = default_val
            else:
                classes = list(map(str, getattr(label_encoders[c], "classes_", [])))
                row[c] = classes[0] if classes else "NA"
        else:
            row[c] = defaults.get(c, 0)

    for k, v in provided.items():
        if k in expected_cols:
            row[k] = v

    if encode_cats:
        for c, le in label_encoders.items():
            if c in expected_cols and isinstance(row[c], str):
                row[c] = encode_value(row[c], le)

    df = pd.DataFrame([row], columns=expected_cols)
    for c in df.columns:
        if c not in label_encoders or encode_cats:
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


label_encoders = prep.get("label_encoders", {})
feature_defaults = prep.get("feature_defaults", {}) or {}
target_transform = prep.get("target_transform", "log")
model_feature_names = get_model_feature_names(model_obj, prep)

# detect native categorical encoding (CatBoost trained with cat_features)
_uses_native_cats = (
    type(model_obj).__name__ == "CatBoostRegressor"
    and hasattr(model_obj, "get_params")
    and bool(model_obj.get_params().get("cat_features"))
)
_encode_cats = not _uses_native_cats

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
    "Missing": (50.131, 14.421),
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
        else ["1+kt", "2+kt", "3+kt"]
    )
    layout = st.selectbox("Layout", layout_opts, index=0)

col4 = st.columns(1)[0]
with col4:
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
        df_in = build_input_row(provided, model_feature_names, prep, encode_cats=_encode_cats)
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
    reference_layouts = ["1+kt", "1+1", "2+kt", "2+1", "3+kt", "3+1", "4+kt", "4+1"]

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
            df_tmp = build_input_row(prov, model_feature_names, prep, encode_cats=_encode_cats)
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
            (np.log(df_map["price_m2"]) - np.log(min_p)) / (np.log(max_p) - np.log(min_p))
        )

        df_map["color_r"] = (80 + df_map["price_norm"] * 175).clip(0, 255)
        df_map["color_g"] = (220 - df_map["price_norm"] * 170).clip(0, 255)
        df_map["color_b"] = (120 - df_map["price_norm"] * 60).clip(0, 255)
        df_map["color_a"] = 200
        df_map["radius"] = df_map["price_norm"] * 350 + 350

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
        df_tmp = build_input_row(prov, model_feature_names, prep, encode_cats=_encode_cats)
        df_tmp = df_tmp.reindex(columns=model_feature_names, fill_value=0)
        try:
            plog = predict_with_blend(model_obj, df_tmp, model_feature_names)
            p = inv_target_transform(np.asarray(plog).ravel(), target_transform)[0]
            prices.append(p)
        except Exception:
            prices.append(None)

    st.line_chart(pd.DataFrame({"Price": prices}, index=sizes))
