# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Any, Dict, List

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
    # model
    model_path = None
    for fn in MODEL_FILENAMES:
        p = artifact_dir / fn
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {artifact_dir}. Expected one of {MODEL_FILENAMES}")

    model_obj = joblib.load(model_path)

    # preprocessing dict
    prep_path = artifact_dir / PREP_FILENAME
    if not prep_path.exists():
        raise FileNotFoundError(f"No preprocessing file at {prep_path}. Please save preprocessing.joblib in {artifact_dir}")

    prep = joblib.load(prep_path)
    if not isinstance(prep, dict):
        raise ValueError("preprocessing.joblib must contain a dict (sanit_map, label_encoders, reduced_feature_names/feature_names, target_transform, feature_defaults)")

    return model_obj, prep

def get_model_feature_names(model_obj: Any, prep: Dict[str, Any]) -> List[str]:
    """
    Prefer model.feature_names_ if available (sklearn/CatBoost). For blends, inspect base model(s).
    Fallback to prep['reduced_feature_names'] or prep['sanit_map'].
    """
    # single estimator
    try:
        if hasattr(model_obj, "feature_names_"):
            return list(map(str, model_obj.feature_names_))
        # sklearn compatible: get_feature_names_out may exist
        if hasattr(model_obj, "get_feature_names_out"):
            return list(map(str, model_obj.get_feature_names_out()))
    except Exception:
        pass

    # blend dict -> inspect base models
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

    # fallback to prep saved lists
    if "reduced_feature_names" in prep and isinstance(prep["reduced_feature_names"], (list, tuple)):
        return list(map(str, prep["reduced_feature_names"]))
    if "feature_names" in prep and isinstance(prep["feature_names"], (list, tuple)):
        return list(map(str, prep["feature_names"]))
    if "sanit_map" in prep and isinstance(prep["sanit_map"], dict):
        return list(map(str, prep["sanit_map"].values()))

    raise RuntimeError("Cannot determine model feature names. Save feature names into artifacts (feature_names or use model.feature_names_).")

def safe_label_encode_scalar(val: Any, le) -> int:
    """Map single scalar value to encoded integer using LabelEncoder le.
       If value unseen: prefer 'NA_LE' class if present else fallback to first class.
    """
    sval = "" if val is None else str(val)
    classes = list(map(str, getattr(le, "classes_", [])))
    if sval in classes:
        return int(le.transform([sval])[0])
    if "NA_LE" in classes:
        return int(le.transform(["NA_LE"])[0])
    if len(classes) > 0:
        return int(le.transform([classes[0]])[0])
    # last resort (may raise)
    return int(le.transform([sval])[0])

def build_input_row(provided: Dict[str, Any], expected_cols: List[str], prep: Dict[str, Any]) -> pd.DataFrame:
    """Return a single-row DataFrame with columns in expected_cols aligned to model."""
    label_encoders: Dict[str, Any] = prep.get("label_encoders", {})
    defaults: Dict[str, Any] = prep.get("feature_defaults", {})

    # Start from defaults
    row: Dict[str, Any] = {}
    for c in expected_cols:
        if c in label_encoders:
            # categorical default: prefer feature_defaults, else 'NA_LE' if encoder has it, else first class, else 'NA_LE'
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
            # numeric default
            row[c] = defaults.get(c, 0)

    # override with provided fields (direct names)
    for k, v in provided.items():
        # Accept synonyms: if model expects 'usable_area' but user provided 'square_meters' etc.
        if k in expected_cols:
            row[k] = v
        else:
            # synonyms handling:
            if k == "usable_area":
                if "usable_area" in expected_cols:
                    row["usable_area"] = v
                if "square_meters" in expected_cols and ("square_meters" not in provided or provided.get("square_meters") is None):
                    row["square_meters"] = v
                if "total_area" in expected_cols and ("total_area" not in provided or provided.get("total_area") is None):
                    row["total_area"] = v
            if k == "square_meters":
                if "square_meters" in expected_cols:
                    row["square_meters"] = v
                if "usable_area" in expected_cols and ("usable_area" not in provided or provided.get("usable_area") is None):
                    row["usable_area"] = v
                if "total_area" in expected_cols and ("total_area" not in provided or provided.get("total_area") is None):
                    row["total_area"] = v
            if k == "total_area":
                if "total_area" in expected_cols:
                    row["total_area"] = v
                if "square_meters" in expected_cols and ("square_meters" not in provided or provided.get("square_meters") is None):
                    row["square_meters"] = v
                if "usable_area" in expected_cols and ("usable_area" not in provided or provided.get("usable_area") is None):
                    row["usable_area"] = v

    # Now encode categoricals
    for c, le in label_encoders.items():
        if c in expected_cols:
            row[c] = safe_label_encode_scalar(row[c], le)

    # Build DataFrame and coerce numeric columns
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
try:
    model_obj, prep = load_artifacts()
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

sanit_map = prep.get("sanit_map", {})
label_encoders = prep.get("label_encoders", {})
feature_defaults = prep.get("feature_defaults", {}) or {}
target_transform = prep.get("target_transform", "log")

try:
    model_feature_names = get_model_feature_names(model_obj, prep)
except Exception as e:
    st.error(f"Failed to determine model feature names: {e}")
    st.stop()

# -----------------------
# District coordinates (extend as needed)
# -----------------------
DISTRICT_COORDS = {
    1: (50.08804, 14.42076),
    2: (50.07124, 14.43774),
    3: (50.07570, 14.45103),
    4: (50.06325, 14.45759),
    5: (50.06688, 14.39545),
    6: (50.09310, 14.45801),
    7: (50.10275, 14.41444),
    8: (50.08913, 14.51845),
    9: (50.08459, 14.53504),
    10: (50.04513, 14.45036),
    # add more if you want
}

def district_to_coords(district_label: str):
    """Try to extract Praha / Prague district number and return lat/lon if known."""
    if not isinstance(district_label, str):
        return None
    s = district_label.strip()
    # look for forms like "Praha 3", "Prague 3", "Praha 3 - X", "Praha 3 (something)"
    import re
    m = re.search(r"(Praha|Prague)\s*([0-9]{1,2})", s, flags=re.IGNORECASE)
    if m:
        num = int(m.group(2))
        if num in DISTRICT_COORDS:
            return DISTRICT_COORDS[num]
    # also try "Praha 1" words without capital etc.
    m2 = re.search(r"^(\d{1,2})$", s)
    if m2:
        num = int(m2.group(1))
        return DISTRICT_COORDS.get(num)
    return None

# -----------------------
# UI
# -----------------------
st.title("🏠 Flat Price Predictor — Prague (demo)")

left, mid, right = st.columns(3)

# Provide three area inputs and a "sync" toggle so users can set one and copy it to others
with left:
    usable_area = st.number_input("Usable area (m²)", min_value=0, max_value=10000,
                                 value=int(max(1, int(round(feature_defaults.get("usable_area", 50))))))

with mid:
    square_meters = st.number_input("Square meters (m²)", min_value=0, max_value=10000,
                                   value=int(max(1, int(round(feature_defaults.get("square_meters", usable_area))))))

with right:
    total_area = st.number_input("Total area (m²)", min_value=0, max_value=10000,
                                value=int(max(1, int(round(feature_defaults.get("total_area", square_meters))))))

sync_areas = st.checkbox("When I change Usable area, copy it to Square meters and Total area", value=True)

if sync_areas:
    # if user changed usable_area, update others visually (Streamlit will redraw)
    square_meters = usable_area
    total_area = usable_area

# few categorical inputs (use encoders if present)
col_a, col_b, col_c = st.columns(3)
with col_a:
    if "district" in label_encoders:
        district_opts = list(map(str, label_encoders["district"].classes_))
        district = st.selectbox("District", district_opts, index=0)
    else:
        district = st.text_input("District", value=str(feature_defaults.get("district", "Praha 1")))

with col_b:
    if "ownership" in label_encoders:
        ownership = st.selectbox("Ownership", list(map(str, label_encoders["ownership"].classes_)), index=0)
    else:
        ownership = st.text_input("Ownership", value=str(feature_defaults.get("ownership", "Personal")))

with col_c:
    if "layout" in label_encoders:
        layout = st.selectbox("Layout", list(map(str, label_encoders["layout"].classes_)), index=0)
    else:
        layout = st.text_input("Layout", value=str(feature_defaults.get("layout", "1+1")))

with st.expander("Advanced options (flags)"):
    terrace_flag = st.checkbox("Has terrace?", value=bool(feature_defaults.get("terrace", 0)))
    garage_flag = st.checkbox("Has garage?", value=bool(feature_defaults.get("garage", 0)))
    cellar_flag = st.checkbox("Has cellar?", value=bool(feature_defaults.get("cellar", 0)))

# -----------------------
# Prediction
# -----------------------
provided = {
    "usable_area": usable_area,
    "square_meters": square_meters,
    "total_area": total_area,
    "district": district,
    "ownership": ownership,
    "layout": layout,
    "terrace": int(terrace_flag),
    "garage": int(garage_flag),
    "cellar": int(cellar_flag),
}

if st.button("Predict price"):
    try:
        # Build input aligned to model_feature_names
        df_in = build_input_row(provided, model_feature_names, prep)
        # Reindex to exact order expected (CatBoost is strict about order & names)
        df_in = df_in.reindex(columns=model_feature_names, fill_value=0)

        # small sanity check
        if str(layout).startswith("6") and df_in.get("usable_area", df_in.get("square_meters", pd.Series([0]))).iloc[0] < 60:
            st.warning("⚠️ Unusual combo: very small area with 6+ rooms")

        # make predictions (support blend dict or estimator)
        if not isinstance(model_obj, dict):
            preds_log = model_obj.predict(df_in)
        else:
            # blend logic: base_models + weights (respect order if present)
            base_dict = model_obj.get("base_models") or model_obj.get("models") or {}
            order = model_obj.get("order", list(base_dict.keys()))
            weights = model_obj.get("weights", None)
            # interpret weights
            if isinstance(weights, dict):
                w = np.array([float(weights.get(n, 0.0)) for n in order], dtype=float)
            else:
                w = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(order), dtype=float)
                if w.shape[0] != len(order):
                    w = np.ones(len(order), dtype=float)
            w[w < 0] = 0.0
            if w.sum() <= 0:
                w = np.ones_like(w)
            w = w / w.sum()

            preds_components = []
            for i, name in enumerate(order):
                if name not in base_dict:
                    raise KeyError(f"Blend expects base model named '{name}', but it's missing.")
                mdl = base_dict[name]
                # determine columns this base expects
                if hasattr(mdl, "feature_names_"):
                    cols_for_base = list(map(str, mdl.feature_names_))
                else:
                    cols_for_base = model_feature_names
                Xb = df_in.reindex(columns=cols_for_base, fill_value=0)
                p = np.asarray(mdl.predict(Xb)).ravel()
                preds_components.append(p * w[i])
            preds_log = np.sum(np.vstack(preds_components), axis=0)

        preds = inv_target_transform(np.asarray(preds_log).ravel(), target_transform)
        price = float(np.asarray(preds).ravel()[0])
        st.success(f"💰 Estimated price: {price:,.0f} CZK")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

    # show map if we can find coords for district
    coords = district_to_coords(str(district))
    if coords is not None:
        st.map(pd.DataFrame([{"lat": coords[0], "lon": coords[1]}]))
    else:
        st.info("Map coordinates not available for this district. You can extend DISTRICT_COORDS for more districts.")

    # Sensitivity plot: price vs size
    try:
        st.subheader("📊 Sensitivity: Price vs. Usable area (others fixed)")
        sizes = np.linspace(20, 200, 20)
        prices = []
        for s in sizes:
            prov = provided.copy()
            prov["usable_area"] = float(s)
            prov["square_meters"] = float(s)
            prov["total_area"] = float(s)
            df_tmp = build_input_row(prov, model_feature_names, prep)
            df_tmp = df_tmp.reindex(columns=model_feature_names, fill_value=0)
            if not isinstance(model_obj, dict):
                plog = model_obj.predict(df_tmp)
            else:
                # simple average across base models for the curve (blend weighting could be applied similarly)
                base_dict = model_obj.get("base_models") or model_obj.get("models") or {}
                tmp_preds = []
                for m in base_dict.values():
                    cols_for_m = list(map(str, getattr(m, "feature_names_", model_feature_names)))
                    tmp_preds.append(np.asarray(m.predict(df_tmp.reindex(columns=cols_for_m, fill_value=0))).ravel())
                plog = np.mean(np.vstack(tmp_preds), axis=0)
            p = inv_target_transform(np.asarray(plog).ravel(), target_transform)
            prices.append(float(np.asarray(p).ravel()[0]))
        st.line_chart(pd.DataFrame({"Price": prices}, index=sizes))
    except Exception as e:
        st.info("Could not build sensitivity plot.")
        st.exception(e)

# -----------------------
# Preprocessing debug
# -----------------------
with st.expander("Preprocessing summary (debug)"):
    st.write("Feature names expected by model (first 30):", model_feature_names[:30])
    st.write("Encoders:", list(label_encoders.keys()))
    st.write("Sample feature defaults (first 20):", {k: feature_defaults.get(k) for k in list(feature_defaults)[:20]})
    st.write("Sanitization map (sample):", {k: sanit_map.get(k) for k in list(sanit_map)[:15]})
