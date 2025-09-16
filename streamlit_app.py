# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Any, Dict, List

# -----------------------
# CONFIG
# -----------------------
ARTIFACT_DIR = Path("deployable")
MODEL_FILENAMES = ["model.joblib", "model.pkl", "model.pkl.joblib", "model.joblib"]
PREP_FILENAME = "preprocessing.joblib"

st.set_page_config(layout="wide", page_title="🏠 Flat Price Predictor")

# -----------------------
# Helpers: load artifacts
# -----------------------
@st.cache_resource
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
        raise FileNotFoundError(f"No preprocessing file found at {prep_path}. Save preprocessing.joblib in {artifact_dir}.")

    prep = joblib.load(prep_path)
    if not isinstance(prep, dict):
        raise ValueError("preprocessing.joblib must contain a dict with keys: sanit_map, label_encoders, reduced_feature_names (or feature_names), target_transform, feature_defaults")

    return model_obj, prep

# safe label-encoding: unseen → NA_LE (if present) else first class
def safe_label_encode_scalar(val: Any, le) -> Any:
    sval = "" if val is None else str(val)
    classes = list(map(str, getattr(le, "classes_", [])))
    if sval in classes:
        return le.transform([sval])[0]
    if "NA_LE" in classes:
        return le.transform(["NA_LE"])[0]
    if len(classes) > 0:
        return le.transform([classes[0]])[0]
    # last resort: try to transform (may fail)
    return le.transform([sval])[0]

# pick model feature names (prefer model's metadata)
def get_model_feature_names(model_obj, prep: Dict) -> List[str]:
    # prefer estimator.feature_names_ (sklearn) or model.feature_names_ (catboost)
    try:
        if hasattr(model_obj, "feature_names_"):
            return list(map(str, model_obj.feature_names_))
    except Exception:
        pass

    # if blend / dict: inspect first base model
    if isinstance(model_obj, dict):
        for key in ("base_models", "models", "base_fitted"):
            if key in model_obj and isinstance(model_obj[key], dict):
                for m in model_obj[key].values():
                    if hasattr(m, "feature_names_"):
                        return list(map(str, m.feature_names_))
    # fallback to prep keys
    if "feature_names" in prep:
        return list(map(str, prep["feature_names"]))
    if "reduced_feature_names" in prep:
        return list(map(str, prep["reduced_feature_names"]))
    if "sanit_map" in prep and isinstance(prep["sanit_map"], dict):
        return list(map(str, prep["sanit_map"].values()))

    raise RuntimeError("Could not determine expected feature names for the model. Ensure artifacts include feature names.")

# build one-row dataframe aligned to expected_cols
def build_input_row(provided: Dict[str, Any], expected_cols: List[str], prep: Dict):
    label_encoders: Dict[str, Any] = prep.get("label_encoders", {})
    defaults: Dict[str, Any] = prep.get("feature_defaults", {})

    # start with defaults: categorical -> 'NA_LE' or encoder-first, numeric -> 0 or saved median
    row = {}
    for c in expected_cols:
        if c in label_encoders:
            # choose default: first try dataset default, else NA_LE if encoder has it, else first class if any, else "NA_LE"
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

    # override with provided (provided keys may be 'usable_area' or original names)
    # We assume provided keys are sanitized or common names used in UI (we attempt both)
    for k, v in provided.items():
        if k in expected_cols:
            row[k] = v
        else:
            # try to map common synonyms: usable_area <-> square_meters
            if k == "usable_area" and "square_meters" in expected_cols and "usable_area" not in expected_cols:
                row["square_meters"] = v
            if k == "square_meters" and "usable_area" in expected_cols and "square_meters" not in expected_cols:
                row["usable_area"] = v
            # otherwise ignore if unknown

    # now encode categoricals (turn into numeric labels)
    for c, le in label_encoders.items():
        if c in expected_cols:
            row[c] = safe_label_encode_scalar(row[c], le)

    # make DataFrame and coerce numerics
    df = pd.DataFrame([row], columns=expected_cols)
    for c in df.columns:
        if c not in label_encoders:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def inv_target_transform(y_arr, tgt: str):
    if tgt == "log1p":
        return np.expm1(y_arr)
    if tgt == "log":
        return np.exp(y_arr)
    return y_arr

# -----------------------
# Load artifacts (stop if missing)
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

# get model_feature_names (list)
try:
    model_feature_names = get_model_feature_names(model_obj, prep)
except Exception as e:
    st.error(f"Failed to determine model feature names: {e}")
    st.stop()

# -----------------------
# UI - keep minimal and similar to your working version
# -----------------------
st.title("🏠 Flat Price Predictor — Prague (demo)")

col1, col2, col3 = st.columns(3)

# helper for safe default for widget values (ensures >= min_value)
def safe_default_for_widget(key: str, fallback: int, min_value: int = 1):
    raw = feature_defaults.get(key, fallback)
    try:
        ival = int(np.round(float(raw)))
    except Exception:
        ival = fallback
    return max(ival, min_value)

with col1:
    # prefer 'usable_area' input (many CatBoost models used 'usable_area')
    usable_area = st.number_input("Usable area (m²)", min_value=1, max_value=5000,
                                 value=safe_default_for_widget("usable_area", 50, 1))
with col2:
    # use encoder labels where available, else free text fallback
    if "district" in label_encoders:
        district_opts = list(map(str, label_encoders["district"].classes_))
        district = st.selectbox("District", district_opts, index=0)
    else:
        district = st.text_input("District", value=str(feature_defaults.get("district", "NA_LE")))
with col3:
    if "ownership" in label_encoders:
        ownership_opts = list(map(str, label_encoders["ownership"].classes_))
        ownership = st.selectbox("Ownership", ownership_opts, index=0)
    else:
        ownership = st.text_input("Ownership", value=str(feature_defaults.get("ownership", "NA_LE")))

if "layout" in label_encoders:
    layout_opts = list(map(str, label_encoders["layout"].classes_))
    layout = st.selectbox("Layout", layout_opts, index=0)
else:
    layout = st.text_input("Layout", value=str(feature_defaults.get("layout", "NA_LE")))

with st.expander("Advanced options"):
    # keep boolean flags (0/1) – simpler and commonly useful
    terrace = st.checkbox("Has terrace?", value=bool(feature_defaults.get("terrace", 0)))
    garage = st.checkbox("Has garage?", value=bool(feature_defaults.get("garage", 0)))
    cellar = st.checkbox("Has cellar?", value=bool(feature_defaults.get("cellar", 0)))

# -----------------------
# Prediction logic
# -----------------------
provided = {
    # provide both usable and square_meters synonyms if model expects either
    "usable_area": usable_area,
    "square_meters": usable_area,  # copy so both names are available for alignment
    "district": district,
    "ownership": ownership,
    "layout": layout,
    "terrace": int(terrace),
    "garage": int(garage),
    "cellar": int(cellar),
}

if st.button("Predict price"):
    try:
        # Build input aligned to model features
        df_in = build_input_row(provided, model_feature_names, prep)

        # quick sanity warning
        if str(layout).startswith("6") and (df_in.get("usable_area", df_in.get("square_meters", pd.Series([0]))).iloc[0] < 60):
            st.warning("⚠️ Unusual combo: very small area with 6+ rooms")

        # predict: support single estimator and blend dict
        if not isinstance(model_obj, dict):
            preds_log = model_obj.predict(df_in)
        else:
            # model_obj is a dict: try "base_models" or "models"; weights may be non-deterministic order,
            # so respect model_obj["order"] if present
            base_dict = model_obj.get("base_models") or model_obj.get("models") or {}
            if "order" in model_obj and isinstance(model_obj["order"], (list, tuple)):
                order = model_obj["order"]
            else:
                order = list(base_dict.keys())
            weights = model_obj.get("weights")
            # compute weighted average (weights may be dict or array)
            if isinstance(weights, dict):
                w = np.array([float(weights.get(n, 0.0)) for n in order], dtype=float)
            else:
                w = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(order), dtype=float)
                if w.shape[0] != len(order):
                    w = np.ones(len(order), dtype=float)
            # normalize non-negative
            w[w < 0] = 0.0
            if w.sum() <= 0:
                w = np.ones_like(w)
            w = w / w.sum()
            preds_components = []
            for i, name in enumerate(order):
                if name not in base_dict:
                    raise KeyError(f"Blend expects base model named '{name}', but it's missing in artifact.")
                mdl = base_dict[name]
                # choose the columns the base model expects if available, else use model_feature_names
                cols_for_m = None
                if hasattr(mdl, "feature_names_"):
                    cols_for_m = list(map(str, mdl.feature_names_))
                else:
                    cols_for_m = model_feature_names
                X_m = df_in.reindex(columns=cols_for_m, fill_value=0)
                p = np.asarray(mdl.predict(X_m)).ravel()
                preds_components.append(p * w[i])
            preds_log = np.sum(np.vstack(preds_components), axis=0)

        # inverse target transform
        preds = inv_target_transform(np.asarray(preds_log).ravel(), target_transform)
        price = float(np.asarray(preds).ravel()[0])
        st.success(f"💰 Estimated price: {price:,.0f} CZK")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

    # small map for a few known districts (optional)
    coords = {
        "Praha 1": (50.088, 14.420),
        "Prague 1": (50.088, 14.420),
        "Praha 2": (50.071, 14.437),
        "Prague 2": (50.071, 14.437),
        "Praha 5": (50.067, 14.395),
        "Prague 5": (50.067, 14.395),
    }
    if str(district) in coords:
        lat, lon = coords[str(district)]
        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]))

    # Sensitivity / simple size vs price curve
    try:
        st.subheader("📊 Sensitivity: Price vs. Size (fixed other inputs)")
        sizes = np.linspace(20, 200, 20)
        prices = []
        for s in sizes:
            prov = provided.copy()
            prov["usable_area"] = float(s)
            prov["square_meters"] = float(s)
            df_tmp = build_input_row(prov, model_feature_names, prep)
            if not isinstance(model_obj, dict):
                preds_tmp_log = model_obj.predict(df_tmp)
            else:
                # simple average across base models if blend
                base_dict = model_obj.get("base_models") or model_obj.get("models") or {}
                tmp_preds = []
                for m in base_dict.values():
                    cols_for_m = list(map(str, getattr(m, "feature_names_", model_feature_names)))
                    ptmp = m.predict(df_tmp.reindex(columns=cols_for_m, fill_value=0))
                    tmp_preds.append(np.asarray(ptmp).ravel())
                preds_tmp_log = np.mean(np.vstack(tmp_preds), axis=0)
            preds_tmp = inv_target_transform(np.asarray(preds_tmp_log).ravel(), target_transform)
            prices.append(float(np.asarray(preds_tmp).ravel()[0]))
        st.line_chart(pd.DataFrame({"Price": prices}, index=sizes))
    except Exception as e:
        st.info("Could not build sensitivity plot.")
        st.exception(e)

# -----------------------
# Preprocessing summary
# -----------------------
with st.expander("Preprocessing summary (debug)"):
    st.write("Encoders keys:", list(label_encoders.keys()))
    st.write("Expected features (first 15):", model_feature_names[:15])
    st.write("Feature defaults sample:", {k: feature_defaults.get(k) for k in list(feature_defaults)[:10]})
