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
ARTIFACT_DIR = Path("deployable")  # change to your drive path if you saved artifacts to Drive
MODEL_FILENAMES = ["model.joblib", "model.pkl", "model.pkl.joblib", "model.pkl"]  # tried names
PREP_FILENAME = "preprocessing.joblib"

TARGET_TRANSFORM_KEY = "target_transform"  # inside preprocessing.joblib
# -----------------------

st.set_page_config(layout="wide", page_title="Flat Price Predictor")

# -----------------------
# Helpers: loading artifacts
# -----------------------
@st.cache_resource
def load_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> Tuple[Any, Dict[str, Any]]:
    """Load model artifact and preprocessing dict. Returns (model_obj, prep_dict)."""

    # 1) model
    model_path = None
    for fn in MODEL_FILENAMES:
        p = artifact_dir / fn
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {artifact_dir}. Expected one of {MODEL_FILENAMES}")

    model_obj = joblib.load(model_path)

    # 2) preprocessing dict
    prep_path = artifact_dir / PREP_FILENAME
    if not prep_path.exists():
        # maybe user saved preprocessing under different name
        raise FileNotFoundError(f"No preprocessing file found at {prep_path}. Save preprocessing.joblib with keys: sanit_map, label_encoders, reduced_feature_names, target_transform")

    prep = joblib.load(prep_path)
    if not isinstance(prep, dict):
        raise ValueError(f"preprocessing.joblib should contain a dict (was {type(prep)}). Expected keys: sanit_map, label_encoders, reduced_feature_names, target_transform")

    return model_obj, prep


def find_feature_names_from_model(model_obj, prep: dict) -> list:
    """
    Determine the feature names expected by the model or the saved base models (for blends).
    Strategy:
      1) if model is a single estimator and has feature_names_in_, use it
      2) if model is a dict (blend) and base_models exist, inspect base models for feature_names_in_
      3) fallback: try 'permutation_importance.csv' in artifact_dir (if present) to recover columns
      4) fallback: if prep contains 'reduced_feature_names' or 'sanit_map' use those (best-effort)
      5) otherwise error and instruct to re-save artifacts including feature names
    """
    # 1) single estimator
    if not isinstance(model_obj, dict):
        if hasattr(model_obj, "feature_names_in_"):
            return list(map(str, model_obj.feature_names_in_))
        # xgboost/catboost may not have feature_names_in_; try to read from model if possible
        try:
            # many sklearn models store _feature_names_in or get_feature_names_out
            if hasattr(model_obj, "get_feature_names_out"):
                return list(map(str, model_obj.get_feature_names_out()))
        except Exception:
            pass

    # 2) blend / dict
    if isinstance(model_obj, dict):
        # possible keys: "base_models", "models", "order"
        base_dict = None
        for key in ("base_models", "models", "base_fitted"):
            if key in model_obj and isinstance(model_obj[key], dict):
                base_dict = model_obj[key]
                break
        if base_dict:
            for m in base_dict.values():
                if hasattr(m, "feature_names_in_"):
                    return list(map(str, m.feature_names_in_))
                try:
                    if hasattr(m, "get_feature_names_out"):
                        return list(map(str, m.get_feature_names_out()))
                except Exception:
                    pass

    # 3) try permutation_importance.csv in artifact_dir
    pi_path = ARTIFACT_DIR / "permutation_importance.csv"
    if pi_path.exists():
        try:
            dfpi = pd.read_csv(pi_path)
            if "feature" in dfpi.columns:
                return dfpi["feature"].astype(str).tolist()
        except Exception:
            pass

    # 4) fallback to preprocessing saved lists
    if "reduced_feature_names" in prep and isinstance(prep["reduced_feature_names"], (list, tuple)):
        # NOTE: this is likely the reduced set used by linear models only, but it's better than nothing
        return list(map(str, prep["reduced_feature_names"]))

    if "sanit_map" in prep and isinstance(prep["sanit_map"], dict):
        # take sanitized names (values)
        return list(map(str, prep["sanit_map"].values()))

    # give up with a helpful message
    raise RuntimeError(
        "Could not determine model feature names. Re-save your artifacts so the model or preprocessing contains "
        "'feature_names' or ensure preprocessing.joblib contains 'reduced_feature_names' or 'sanit_map'."
    )


def safe_label_encode_series(series: pd.Series, le) -> np.ndarray:
    """Transform series with LabelEncoder le; map unseen → 'NA_LE' when available."""
    s = series.astype("string").fillna("NA_LE").astype(str)
    classes = set(map(str, getattr(le, "classes_", [])))
    if "NA_LE" in classes:
        # map unseen to NA_LE
        s = s.apply(lambda x: x if x in classes else "NA_LE")
    else:
        # if NA_LE not present, map unseen to first class (best-effort)
        fallback = next(iter(classes)) if len(classes) > 0 else None
        s = s.apply(lambda x: x if x in classes else (fallback if fallback is not None else x))
    return le.transform(s)


# -----------------------
# Load artifacts
# -----------------------
try:
    model_obj, prep = load_artifacts()
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

# Extract commonly used items from prep dict
sanit_map = prep.get("sanit_map", {})              # original_name -> sanitized_name
label_encoders: Dict[str, Any] = prep.get("label_encoders", {})  # col -> LabelEncoder
reduced_feature_names = prep.get("reduced_feature_names", [])
TARGET_TRANSFORM = prep.get("target_transform", "log")  # 'log' or 'log1p' or 'none'

# Find expected model features (list)
try:
    model_feature_names = find_feature_names_from_model(model_obj, prep)
except Exception as e:
    st.error(f"Could not determine expected model features: {e}")
    st.stop()

# -----------------------
# UI: minimal 4 inputs
# -----------------------
st.title("🏠 Flat Price Predictor — Prague (minimal inputs)")

st.markdown(
    """
    **Inputs used by this simplified UI** — we ask for 4 fields only.
    The app will fill the other features with defaults (0 / NA).  
    This is convenient for quick demo, but predictions may be inaccurate if many features are missing.
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    square_meters = st.number_input("Size (m²)", min_value=5, max_value=2000, value=50, step=1)

with col2:
    if "district" in label_encoders:
        district_options = list(label_encoders["district"].classes_)
        district = st.selectbox("District", district_options, index=0)
    else:
        st.warning("Encoder for 'district' not found in artifacts — please use text input.")
        district = st.text_input("District (free text)", value="NA_LE")

with col3:
    if "ownership" in label_encoders:
        ownership_options = list(label_encoders["ownership"].classes_)
        ownership = st.selectbox("Ownership", ownership_options, index=0)
    else:
        st.warning("Encoder for 'ownership' not found in artifacts — using free text.")
        ownership = st.text_input("Ownership", value="NA_LE")

layout = None
if "layout" in label_encoders:
    layout_options = list(label_encoders["layout"].classes_)
    layout = st.selectbox("Layout", layout_options, index=0)
else:
    layout = st.text_input("Layout (e.g., 1+1)", value="NA_LE")

# Debug / advanced
debug = st.checkbox("Show debug info (inputs & prepared DF)")

# -----------------------
# Prediction code
# -----------------------
def build_input_dataframe(provided: dict, expected_cols: list, label_encoders: dict) -> pd.DataFrame:
    """
    Build a full dataframe ready for model input:
      - expected_cols: list of column names the model expects (sanitized names)
      - provided: dict mapping sanitized column -> value (we'll do best-effort mapping)
      - label_encoders: dict of LabelEncoders keyed by sanitized name
    Missing numeric -> 0, missing categorical -> 'NA_LE' or first class.
    """
    # construct empty DF first (one row)
    df = pd.DataFrame(index=[0], columns=expected_cols, dtype=object)

    # Fill defaults for numeric vs categorical:
    # If we have encoder for a column -> treat as categorical else numeric
    for c in expected_cols:
        if c in label_encoders:
            # choose default: 'NA_LE' if encoder has it, else the first encoder class, else string 'NA_LE'
            classes = list(map(str, getattr(label_encoders[c], "classes_", [])))
            if "NA_LE" in classes:
                df.at[0, c] = "NA_LE"
            elif len(classes) > 0:
                df.at[0, c] = classes[0]
            else:
                df.at[0, c] = "NA_LE"
        else:
            # numeric default
            df.at[0, c] = 0

    # Now set provided values (they come from our UI; keys may be sanitized already)
    for k, v in provided.items():
        if k in df.columns:
            df.at[0, k] = v
        else:
            # try to map from original name -> sanitized via sanit_map (prep)
            # sanit_map is original->sanitized; reverse map if necessary
            rev_map = {v2: k2 for k2, v2 in sanit_map.items()} if sanit_map else {}
            if k in rev_map:
                sanitized = rev_map[k]
                if sanitized in df.columns:
                    df.at[0, sanitized] = v

    # Apply label encoders to categorical cols
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = safe_label_encode_series(df[col], le)
            except Exception as e:
                # fallback: try to coerce unseen to NA_LE or 0
                classes = list(map(str, getattr(le, "classes_", [])))
                if "NA_LE" in classes:
                    df[col] = df[col].astype(str).apply(lambda x: x if x in classes else "NA_LE")
                    df[col] = le.transform(df[col])
                else:
                    try:
                        # transform with fallback mapping
                        df[col] = df[col].astype(str).apply(lambda x: x if x in classes else (classes[0] if classes else x))
                        df[col] = le.transform(df[col])
                    except Exception:
                        # last resort: numeric 0
                        df[col] = 0

    # Convert numeric columns to float where possible
    for c in df.columns:
        if c not in label_encoders:
            # try convert to numeric
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            except Exception:
                pass

    # reorder columns same as expected_cols
    df = df[expected_cols]
    return df


def predict_from_artifacts(df_input: pd.DataFrame, model_obj, model_feature_names: list, prep: dict) -> np.ndarray:
    """
    Core prediction: handles single model or dict/blend.
    Ensures each base model receives the features it expects.
    """
    # single estimator
    if not isinstance(model_obj, dict):
        estimator = model_obj
        # choose columns to feed: use estimator.feature_names_in_ if available else model_feature_names
        if hasattr(estimator, "feature_names_in_"):
            cols_for_model = list(map(str, estimator.feature_names_in_))
        else:
            cols_for_model = model_feature_names
        X_for_model = df_input[cols_for_model]
        preds = estimator.predict(X_for_model)
        return np.asarray(preds)

    # model_obj is dict -> possible shapes:
    #  - {"weights": ..., "base_models": {...}, "order": [...]}
    #  - {"weights": np.array([...]), "models": {...}, "order": [...]}
    # try to converge to a common representation
    base_models = None
    order = None
    weights = None

    if "base_models" in model_obj and isinstance(model_obj["base_models"], dict):
        base_models = model_obj["base_models"]
    elif "models" in model_obj and isinstance(model_obj["models"], dict):
        base_models = model_obj["models"]
    else:
        # maybe the dict is a direct sklearn estimator (rare) - bail out
        raise ValueError("Blend dict found but 'base_models' / 'models' dict is missing.")

    if "order" in model_obj:
        order = model_obj["order"]
    else:
        order = list(base_models.keys())

    # weights may be saved in various formats (dict, list, np.array)
    if "weights" in model_obj:
        w = model_obj["weights"]
        if isinstance(w, dict):
            weights = [float(w[name]) for name in order]
        elif isinstance(w, (list, tuple, np.ndarray)):
            # assume order aligned: if length equals len(order) use directly
            arr = np.asarray(w, dtype=float)
            if arr.shape[0] == len(order):
                weights = list(arr.astype(float))
            else:
                # unknown alignment: default equal weights
                weights = [1.0] * len(order)
        else:
            weights = [1.0] * len(order)
    else:
        weights = [1.0] * len(order)

    # normalize weights (non-negative)
    w = np.array(weights, dtype=float)
    w[w < 0] = 0.0
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()

    # get predictions from each base model (use proper columns for each)
    preds_list = []
    for i, name in enumerate(order):
        if name not in base_models:
            raise KeyError(f"Base model {name} not found in blend artifact.")
        m = base_models[name]
        # find model-specific feature names
        if hasattr(m, "feature_names_in_"):
            cols_m = list(map(str, m.feature_names_in_))
        else:
            cols_m = model_feature_names  # fallback

        X_m = df_input[cols_m]
        p = np.asarray(m.predict(X_m))
        preds_list.append(p * w[i])

    # weighted average
    preds_stack = np.vstack(preds_list)  # shape (n_models, n_samples)
    yhat = preds_stack.sum(axis=0)
    return yhat


def inv_target_transform(y_arr: np.ndarray, tgt_transform: str = TARGET_TRANSFORM) -> np.ndarray:
    if tgt_transform == "log1p":
        return np.expm1(y_arr)
    elif tgt_transform == "log":
        return np.exp(y_arr)
    else:
        return y_arr


# -----------------------
# When user clicks Predict
# -----------------------
provided = {
    # we must use sanitized column names that models expect — hopefully your prep used sanitized names
    # the labels in label_encoders are expected to be sanitized column names as well
    "square_meters": square_meters,
    "district": district,
    "ownership": ownership,
    "layout": layout,
}

if st.button("Predict price"):
    # Build full df for model
    try:
        df_full = build_input_dataframe(provided, model_feature_names, label_encoders)
    except Exception as e:
        st.error(f"Failed to build input dataframe: {e}")
        st.stop()

    # show debug if toggled
    if debug:
        st.subheader("Prepared input (sent to model):")
        st.write(df_full.head(1))

    # Predict
    try:
        yhat_log = predict_from_artifacts(df_full, model_obj, model_feature_names, prep)
        yhat = inv_target_transform(yhat_log)
        if isinstance(yhat, np.ndarray) or isinstance(yhat, list):
            price = float(np.asarray(yhat).ravel()[0])
        else:
            price = float(yhat)
        st.success(f"💰 Estimated price: {price:,.0f} CZK")

        if debug:
            st.write(f"Predicted (log): {yhat_log.ravel()[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        if debug:
            st.exception(e)

# small footer
st.markdown("---")
st.markdown("Notes: this simplified UI uses only 4 inputs and fills other model features with defaults. "
            "For production-grade results, provide full feature set and use the same preprocessing pipeline used during training.")
