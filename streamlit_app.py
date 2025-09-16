# streamlit_app.py
"""
Streamlit app for flat price prediction.
Requirements: streamlit, pandas, numpy, joblib
Place model + preprocessing artifacts in ./deployable/
 - model.joblib (or model.pkl, model.pkl.joblib)
 - preprocessing.joblib (dict with keys at least: sanit_map, label_encoders, feature_defaults OR reduced_feature_names)
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, List

# -----------------------
# CONFIG
# -----------------------
ARTIFACT_DIR = Path("deployable")
MODEL_FILENAMES = ["model.joblib", "model.pkl", "model.pkl.joblib", "model.joblib"]
PREP_FILENAME = "preprocessing.joblib"

st.set_page_config(layout="wide", page_title="🏠 Flat Price Predictor")

# -----------------------
# Helpers: load artifacts + detect expected features
# -----------------------
@st.cache_resource
def load_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> Tuple[Any, dict]:
    # 1) load model
    model_path = None
    for fn in MODEL_FILENAMES:
        p = artifact_dir / fn
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {artifact_dir}. Expected one of: {MODEL_FILENAMES}")

    model_obj = joblib.load(model_path)

    # 2) load preprocessing dict
    prep_path = artifact_dir / PREP_FILENAME
    if not prep_path.exists():
        raise FileNotFoundError(f"No preprocessing file found at {prep_path}. Save preprocessing.joblib with keys: sanit_map, label_encoders, feature_defaults or reduced_feature_names")

    prep = joblib.load(prep_path)
    if not isinstance(prep, dict):
        raise ValueError(f"{prep_path} should contain a dict (was {type(prep)})")

    return model_obj, prep


def find_model_feature_names(model_obj: Any, prep: dict, artifact_dir: Path = ARTIFACT_DIR) -> List[str]:
    """
    Determine the exact feature names and order expected by the model.
    Strategy:
      1) if estimator has 'feature_names_in_' return that
      2) if dict (blend) find a base model with feature_names_in_
      3) try permutation_importance.csv (column 'feature')
      4) try prep['reduced_feature_names']
      5) try prep['feature_defaults'].keys() (or values if mapping available)
      6) try prep['sanit_map'] values
    """
    # 1) single estimator with feature_names_in_
    if not isinstance(model_obj, dict):
        if hasattr(model_obj, "feature_names_in_"):
            return [str(x) for x in list(model_obj.feature_names_in_)]
        # try get_feature_names_out
        try:
            if hasattr(model_obj, "get_feature_names_out"):
                return [str(x) for x in list(model_obj.get_feature_names_out())]
        except Exception:
            pass

    # 2) blend/dict: check base models
    if isinstance(model_obj, dict):
        for key in ("base_models", "models", "base_fitted"):
            if key in model_obj and isinstance(model_obj[key], dict):
                for m in model_obj[key].values():
                    if hasattr(m, "feature_names_in_"):
                        return [str(x) for x in list(m.feature_names_in_)]
                    try:
                        if hasattr(m, "get_feature_names_out"):
                            return [str(x) for x in list(m.get_feature_names_out())]
                    except Exception:
                        pass
                # none of base models exposed names; break to fallback
                break

    # 3) permutation_importance.csv
    pi_path = artifact_dir / "permutation_importance.csv"
    if pi_path.exists():
        try:
            dfpi = pd.read_csv(pi_path)
            if "feature" in dfpi.columns:
                return dfpi["feature"].astype(str).tolist()
        except Exception:
            pass

    # 4) prep['reduced_feature_names']
    if "reduced_feature_names" in prep and isinstance(prep["reduced_feature_names"], (list, tuple)):
        return [str(x) for x in prep["reduced_feature_names"]]

    # 5) prep['feature_defaults'] keys (often full set)
    if "feature_defaults" in prep and isinstance(prep["feature_defaults"], dict):
        return [str(x) for x in prep["feature_defaults"].keys()]

    # 6) sanit_map values (original->sanitized mapping)
    if "sanit_map" in prep and isinstance(prep["sanit_map"], dict):
        # sanit_map maps original->sanitized, we prefer the values (sanitized names)
        vals = list(prep["sanit_map"].values())
        if vals:
            return [str(x) for x in vals]

    raise RuntimeError("Could not determine model feature names. Re-save artifacts so model or preprocessing contains feature names.")


def safe_label_encode_series(value: Any, le) -> int:
    """
    Safely transform a single value using saved LabelEncoder le.
    If unseen and 'NA_LE' present → use it. Else fallback to first class.
    """
    s = str(value)
    classes = list(map(str, getattr(le, "classes_", [])))
    if s in classes:
        return int(le.transform([s])[0])
    if "NA_LE" in classes:
        return int(le.transform(["NA_LE"])[0])
    if len(classes) > 0:
        return int(le.transform([classes[0]])[0])
    # last resort, numerical 0
    return 0


# -----------------------
# Load artifacts
# -----------------------
try:
    model_obj, prep = load_artifacts()
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

# extract commonly used items
sanit_map: Dict[str, str] = prep.get("sanit_map", {})          # original_col -> sanitized_col
label_encoders: Dict[str, Any] = prep.get("label_encoders", {})  # sanitized_col -> LabelEncoder
feature_defaults: Dict[str, Any] = prep.get("feature_defaults", {})  # sanitized_col -> default value
target_transform: str = prep.get("target_transform", "log")

# Build reverse map original<-sanitized for mapping UI keys (we will allow either)
rev_sanit_map = {v: k for k, v in sanit_map.items()} if sanit_map else {}

# model expected features (sanitized names)
try:
    model_feature_names = find_model_feature_names(model_obj, prep)
except Exception as e:
    st.error(f"Failed to determine model feature names: {e}")
    st.stop()

# Make sure defaults exist for expected features; if missing create safe default
for f in model_feature_names:
    if f not in feature_defaults:
        # default: 'NA_LE' for categorical (if encoder exists) else median-like 0
        if f in label_encoders:
            feature_defaults[f] = "NA_LE"
        else:
            feature_defaults[f] = 0.0

# -----------------------
# UI (minimal + a few extras)
# -----------------------
st.title("🏠 Flat Price Predictor — Prague")

st.markdown("Small demo: choose a few key fields; the app fills the other model features with defaults saved during training.")

col1, col2, col3 = st.columns(3)
with col1:
    # UI keys: prefer original column names if available in sanit_map; else use sanitized names directly
    ui_usable = sanit_map.get("usable_area", "usable_area")
    usable_area = st.number_input("Usable area (m²)", min_value=1, max_value=5000,
                                  value=int(feature_defaults.get(ui_usable, feature_defaults.get("usable_area", 50))))
with col2:
    ui_total = sanit_map.get("total_area", "total_area")
    total_area = st.number_input("Total area (m²)", min_value=1, max_value=5000,
                                 value=int(feature_defaults.get(ui_total, feature_defaults.get("total_area", 50))))
with col3:
    ui_floor = sanit_map.get("floorage", "floorage")
    floorage = st.number_input("Floor number", min_value=0, max_value=200,
                               value=int(feature_defaults.get(ui_floor, feature_defaults.get("floorage", 1))))

col4, col5 = st.columns(2)
with col4:
    # district selectable only if encoder exists
    if "district" in label_encoders:
        district_options = list(label_encoders["district"].classes_)
        district = st.selectbox("District", district_options, index=0)
    else:
        district = st.text_input("District", value=str(feature_defaults.get(sanit_map.get("district","district"), "NA_LE")))
with col5:
    if "building" in label_encoders:
        building_options = list(label_encoders["building"].classes_)
        building = st.selectbox("Building type", building_options, index=0)
    else:
        building = st.text_input("Building type", value=str(feature_defaults.get(sanit_map.get("building","building"), "NA_LE")))

ownership = st.selectbox("Ownership",
                         list(label_encoders["ownership"].classes_) if "ownership" in label_encoders else [str(feature_defaults.get(sanit_map.get("ownership","ownership"), "NA_LE"))])
layout = st.selectbox("Layout",
                      list(label_encoders["layout"].classes_) if "layout" in label_encoders else [str(feature_defaults.get(sanit_map.get("layout","layout"), "NA_LE"))])
cadastral_area = st.selectbox("Cadastral area",
                              list(label_encoders["cadastral_area"].classes_) if "cadastral_area" in label_encoders else [str(feature_defaults.get(sanit_map.get("cadastral_area","cadastral_area"), "NA_LE"))])

with st.expander("Advanced options (flags)"):
    terrace_flag = st.checkbox("Terrace / has terrace", value=bool(int(feature_defaults.get(sanit_map.get("terrace","terrace"), feature_defaults.get("terrace", 0)))))
    garage_flag = st.checkbox("Garage / has garage", value=bool(int(feature_defaults.get(sanit_map.get("garage","garage"), feature_defaults.get("garage", 0)))))
    cellar_flag = st.checkbox("Cellar / has cellar", value=bool(int(feature_defaults.get(sanit_map.get("cellar","cellar"), feature_defaults.get("cellar", 0)))))

# -----------------------
# Build input dataframe that matches model_feature_names exactly (order + names)
# -----------------------
def build_model_input(provided_ui: dict, model_feature_names: List[str], sanit_map: Dict[str, str], label_encoders: Dict[str, Any], defaults: Dict[str, Any]) -> pd.DataFrame:
    """
    provided_ui: dict with UI keys in either original names or sanitized names.
    model_feature_names: list of sanitized names expected by model (and in the same order).
    sanit_map: original -> sanitized (values are sanitized)
    defaults: sanitized -> default value
    """
    # Create empty row with expected columns in the exact order
    df = pd.DataFrame(index=[0], columns=model_feature_names, dtype=object)

    # Fill with defaults
    for c in model_feature_names:
        df.at[0, c] = defaults.get(c, "NA_LE" if c in label_encoders else 0.0)

    # Helper to find sanitized name for a UI key (UI key might be original name or sanitized)
    def to_sanitized(key: str) -> str:
        # if key exactly matches sanitized → return
        if key in model_feature_names:
            return key
        # if key is original and in sanit_map -> return sanitized
        if key in sanit_map:
            return sanit_map[key]
        # also check reverse mapping in case user provided original but sanit_map maps other way
        if key in rev_sanit_map:
            return key  # already sanitized
        # lastly, try lowercased approximate match
        for m in model_feature_names:
            if m.lower() == key.lower():
                return m
        return key  # fallback (may cause model error if not in expected names)

    # Place provided UI values into sanitized columns
    for ui_k, v in provided_ui.items():
        sname = to_sanitized(ui_k)
        if sname in df.columns:
            df.at[0, sname] = v
        else:
            # ignore silently if UI key not relevant
            pass

    # Apply label encoders on categorical sanitized columns
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df.at[0, col] = safe_label_encode_series(df.at[0, col], le)
            except Exception:
                # fallback: use encoded default
                df.at[0, col] = safe_label_encode_series(defaults.get(col, "NA_LE"), le)

    # Convert numeric columns to numeric dtype (float)
    for c in df.columns:
        if c not in label_encoders:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            except Exception:
                df[c] = 0.0

    # reorder to model_feature_names explicitly and return
    df = df[model_feature_names]
    return df


# -----------------------
# Prediction action
# -----------------------
provided_ui = {
    # accept original names (if your sanit_map had original names like "usable_area") or sanitized
    "usable_area": usable_area,
    "total_area": total_area,
    "floorage": floorage,
    "district": district,
    "building": building,
    "ownership": ownership,
    "layout": layout,
    "cadastral_area": cadastral_area,
    "terrace": int(terrace_flag),
    "garage": int(garage_flag),
    "cellar": int(cellar_flag),
}

if st.button("Predict price"):
    try:
        X_input = build_model_input(provided_ui, model_feature_names, sanit_map, label_encoders, feature_defaults)
    except Exception as e:
        st.error(f"Failed to build model input dataframe: {e}")
        st.stop()

    st.write("Prepared input sent to model (first 20 cols):")
    st.write(X_input.iloc[:, :20].T)

    # Predict
    try:
        if hasattr(model_obj, "predict"):
            preds = model_obj.predict(X_input)
        else:
            # blend dict fallback: average base models (weights ignored here)
            base_models = model_obj.get("base_models", model_obj.get("models", {}))
            if not isinstance(base_models, dict) or len(base_models) == 0:
                raise RuntimeError("Blend artifact format unexpected (no base_models)")
            preds_list = []
            for m in base_models.values():
                # ensure each base model gets its required columns (if available use feature_names_in_)
                try:
                    if hasattr(m, "feature_names_in_"):
                        cols = list(map(str, m.feature_names_in_))
                    else:
                        cols = model_feature_names
                    preds_list.append(m.predict(X_input[cols]))
                except Exception:
                    # try full row
                    preds_list.append(m.predict(X_input))
            preds = np.mean(np.vstack(preds_list), axis=0)

        yhat = preds
        # if model outputs a 2d array (n_samples,1) -> flatten
        yhat_arr = np.asarray(yhat).ravel()
        yhat_price = (np.expm1(yhat_arr) if target_transform == "log1p" else (np.exp(yhat_arr) if target_transform == "log" else yhat_arr))
        price = float(yhat_price[0])
        st.success(f"💰 Estimated price: {price:,.0f} CZK")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

    # Map (example coords — replace with a proper mapping if available)
    coords = {
        "Praha 1": (50.08804, 14.42076),
        "Praha 2": (50.07366, 14.43424),
        "Praha 5": (50.061, 14.385),
        "Prague 1": (50.08804, 14.42076),
        "Prague 2": (50.07366, 14.43424),
    }
    # try find a matching district key
    dkey = str(district)
    if dkey in coords:
        st.map(pd.DataFrame([{"lat": coords[dkey][0], "lon": coords[dkey][1]}]))

    # Sensitivity: vary usable area while keeping other inputs fixed
    st.subheader("📊 Sensitivity: Price vs. Usable area")
    sizes = np.linspace(20, 200, 30)
    curve_prices = []
    for s in sizes:
        prov2 = provided_ui.copy()
        prov2["usable_area"] = float(s)
        Xtmp = build_model_input(prov2, model_feature_names, sanit_map, label_encoders, feature_defaults)
        try:
            if hasattr(model_obj, "predict"):
                p = model_obj.predict(Xtmp)
            else:
                # take mean of base models as fallback
                base_models = model_obj.get("base_models", model_obj.get("models", {}))
                preds_list = []
                for m in base_models.values():
                    try:
                        cols = list(map(str, m.feature_names_in_)) if hasattr(m, "feature_names_in_") else model_feature_names
                        preds_list.append(m.predict(Xtmp[cols]))
                    except Exception:
                        preds_list.append(m.predict(Xtmp))
                p = np.mean(np.vstack(preds_list), axis=0)
            p_arr = np.asarray(p).ravel()
            val = (np.expm1(p_arr) if target_transform == "log1p" else (np.exp(p_arr) if target_transform == "log" else p_arr))
            curve_prices.append(float(val[0]))
        except Exception:
            curve_prices.append(np.nan)

    df_curve = pd.DataFrame({"usable_area": sizes, "price": curve_prices}).set_index("usable_area")
    st.line_chart(df_curve)

# -----------------------
# Preprocessing summary
# -----------------------
with st.expander("Preprocessing summary & defaults"):
    st.write("Sanitization map (original -> sanitized):")
    st.write(sanit_map if sanit_map else "— none saved")

    st.write("Label encoders (sanitized columns):")
    st.write(sorted(list(label_encoders.keys())))

    st.write("Model expected features (first 30):")
    st.write(model_feature_names[:30])

    if feature_defaults:
        dfd = pd.DataFrame([{"feature": k, "default": v, "type": ("categorical" if k in label_encoders else "numeric")} for k, v in list(feature_defaults.items())[:50]])
        st.dataframe(dfd)
    else:
        st.warning("No feature_defaults found in preprocessing.joblib — the app will fall back to simple defaults.")
