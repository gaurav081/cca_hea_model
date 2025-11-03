import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("🔧 Vicker Hardness (HV) Predictor — tailored to your database")
st.markdown(
    "This app is configured for the uploaded dataset structure (the CSV you provided). "
    "It trains a Random Forest on the `HV` column and returns a single HV prediction plus MAE and R² on a held-out test split. "
    "No files are saved."
)

# -----------------------------
# 1) Upload dataset (or use the uploaded CSV)
# -----------------------------
uploaded_file = st.file_uploader("Upload your dataset (.csv or .xlsx) — use the provided database file", type=["csv", "xlsx"])
if uploaded_file is None:
    st.info("👆 Upload the CSV (your database). The app expects a column named `HV` (target) and a column `Composition` (categorical).")
    st.stop()

# load dataframe
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.write(f"Loaded dataset with shape: {df.shape}")

# -----------------------------
# 2) Dataset-specific tweaks (based on your CSV)
# -----------------------------
# Known target in your file:
TARGET_COL = "HV"
if TARGET_COL not in df.columns:
    # fallback detection if name slightly different
    possible_targets = [c for c in df.columns if ('hv' == c.lower()) or ('vick' in c.lower()) or ('vickers' in c.lower()) or ('hard' in c.lower())]
    if len(possible_targets) == 0:
        st.error("Could not find target column 'HV' (or a close name). Available columns: " + ", ".join(df.columns.tolist()))
        st.stop()
    TARGET_COL = possible_targets[0]

st.write(f"Using target column: **{TARGET_COL}**")

# make object columns strings
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str)

# coerce target to numeric and drop rows with missing target
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
n_before = len(df)
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
st.write(f"Rows with valid {TARGET_COL}: {len(df)} (dropped {n_before - len(df)})")

if len(df) < 2:
    st.error("Not enough rows with a valid target to train a model.")
    st.stop()

# -----------------------------
# 3) Features selection (tailored)
# -----------------------------
# Drop obvious index-like column if present
drop_cols = [c for c in df.columns if c.lower().startswith("unnamed") or c.lower() == "index"]
if drop_cols:
    df = df.drop(columns=drop_cols)
    st.write("Dropped index-like columns:", drop_cols)

# Explicitly treat 'Composition' as categorical (your file has this column)
categorical_cols = []
if "Composition" in df.columns:
    categorical_cols.append("Composition")

# Treat any remaining non-numeric as categorical
for c in df.columns:
    if c == TARGET_COL or c in categorical_cols:
        continue
    if not pd.api.types.is_numeric_dtype(df[c]):
        categorical_cols.append(c)

# Numeric cols are the rest (exclude target)
numerical_cols = [c for c in df.columns if c != TARGET_COL and c not in categorical_cols]

st.write("Categorical columns detected (will be one-hot / imputed):", categorical_cols)
st.write("Numerical columns detected (will be median-imputed):", numerical_cols)

# -----------------------------
# 4) Build preprocessing & model pipeline
# -----------------------------
# OneHotEncoder compatibility for sklearn versions
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

transformers = []
if len(categorical_cols) > 0:
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])
    transformers.append(("cat", cat_pipeline, categorical_cols))

if len(numerical_cols) > 0:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    transformers.append(("num", num_pipeline, numerical_cols))

if len(transformers) == 0:
    st.error("No usable features found after preprocessing decisions.")
    st.stop()

preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# -----------------------------
# 5) Train/Test split and training
# -----------------------------
test_size = 0.20
if len(df) < 5:
    # extremely small dataset: train on all (user risk)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = X, X, y, y
else:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

with st.spinner("Training Random Forest..."):
    pipeline.fit(X_train, y_train)
st.success("Model trained (in-memory).")

# -----------------------------
# 6) Compute MAE and R^2 on test set
# -----------------------------
try:
    y_test_pred = pipeline.predict(X_test)
    mae_val = mean_absolute_error(y_test, y_test_pred)
    r2_val = r2_score(y_test, y_test_pred)
    st.subheader("Model performance on held-out test set")
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae_val:.4f}")
    col2.metric("R²", f"{r2_val:.4f}")
except Exception as e:
    st.error(f"Could not compute MAE / R² on test set: {e}")

# -----------------------------
# 7) Input form for a single prediction (no graphs, no files)
# -----------------------------
st.subheader("🔎 Enter features for a single HV prediction")
st.markdown("Fill values below and press **Predict HV**. The app will return only the predicted HV value (and performance metrics shown above).")

# Order inputs: categorical first, then numeric
user_input = {}
for col in categorical_cols + numerical_cols:
    if col in categorical_cols:
        opts = list(df[col].dropna().unique())
        if len(opts) == 0:
            user_input[col] = st.text_input(f"{col} (categorical)", value="")
        else:
            # if many options, use text_input to avoid a huge selectbox
            if len(opts) > 50:
                example = ", ".join(map(str, opts[:8])) + ("..." if len(opts) > 8 else "")
                user_input[col] = st.text_input(f"{col} (categorical) — examples: {example}", value=str(opts[0]))
            else:
                user_input[col] = st.selectbox(f"{col}", options=opts, index=0)
    else:
        # numeric
        try:
            med = float(df[col].median()) if df[col].notna().any() else 0.0
        except Exception:
            med = 0.0
        step = 0.01 if abs(med) < 1 else 0.1
        user_input[col] = st.number_input(f"{col}", value=med, step=step, format="%.6f")

if st.button("Predict HV"):
    # create DataFrame in same column order
    ordered_cols = categorical_cols + numerical_cols
    new_df = pd.DataFrame([user_input], columns=ordered_cols)
    # cast numerics
    for nc in numerical_cols:
        new_df[nc] = pd.to_numeric(new_df[nc], errors="coerce")
    try:
        pred = pipeline.predict(new_df)[0]
        st.success(f"Predicted Vicker Hardness (HV): **{pred:.4f}**")
        # also repeat MAE and R2 for convenience
        try:
            st.write(f"MAE on test set: **{mae_val:.4f}**,  R² on test set: **{r2_val:.4f}**")
        except Exception:
            pass
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------------
# End — no files saved, no graphs displayed
# -----------------------------
