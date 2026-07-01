import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import pickle
import numpy as np

# ── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = torch.load("ann_model_full.pth", weights_only=False)
    model.eval()

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)          # fitted on Gender

    with open("one_hot_encoder_geo.pkl", "rb") as f:
        ohe = pickle.load(f)         # fitted on Geography

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, le, ohe, scaler

model, label_encoder, one_hot_encoder_geo, scaler = load_artifacts()

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("Customer Churn Prediction")
st.markdown("Fill in the customer details below and click **Predict** to see the churn probability.")

col1, col2 = st.columns(2)

with col1:
    geo_input            = st.selectbox("Geography",          ["France", "Spain", "Germany"])
    gender_input         = st.selectbox("Gender",             ["Male", "Female"])
    age_input            = st.slider("Age",                   min_value=18,    max_value=92)
    balance_input        = st.number_input("Balance",         min_value=0.0,   max_value=250000.0)
    credit_score_input   = st.number_input("Credit Score",    min_value=350,   max_value=850)

with col2:
    estimated_salary_input = st.number_input("Estimated Salary", min_value=10000.0, max_value=200000.0)
    tenure_input           = st.number_input("Tenure (years)",   min_value=0,   max_value=10)
    num_of_products_input  = st.slider("Number of Products",     min_value=1,   max_value=4)
    has_cr_card_input      = st.selectbox("Has Credit Card",     ["Yes", "No"])
    is_active_member_input = st.selectbox("Is Active Member",    ["Yes", "No"])

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("Predict", use_container_width=True, type="primary"):

    # 1. Raw DataFrame (matches training column order)
    input_df = pd.DataFrame({
        "CreditScore":      [credit_score_input],
        "Gender":           [gender_input],
        "Age":              [age_input],
        "Tenure":           [tenure_input],
        "Balance":          [balance_input],
        "NumOfProducts":    [num_of_products_input],
        "HasCrCard":        [1 if has_cr_card_input      == "Yes" else 0],
        "IsActiveMember":   [1 if is_active_member_input == "Yes" else 0],
        "EstimatedSalary":  [estimated_salary_input],
        "Geography":        [geo_input],
    })

    # 2. Encode Gender with LabelEncoder
    input_df["Gender"] = label_encoder.transform(input_df["Gender"])

    # 3. One-hot encode Geography and merge
    geo_encoded = one_hot_encoder_geo.transform(input_df[["Geography"]])

    # Convert sparse matrix → dense numpy array before passing to DataFrame
    if hasattr(geo_encoded, "toarray"):
        geo_encoded = geo_encoded.toarray()

    geo_columns = one_hot_encoder_geo.get_feature_names_out(["Geography"])
    geo_df = pd.DataFrame(geo_encoded, columns=geo_columns, index=input_df.index)
    input_df = pd.concat([input_df.drop(columns=["Geography"]), geo_df], axis=1)

    # 4. Scale
    input_scaled = scaler.transform(input_df)

    # 5. Inference
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        # Support both raw logits (BCEWithLogitsLoss) and sigmoid output (BCELoss)
        if output.min() < 0 or output.max() > 1:
            prob = torch.sigmoid(output).item()
        else:
            prob = output.item()

    # 6. Display result
    st.divider()
    pct = prob * 100

    if prob >= 0.5:
        st.error(f"**High churn risk** — {pct:.1f}% probability of churning")
    else:
        st.success(f" **Low churn risk** — {pct:.1f}% probability of churning")

    st.progress(prob, text=f"Churn probability: {pct:.1f}%")

    with st.expander("View processed input sent to model"):
        st.dataframe(pd.DataFrame(input_scaled, columns=input_df.columns), use_container_width=True)