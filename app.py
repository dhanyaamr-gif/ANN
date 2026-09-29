import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

# Load preprocessor
preprocessor = joblib.load("bank_preprocessor.pkl")

# Create model
model = Sequential([
    Input(shape=(65,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Load trained weights
model.load_weights("bank_ann.weights.h5")

st.title("🏦 Bank Marketing Prediction")

age = st.number_input("Age", 18, 100, 30)

job = st.selectbox("Job", [
    "admin.", "blue-collar", "entrepreneur", "housemaid",
    "management", "retired", "self-employed", "services",
    "student", "technician", "unemployed", "unknown"
])

marital = st.selectbox("Marital", [
    "married", "single", "divorced", "unknown"
])

education = st.selectbox("Education", [
    "basic.4y", "basic.6y", "basic.9y",
    "high.school", "illiterate", "professional.course",
    "university.degree", "unknown"
])

default = st.selectbox("Default", ["no", "yes", "unknown"])
housing = st.selectbox("Housing Loan", ["no", "yes", "unknown"])
loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])

contact = st.selectbox("Contact", ["cellular", "telephone"])

month = st.selectbox("Month", [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
])

day_of_week = st.selectbox("Day of Week", [
    "mon", "tue", "wed", "thu", "fri"
])

duration = st.number_input("Duration", 0, 5000, 300)
campaign = st.number_input("Campaign", 1, 50, 1)
pdays = st.number_input("Days Since Previous Contact", -1, 1000, -1)
previous = st.number_input("Previous Contacts", 0, 20, 0)

poutcome = st.selectbox("Previous Outcome", [
    "failure", "nonexistent", "success"
])

emp_var_rate = st.number_input("Employment Variation Rate", value=1.1)
cons_price_idx = st.number_input("Consumer Price Index", value=93.994)
cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4)
euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857)
nr_employed = st.number_input("Number of Employees", value=5191.0)


if st.button("Predict"):

    # Create input data FIRST
    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed
    }])

    # Feature engineering
    input_data["previously_contacted"] = (
        input_data["pdays"] != -1
    ).astype(int)

    input_data["total_contacts"] = (
        input_data["campaign"] + input_data["previous"]
    )

    input_data["has_any_loan"] = (
        (input_data["housing"] == "yes") |
        (input_data["loan"] == "yes")
    ).astype(int)

    # Preprocess
    processed_data = preprocessor.transform(input_data)

    # Convert sparse matrix if necessary
    if hasattr(processed_data, "toarray"):
        processed_data = processed_data.toarray()

    # Prediction
    probability = model.predict(processed_data, verbose=0)[0][0]

    if probability >= 0.5:
        st.success("✅ Customer is likely to subscribe!")
    else:
        st.warning("❌ Customer is unlikely to subscribe.")

    st.write(f"Probability of subscription: **{probability * 100:.2f}%**")