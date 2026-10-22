import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="EV Purchase Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Electric Vehicle Purchase Prediction")
st.write("Data analysis dashboard for EV purchase behaviour.")

# Load dataset
df = pd.read_csv("ev_data.csv")

# ========================================================
# SIDEBAR
# ========================================================

st.sidebar.header("Dataset Controls")

show_data = st.sidebar.checkbox("Show Dataset")

# ========================================================
# DATASET OVERVIEW
# ========================================================

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Total Features", len(df.columns))
col3.metric(
    "Potential EV Buyers",
    f"{(df['Will_Buy_EV'] == 'Yes').sum():,}"
)

# ========================================================
# DATASET
# ========================================================

if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(100))

# ========================================================
# EV PURCHASE DISTRIBUTION
# ========================================================

st.header("EV Purchase Distribution")

fig, ax = plt.subplots()

sns.countplot(
    data=df,
    x="Will_Buy_EV",
    ax=ax
)

ax.set_xlabel("Will Buy EV")
ax.set_ylabel("Number of Customers")

st.pyplot(fig)

# ========================================================
# ENVIRONMENTAL CONCERN
# ========================================================

st.header("Environmental Concern vs EV Purchase")

fig, ax = plt.subplots()

sns.countplot(
    data=df,
    x="Environmental_Concern_Level",
    hue="Will_Buy_EV",
    ax=ax
)

ax.set_xlabel("Environmental Concern Level")
ax.set_ylabel("Number of Customers")

st.pyplot(fig)

# ========================================================
# INCOME
# ========================================================

st.header("Annual Income Distribution")

fig, ax = plt.subplots()

sns.histplot(
    data=df,
    x="Annual_Income_USD",
    bins=30,
    ax=ax
)

ax.set_xlabel("Annual Income (USD)")
ax.set_ylabel("Number of Customers")

st.pyplot(fig)

# ========================================================
# PROJECT RESULTS
# ========================================================

st.header("Machine Learning Results")

col1, col2 = st.columns(2)

col1.metric("Accuracy", "89.49%")
col2.metric("ROC-AUC", "0.938")

st.success("Dashboard loaded successfully!")
# ==========================================
# EV PURCHASE PREDICTION
# ==========================================

import streamlit as st
import pandas as pd

st.markdown("---")
st.header("⚡ Predict EV Purchase")
st.write(
    "Enter customer details below to predict "
    "whether they are likely to purchase an electric vehicle."
)

# Input form
with st.form("ev_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=28
        )

        income = st.number_input(
            "Annual Income (₹)",
            min_value=0,
            value=800000,
            step=50000
        )

        commute_distance = st.number_input(
            "Daily Commute Distance (km)",
            min_value=0.0,
            value=25.0,
            step=1.0
        )

    with col2:
        home_charging = st.selectbox(
            "Home Charging Availability",
            ["Yes", "No"]
        )

        environmental_awareness = st.selectbox(
            "Environmental Awareness",
            ["Low", "Medium", "High"]
        )

        vehicle_preference = st.selectbox(
            "Vehicle Preference",
            ["Electric Car", "Electric Two-Wheeler", "Other"]
        )

    submitted = st.form_submit_button(
        "🔍 Predict EV Purchase",
        use_container_width=True
    )


# ==========================================
# MAKE PREDICTION
# ==========================================

if submitted:

    # Replace these example feature names and values
    # with the exact features expected by your trained model.
    input_data = pd.DataFrame([{
        "Age": age,
        "Annual_Income": income,
        "Daily_Commute_Distance": commute_distance,
        "Home_Charging": home_charging,
        "Environmental_Awareness": environmental_awareness,
        "Vehicle_Preference": vehicle_preference
    }])

    st.subheader("Prediction Result")

    # Connect this section to your actual trained model.
    # For example, if your loaded model is named `model`:
    #
    # prediction = model.predict(input_data)[0]
    #
    # if hasattr(model, "predict_proba"):
    #     probabilities = model.predict_proba(input_data)[0]
    #     probability = max(probabilities) * 100
    #
    # Do not uncomment these lines until the feature names,
    # preprocessing, and model variable match your project.

    st.warning(
        "The input form is ready. Connect your trained model "
        "and its preprocessing pipeline to generate predictions."
    )

    st.write("**Customer details submitted:**")
    st.dataframe(input_data, use_container_width=True)
