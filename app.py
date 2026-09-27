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