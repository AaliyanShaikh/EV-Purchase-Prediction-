import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import clone


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EV Purchase Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Electric Vehicle Purchase Prediction")
st.write(
    "Analyze customer behaviour and predict the likelihood "
    "of purchasing an electric vehicle."
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv("ev_data.csv")


try:
    df = load_dataset()
except FileNotFoundError:
    st.error(
        "Dataset not found. Make sure ev_data.csv is present "
        "in your GitHub repository."
    )
    st.stop()


TARGET = "Will_Buy_EV"

if TARGET not in df.columns:
    st.error(f"The dataset must contain a '{TARGET}' column.")
    st.stop()

if len(df) < 10:
    st.error("The dataset is too small to train a reliable model.")
    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

data = df.copy()

# Remove ID column if present.
id_columns = [
    col for col in data.columns
    if col.lower() == "id"
]

data = data.drop(columns=id_columns, errors="ignore")

# Remove rows with missing target values.
data = data.dropna(subset=[TARGET])

# Convert Yes/No targets to 1/0.
target_values = data[TARGET].astype(str).str.strip().str.lower()

if set(target_values.unique()) == {"yes", "no"}:
    y = target_values.map({"no": 0, "yes": 1})
    target_labels = {0: "Unlikely to Purchase", 1: "Likely to Purchase"}
else:
    encoder = LabelEncoder()
    y = pd.Series(
        encoder.fit_transform(data[TARGET].astype(str)),
        index=data.index
    )
    target_labels = {
        i: str(label)
        for i, label in enumerate(encoder.classes_)
    }

X = data.drop(columns=[TARGET])

if X.shape[1] == 0:
    st.error("No usable prediction features were found.")
    st.stop()

if y.nunique() < 2:
    st.error("The target must contain at least two different classes.")
    st.stop()


# ============================================================
# TRAIN MODEL AND PREPROCESSING PIPELINE
# ============================================================

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Detect numeric and categorical features.
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        col for col in X.columns
        if col not in numeric_columns
    ]

    # Numeric preprocessing.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Train on the training split.
    model.fit(X_train, y_train)

    # Evaluate on unseen test data.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    auc = None

    if len(model.classes_) == 2:
        probabilities = model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, probabilities)
        except ValueError:
            auc = None

    return model, accuracy, auc


with st.spinner("Training the EV purchase prediction model..."):
    try:
        model, accuracy, auc = train_model(X, y)
    except Exception as error:
        st.error(f"Could not train the model: {error}")
        st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dataset Controls")

show_data = st.sidebar.checkbox("Show Dataset")


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Total Features", len(X))
col3.metric(
    "Potential EV Buyers",
    f"{(target_values.loc[data.index] == 'yes').sum():,}"
    if set(target_values.unique()) == {"yes", "no"}
    else f"{(y == 1).sum():,}"
)


# ============================================================
# DATASET PREVIEW
# ============================================================

if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)


# ============================================================
# EV PURCHASE DISTRIBUTION
# ============================================================

st.header("EV Purchase Distribution")

fig, ax = plt.subplots()

sns.countplot(
    data=df,
    x=TARGET,
    ax=ax
)

ax.set_xlabel("EV Purchase Decision")
ax.set_ylabel("Number of Customers")

st.pyplot(fig)
plt.close(fig)


# ============================================================
# ENVIRONMENTAL CONCERN
# ============================================================

if "Environmental_Concern_Level" in df.columns:

    st.header("Environmental Concern vs EV Purchase")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="Environmental_Concern_Level",
        hue=TARGET,
        ax=ax
    )

    ax.set_xlabel("Environmental Concern Level")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# INCOME DISTRIBUTION
# ============================================================

if "Annual_Income_USD" in df.columns:

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
    plt.close(fig)


# ============================================================
# MACHINE LEARNING RESULTS
# ============================================================

st.header("🤖 Machine Learning Results")

col1, col2 = st.columns(2)

col1.metric("Test Accuracy", f"{accuracy:.2%}")

if auc is not None:
    col2.metric("Test ROC-AUC", f"{auc:.3f}")
else:
    col2.metric("Test ROC-AUC", "Unavailable")


# ============================================================
# EV PURCHASE PREDICTION FORM
# ============================================================

st.markdown("---")

st.header("⚡ Predict EV Purchase")

st.write(
    "Enter customer details below. The trained Logistic "
    "Regression model will predict the customer's EV purchase decision."
)

st.info(
    "The form uses the same feature columns as your dataset. "
    "The ID and target columns are excluded automatically."
)

with st.form("ev_prediction_form"):

    st.subheader("Customer Information")

    input_values = {}

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        col for col in X.columns
        if col not in numeric_columns
    ]

    col1, col2 = st.columns(2)

    # Numeric inputs.
    for index, column in enumerate(numeric_columns):

        series = pd.to_numeric(X[column], errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan)
        valid_values = series.dropna()

        if valid_values.empty:
            default_value = 0.0
            minimum = 0.0
            maximum = 100.0
        else:
            default_value = float(valid_values.median())
            minimum = float(valid_values.min())
            maximum = float(valid_values.max())

        if minimum == maximum:
            minimum = min(minimum, default_value - 1)
            maximum = max(maximum, default_value + 1)

        step = 1.0 if series.dropna().apply(
            lambda value: float(value).is_integer()
        ).all() else 0.1

        container = col1 if index % 2 == 0 else col2

        with container:
            if step == 1.0:
                input_values[column] = st.number_input(
                    column.replace("_", " "),
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(np.clip(
                        round(default_value),
                        minimum,
                        maximum
                    )),
                    step=1,
                    key=f"numeric_{column}"
                )
            else:
                input_values[column] = st.number_input(
                    column.replace("_", " "),
                    min_value=minimum,
                    max_value=maximum,
                    value=float(np.clip(
                        default_value,
                        minimum,
                        maximum
                    )),
                    step=step,
                    key=f"numeric_{column}"
                )

    # Categorical inputs.
    for index, column in enumerate(categorical_columns):

        values = (
            X[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if not values:
            values = ["Unknown"]

        mode_values = X[column].dropna().astype(str).mode()
        default_value = (
            mode_values.iloc[0]
            if not mode_values.empty
            else values[0]
        )

        default_index = (
            values.index(default_value)
            if default_value in values
            else 0
        )

        container = col1 if index % 2 == 0 else col2

        with container:
            input_values[column] = st.selectbox(
                column.replace("_", " "),
                options=values,
                index=default_index,
                key=f"category_{column}"
            )

    submitted = st.form_submit_button(
        "🔍 Predict EV Purchase",
        use_container_width=True
    )


# ============================================================
# PREDICTION RESULT
# ============================================================

if submitted:

    input_data = pd.DataFrame(
        [input_values],
        columns=X.columns
    )

    try:
        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.subheader("🔮 Prediction Result")

        predicted_label = target_labels.get(
            int(prediction),
            str(prediction)
        )

        if len(model.classes_) == 2:

            probabilities = model.predict_proba(input_data)[0]

            positive_class = (
                1 if 1 in model.classes_
                else model.classes_[-1]
            )

            positive_index = list(model.classes_).index(
                positive_class
            )

            probability = float(
                probabilities[positive_index] * 100
            )

            positive_label = target_labels.get(
                int(positive_class),
                str(positive_class)
            )

            if positive_label == "Likely to Purchase":

                if prediction == positive_class:
                    st.success("✅ Likely to Purchase an EV")
                else:
                    st.warning("❌ Unlikely to Purchase an EV")

                st.metric(
                    "Estimated Purchase Probability",
                    f"{probability:.2f}%"
                )

                st.progress(
                    float(np.clip(probability / 100, 0, 1))
                )

                st.caption(
                    "Probability of the positive class predicted "
                    "by the Logistic Regression model."
                )

            else:
                st.success(f"Predicted class: {predicted_label}")

                st.metric(
                    f"Probability of {positive_label}",
                    f"{probability:.2f}%"
                )

        else:
            st.success(f"Predicted class: {predicted_label}")

        with st.expander("View submitted customer details"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "EV Purchase Prediction | Logistic Regression | "
    "Predictions are estimates based on the training dataset."
)
