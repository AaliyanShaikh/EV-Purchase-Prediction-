# ============================================================
# ELECTRIC VEHICLE PURCHASE PREDICTION
# Python Mini Project
# ============================================================

import pandas as pd
import numpy as np
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# 1. BASIC PYTHON DATA TYPES
# ============================================================

project_name = "Electric Vehicle Purchase Prediction"
version = 1
is_classification_problem = True

print("=" * 60)
print(project_name)
print("Version:", version)
print("Classification Problem:", is_classification_problem)
print("=" * 60)


# ============================================================
# 2. BASIC DATA STRUCTURES
# ============================================================

# List
important_features = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Environmental_Concern_Level",
    "Subsidy_Available"
]

# Tuple
model_information = (
    "Logistic Regression",
    "Binary Classification",
    "ROC-AUC"
)

# Dictionary
project_info = {
    "name": project_name,
    "model": "Logistic Regression",
    "target": "Will_Buy_EV"
}

# Set
model_types = {
    "Logistic Regression",
    "Decision Tree",
    "Random Forest"
}


# ============================================================
# 3. FUNCTIONS
# ============================================================

def display_project_info():
    """Display basic project information."""

    print("\nPROJECT INFORMATION")
    print("-" * 40)

    for key, value in project_info.items():
        print(f"{key.title()}: {value}")


def check_file(filename):
    """Check whether the dataset exists."""

    if os.path.exists(filename):
        print(f"\nFile found: {filename}")
        return True

    print(f"\nFile not found: {filename}")
    return False


def load_dataset(filename):
    """Load the CSV dataset."""

    try:
        df = pd.read_csv(filename)

        print("\nDataset loaded successfully!")
        print("Dataset shape:", df.shape)

        return df

    except FileNotFoundError:
        print("Error: Dataset file was not found.")
        return None

    except Exception as error:
        print("Unexpected error:", error)
        return None


# ============================================================
# 4. OBJECT-ORIENTED PROGRAMMING
# ============================================================

class DatasetManager:
    """
    Base class for managing a dataset.

    Demonstrates:
    - Class
    - Object
    - Constructor
    - self
    """

    def __init__(self, filename):
        self.filename = filename
        self.data = None

    def load_data(self):
        """Load the dataset."""

        try:
            self.data = pd.read_csv(self.filename)
            print("\nDataset loaded using DatasetManager.")

        except Exception as error:
            print("Error:", error)

    def show_basic_info(self):
        """Display basic dataset information."""

        if self.data is not None:
            print("\nDATASET INFORMATION")
            print("-" * 40)
            print("Rows:", self.data.shape[0])
            print("Columns:", self.data.shape[1])


# ============================================================
# 5. INHERITANCE + METHOD OVERRIDING
# ============================================================

class EVDataManager(DatasetManager):
    """
    Child class inheriting from DatasetManager.
    """

    def __init__(self, filename):
        super().__init__(filename)

    # Method overriding
    def show_basic_info(self):

        print("\nEV DATASET INFORMATION")
        print("-" * 40)

        if self.data is not None:
            print("Rows:", self.data.shape[0])
            print("Columns:", self.data.shape[1])
            print("Target:", "Will_Buy_EV")
def create_visualizations(df):

    os.makedirs("visualizations", exist_ok=True)

    # ========================================================
    # 1. EV PURCHASE DISTRIBUTION
    # ========================================================

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Will_Buy_EV"
    )

    plt.title("EV Purchase Distribution")
    plt.xlabel("Will Buy EV")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.savefig("visualizations/ev_purchase_distribution.png")
    plt.show()

    # ========================================================
    # 2. INCOME DISTRIBUTION
    # ========================================================

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="Annual_Income_USD",
        bins=30
    )

    plt.title("Annual Income Distribution")
    plt.xlabel("Annual Income (USD)")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.savefig("visualizations/income_distribution.png")
    plt.show()

    # ========================================================
    # 3. ENVIRONMENTAL CONCERN VS EV PURCHASE
    # ========================================================

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Environmental_Concern_Level",
        hue="Will_Buy_EV"
    )

    plt.title("Environmental Concern vs EV Purchase")
    plt.xlabel("Environmental Concern Level")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.savefig(
        "visualizations/environmental_concern_vs_purchase.png"
    )
    plt.show()

def train_ml_model(df):

    print("\n" + "=" * 60)
    print("MACHINE LEARNING - LOGISTIC REGRESSION")
    print("=" * 60)

    # Remove ID
    df = df.drop(columns=["id"])

    # Convert target
    df["Will_Buy_EV"] = df["Will_Buy_EV"].map({
        "No": 0,
        "Yes": 1
    })

    # Find categorical columns
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    X = df.drop(columns=["Will_Buy_EV"])
    y = df["Will_Buy_EV"]

    # Convert categorical data to numerical data
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Feature scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create model
    model = LogisticRegression(max_iter=1000)

    # Train model
    model.fit(X_train_scaled, y_train)

    print("\nModel training completed.")

    # Predictions
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\nMODEL PERFORMANCE")
    print("-" * 40)
    print("Accuracy:", round(accuracy, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # ========================================================
    # SAVE PREDICTIONS TO FILE
    # ========================================================



    print("\nPredictions saved to predictions.csv")

    return model

# ============================================================
# 6. MAIN PROGRAM
# ============================================================

def main():

    filename = "ev_data.csv"

    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    display_project_info()

    # ========================================================
    # CHECK DATASET
    # ========================================================

    if not check_file(filename):
        return

    # ========================================================
    # LOAD DATASET
    # ========================================================

    df = load_dataset(filename)

    if df is None:
        return

    # ========================================================
    # DISPLAY DATASET
    # ========================================================

    print("\nFIRST 5 ROWS")
    print(df.head())

    print("\nCOLUMNS")
    print(df.columns.tolist())

    # ========================================================
    # OOP - CREATE OBJECT
    # ========================================================

    ev_manager = EVDataManager(filename)

    ev_manager.load_data()
    ev_manager.show_basic_info()

    # ========================================================
    # PANDAS + NUMPY DATA ANALYSIS
    # ========================================================

    print("\n" + "=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)

    print("\nBasic Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nEV Purchase Distribution:")
    print(df["Will_Buy_EV"].value_counts())

    purchase_count = (df["Will_Buy_EV"] == "Yes").sum()
    total_customers = len(df)

    purchase_percentage = (purchase_count / total_customers) * 100

    print(
        "\nPercentage willing to buy EV:",
        round(purchase_percentage, 2),
        "%"
    )

    # NumPy analysis
    income = df["Annual_Income_USD"].dropna().to_numpy()

    print("\nNumPy Income Analysis:")
    print("Mean:", round(np.mean(income), 2))
    print("Median:", round(np.median(income), 2))
    print("Minimum:", round(np.min(income), 2))
    print("Maximum:", round(np.max(income), 2))
        # ========================================================
    # DATA VISUALIZATION
    # ========================================================

    print("\nCreating visualizations...")

    create_visualizations(df)

    print("Visualizations saved in 'visualizations' folder.")
    print("\nCreating visualizations...")

    create_visualizations(df)

    print("Visualizations saved in 'visualizations' folder.")
        # ========================================================
    # SAVE PREDICTIONS TO FILE
    # ========================================================

    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions,
        "Purchase_Probability": probabilities
    })

    results.to_csv(
        "predictions.csv",
        index=False
    )

    print("\nPredictions saved to predictions.csv")

    return model

    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    train_ml_model(df)

# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

    