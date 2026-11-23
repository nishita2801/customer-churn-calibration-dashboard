"""Load and prepare the IBM Telco churn dataset."""

from pathlib import Path

import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)
TARGET_COLUMN = "Churn"


def load_churn_data(path: Path = DATA_PATH):
    """Load churn features and a numeric binary target."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download it into data/raw/Telco-Customer-Churn.csv."
        )

    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset must contain '{TARGET_COLUMN}'.")

    # TotalCharges contains blank strings in some rows.
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(
            frame["TotalCharges"],
            errors="coerce",
        )

    target = frame.pop(TARGET_COLUMN).astype("string").str.strip()
    y = target.map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError(
            "The Churn column contains missing or unexpected values."
        )

    # CustomerID identifies a record; it is not a predictive feature.
    X = frame.drop(columns=["customerID"], errors="ignore")

    return X.reset_index(drop=True), y.astype(int).reset_index(drop=True)


def split_feature_types(X: pd.DataFrame):
    """Identify numeric and categorical input columns."""
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    return numeric_features, categorical_features