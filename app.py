"""Dashboard for evaluating calibrated churn probabilities."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from churn_calibration.data import load_churn_data, split_feature_types
from churn_calibration.metrics import (
    expected_calibration_error,
    reliability_points,
)
from churn_calibration.modeling import build_models


st.set_page_config(
    page_title="Customer Churn Calibration",
    page_icon="📊",
    layout="wide",
)

# Loads the stylesheet from the assets folder.
st.html(Path(__file__).resolve().parent / "assets" / "style.css")


with st.container(key="dashboard-title"):
    st.title("Customer Churn Probability & Calibration")
    st.write(
        "Compare Logistic Regression and Random Forest probabilities "
        "before and after calibration."
    )

st.caption(
    "Dataset: IBM Telco Customer Churn sample. "
    "This is an educational demonstration."
)


@st.cache_resource(show_spinner=False)
def train_models():
    """Fit models on training data and score a held-out test set."""
    X, y = load_churn_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    numeric_features, categorical_features = split_feature_types(X)

    models = build_models(
        numeric_features,
        categorical_features,
    )

    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict_proba(X_test)[:, 1]

    return X, y, X_test, y_test, models, predictions


with st.spinner(
    "Training models and evaluating calibration on held-out data..."
):
    X, y, X_test, y_test, models, predictions = train_models()


st.subheader("Dataset overview")
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(key="customer-count"):
        st.metric("Customer records", f"{len(X):,}")

with col2:
    with st.container(key="churn-rate"):
        st.metric("Observed churn rate", f"{y.mean():.1%}")

with col3:
    with st.container(key="feature-count"):
        st.metric("Input features", len(X.columns))


with st.expander("Preview dataset"):
    preview = X.head(10).copy()
    preview["Churn"] = y.head(10).map({0: "No", 1: "Yes"})
    st.dataframe(preview, width="stretch")


st.subheader("Model comparison")
rows = []

for name, probabilities in predictions.items():
    rows.append(
        {
            "Model": name,
            "ECE ↓": expected_calibration_error(
                y_test,
                probabilities,
            ),
            "Brier score ↓": brier_score_loss(
                y_test,
                probabilities,
            ),
            "Log loss ↓": log_loss(
                y_test,
                probabilities,
                labels=[0, 1],
            ),
            "ROC AUC ↑": roc_auc_score(
                y_test,
                probabilities,
            ),
        }
    )

results = pd.DataFrame(rows).set_index("Model")
st.dataframe(
    results.style.format("{:.3f}"),
    width="stretch",
)


st.subheader("Reliability diagram")
model_family = st.selectbox(
    "Choose a classifier",
    ["Logistic Regression", "Random Forest"],
)

fig, ax = plt.subplots(figsize=(7, 5))

for calibration in ["Uncalibrated", "Platt", "Isotonic"]:
    name = f"{model_family} · {calibration}"
    probabilities = predictions[name]

    observed_rate, mean_probability = reliability_points(
        y_test,
        probabilities,
    )

    ax.plot(
        mean_probability,
        observed_rate,
        marker="o",
        label=calibration,
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Perfect calibration",
)
ax.set(
    xlabel="Mean predicted churn probability",
    ylabel="Observed churn rate",
    title="Reliability diagram on held-out data",
    xlim=(0, 1),
    ylim=(0, 1),
)
ax.grid(alpha=0.25)
ax.legend()
st.pyplot(fig)
plt.close(fig)


st.subheader("Example probability scores")
examples = X_test.head(10).copy()
examples["Actual churn"] = y_test.head(10).map(
    {0: "No", 1: "Yes"}
).values
examples["Logistic Regression · Platt"] = predictions[
    "Logistic Regression · Platt"
][:10]
examples["Random Forest · Isotonic"] = predictions[
    "Random Forest · Isotonic"
][:10]
st.dataframe(examples, width="stretch")


best_model = results["ECE ↓"].idxmin()
best_ece = results.loc[best_model, "ECE ↓"]

st.subheader("Brief findings")
st.write(
    f"On the held-out test set, **{best_model}** has the lowest "
    f"ECE ({best_ece:.3f}). ECE summarizes the difference between "
    "predicted probabilities and observed churn rates across bins. "
    "Review the reliability diagram and other metrics too; no single "
    "score tells the full story."
)
st.write(
    "In a retention setting, calibrated probabilities can help "
    "prioritize outreach. These results describe this sample and "
    "should not be treated as a guarantee about other customers."
)