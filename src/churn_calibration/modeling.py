"""Preprocessing pipelines and calibrated churn classifiers."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold


def make_preprocessor(
    numeric_features,
    categorical_features,
    scale_numeric=True,
):
    """Impute missing values, scale numbers, and encode categories."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_models(numeric_features, categorical_features, random_state=42):
    """Build two classifiers and their Platt/isotonic variants."""
    logistic_pipeline = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                    scale_numeric=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=2000),
            ),
        ]
    )

    forest_pipeline = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                    scale_numeric=False,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    base_models = {
        "Logistic Regression": logistic_pipeline,
        "Random Forest": forest_pipeline,
    }

    calibration_folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    models = {}

    for name, estimator in base_models.items():
        models[f"{name} · Uncalibrated"] = estimator

        models[f"{name} · Platt"] = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=calibration_folds,
        )

        models[f"{name} · Isotonic"] = CalibratedClassifierCV(
            estimator=estimator,
            method="isotonic",
            cv=calibration_folds,
        )

    return models