"""Calibration metrics and reliability-curve points."""

import numpy as np
from sklearn.calibration import calibration_curve


def expected_calibration_error(y_true, probabilities, n_bins=10):
    """Calculate ECE using equally sized probability ranges."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for index in range(n_bins):
        if index == n_bins - 1:
            in_bin = (
                (probabilities >= edges[index])
                & (probabilities <= edges[index + 1])
            )
        else:
            in_bin = (
                (probabilities >= edges[index])
                & (probabilities < edges[index + 1])
            )

        if in_bin.any():
            average_confidence = probabilities[in_bin].mean()
            observed_rate = y_true[in_bin].mean()
            ece += in_bin.mean() * abs(
                observed_rate - average_confidence
            )

    return float(ece)


def reliability_points(y_true, probabilities, n_bins=10):
    """Return observed churn rates and mean predicted probabilities per bin."""
    return calibration_curve(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy="uniform",
    )