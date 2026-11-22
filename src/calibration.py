"""
Probability Calibration and Uncertainty Quantification Module.
Implements Platt scaling (sigmoid calibration) via CalibratedClassifierCV and defines
a multi-tier human-in-the-loop underwriting triage system based on predictive uncertainty.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.config import AUTO_APPROVE_THRESHOLD, AUTO_DECLINE_THRESHOLD, RANDOM_STATE


def calibrate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "sigmoid",
    cv: int = 5
) -> CalibratedClassifierCV:
    """
    Applies Platt scaling (method='sigmoid') using cross-validation on the training set.
    """
    calibrated_clf = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv
    )
    calibrated_clf.fit(X_train, y_train)
    return calibrated_clf


def compute_calibration_metrics(
    y_true: np.ndarray,
    prob_uncalibrated: np.ndarray,
    prob_calibrated: np.ndarray
) -> Dict[str, float]:
    """
    Computes Brier Score loss before and after calibration.
    Lower Brier score indicates better probability calibration.
    """
    bs_uncal = brier_score_loss(y_true, prob_uncalibrated)
    bs_cal = brier_score_loss(y_true, prob_calibrated)
    return {
        "brier_score_uncalibrated": float(bs_uncal),
        "brier_score_calibrated": float(bs_cal),
        "improvement_pct": float((bs_uncal - bs_cal) / bs_uncal * 100)
    }


def triage_loan_applications(
    probabilities: np.ndarray,
    auto_approve_th: float = AUTO_APPROVE_THRESHOLD,
    auto_decline_th: float = AUTO_DECLINE_THRESHOLD
) -> pd.DataFrame:
    """
    Categorizes loan applicants into 3 operational tiers:
      - Tier 1: Auto-Approve (Low Risk: P(Default) < auto_approve_th)
      - Tier 2: Human Underwriter Review (High Uncertainty: auto_approve_th <= P <= auto_decline_th)
      - Tier 3: Auto-Decline (High Risk: P(Default) > auto_decline_th)

    Returns:
        DataFrame with columns: ['default_prob', 'triage_decision', 'uncertainty_score']
    """
    decisions = []
    uncertainties = []

    for p in probabilities:
        # Distance to maximum uncertainty (0.50): uncertainty = 1 - 2 * |p - 0.5|
        unc = float(1.0 - 2.0 * abs(p - 0.5))
        uncertainties.append(unc)

        if p < auto_approve_th:
            decisions.append("Auto-Approve")
        elif p > auto_decline_th:
            decisions.append("Auto-Decline")
        else:
            decisions.append("Human Underwriter Review")

    df_triage = pd.DataFrame({
        "default_probability": probabilities,
        "triage_decision": decisions,
        "uncertainty_score": uncertainties
    })
    return df_triage


def plot_calibration_curves(
    uncalibrated_model: Any,
    calibrated_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    save_path: str
):
    """
    Generates and saves a reliability curve comparing uncalibrated vs Platt-calibrated models.
    """
    prob_uncal = uncalibrated_model.predict_proba(X_test)[:, 1]
    prob_cal = calibrated_model.predict_proba(X_test)[:, 1]

    fraction_of_pos_uncal, mean_predicted_value_uncal = calibration_curve(y_test, prob_uncal, n_bins=10)
    fraction_of_pos_cal, mean_predicted_value_cal = calibration_curve(y_test, prob_cal, n_bins=10)

    bs_uncal = brier_score_loss(y_test, prob_uncal)
    bs_cal = brier_score_loss(y_test, prob_cal)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated (Ideal)", alpha=0.7)
    plt.plot(
        mean_predicted_value_uncal,
        fraction_of_pos_uncal,
        "s-",
        color="#d9534f",
        label=f"Uncalibrated {model_name} (Brier={bs_uncal:.4f})"
    )
    plt.plot(
        mean_predicted_value_cal,
        fraction_of_pos_cal,
        "o-",
        color="#2b5b84",
        label=f"Platt Calibrated (Brier={bs_cal:.4f})"
    )

    plt.title(f"Reliability Diagram: Platt Calibration ({model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Predicted Default Probability", fontsize=11)
    plt.ylabel("Fraction of True Defaults (Empirical)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

