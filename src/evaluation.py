"""
Evaluation Module.
Computes comprehensive predictive performance metrics:
- ROC-AUC
- Precision-Recall AUC (PR-AUC)
- F1-Score (Default class)
- False Positive Rate (FPR = FP / (FP + TN), corresponding to denying creditworthy applicants)
- Brier Score
- Explainability Quality: Cross-validation consistency of SHAP importance rankings.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    accuracy_score
)
from sklearn.model_selection import StratifiedKFold
import shap

from src.config import RANDOM_STATE


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, float]:
    """
    Computes all project proposal metrics for a model.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate = FP / (FP + TN) : Denying creditworthy applicants
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    # False Negative Rate = FN / (FN + TP) : Approving defaulting borrowers (Credit Loss)
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "F1-Score": f1,
        "False_Positive_Rate (Denying Good)": fpr,
        "False_Negative_Rate (Missed Default)": fnr,
        "Accuracy": acc,
        "Brier_Score": brier,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }


def evaluate_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50
) -> pd.DataFrame:
    """
    Evaluates multiple models on the hold-out test set and returns a comparison table.
    """
    records = []
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
        
        metrics = compute_comprehensive_metrics(y_test.values, y_prob, threshold=threshold)
        metrics["Model"] = name
        records.append(metrics)

    df_results = pd.DataFrame(records).set_index("Model")
    return df_results


def evaluate_shap_consistency(
    model_class,
    model_params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE
) -> Tuple[float, np.ndarray, List[float]]:
    """
    Computes Explainability Quality: Consistency of SHAP feature importance rankings
    across stratified cross-validation folds using pairwise Spearman rank correlations.

    Returns:
        mean_spearman_rho, importance_matrix, pairwise_correlations
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_importances = []

    print(f"--> Evaluating SHAP consistency across {n_splits} CV folds...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        clf = model_class(**model_params)
        clf.fit(X_tr, y_tr)

        # Compute SHAP values on validation fold
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_val)

        # Handle binary classification output formats
        if isinstance(shap_values, list):
            sv = shap_values[1]  # positive class (default)
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        # Mean absolute SHAP importance per feature
        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        fold_importances.append(mean_abs_shap)

    importance_matrix = np.array(fold_importances)  # shape: (n_splits, n_features)

    # Compute pairwise Spearman correlations between folds
    pairwise_rhos = []
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            rho, _ = spearmanr(importance_matrix[i], importance_matrix[j])
            pairwise_rhos.append(rho)

    mean_rho = float(np.mean(pairwise_rhos))
    return mean_rho, importance_matrix, pairwise_rhos


def plot_roc_and_pr_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str
):
    """
    Plots side-by-side ROC Curves and Precision-Recall Curves for all evaluated models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    # ROC Curve
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance (AUC = 0.500)")
    
    # PR Curve
    ax_pr = axes[1]
    baseline_pr = float(y_test.mean())
    ax_pr.plot([0, 1], [baseline_pr, baseline_pr], "k--", alpha=0.6, label=f"No-Skill Baseline ({baseline_pr:.3f})")

    for idx, (name, model) in enumerate(models_dict.items()):
        color = colors[idx % len(colors)]
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc_score:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_score = average_precision_score(y_test, y_prob)
        ax_pr.plot(rec, prec, color=color, lw=2, label=f"{name} (PR-AUC = {pr_score:.3f})")

    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (Denying Creditworthy)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Default Recall)", fontsize=11)
    ax_roc.legend(loc="lower right", frameon=True)
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    ax_pr.set_title("Precision-Recall Curve (PR-AUC)", fontsize=13, fontweight="bold")
    ax_pr.set_xlabel("Recall (Defaults Captured)", fontsize=11)
    ax_pr.set_ylabel("Precision (Defaults Identified Correctly)", fontsize=11)
    ax_pr.legend(loc="lower left", frameon=True)
    ax_pr.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_metrics_comparison_bar(df_results: pd.DataFrame, save_path: str):
    """
    Plots a multi-bar chart comparing ROC-AUC, PR-AUC, F1-Score, and FPR.
    """
    plot_df = df_results[["ROC-AUC", "PR-AUC", "F1-Score", "False_Positive_Rate (Denying Good)"]].copy()
    plot_df.columns = ["ROC-AUC", "PR-AUC", "F1-Score", "False Positive Rate"]

    ax = plot_df.plot(kind="bar", figsize=(12, 6), colormap="viridis", width=0.8)
    plt.title("Comparative Performance Across Models & Key Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score / Rate", fontsize=11)
    plt.xlabel("Model Architecture", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, loc="upper right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate values
    for p in ax.patches:
        height = p.get_height()
        if height > 0.02:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                xytext=(0, 2),
                textcoords="offset points"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

