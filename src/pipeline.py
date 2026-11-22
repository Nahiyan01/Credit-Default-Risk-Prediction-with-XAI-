"""
End-to-End Orchestration Pipeline.
Executes data loading, preprocessing, model tuning, calibration,
evaluation, SHAP explainability analysis, and figure generation.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    RAW_DATA_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    AUTO_APPROVE_THRESHOLD,
    AUTO_DECLINE_THRESHOLD,
    RANDOM_STATE
)
from src.data_loader import load_raw_data, prepare_data
from src.preprocessing import CreditDataPreprocessor, apply_smote
from src.models import tune_and_train_models
from src.calibration import (
    calibrate_model,
    compute_calibration_metrics,
    triage_loan_applications,
    plot_calibration_curves
)
from src.evaluation import (
    evaluate_models,
    evaluate_shap_consistency,
    plot_roc_and_pr_curves,
    plot_metrics_comparison_bar
)
from src.explainability import CreditRiskExplainer


def run_pipeline():
    print("=" * 80)
    print("EXPLAINABLE MACHINE LEARNING FOR CREDIT DEFAULT RISK PREDICTION")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1/7] Loading and Splitting Dataset...")
    df_raw = load_raw_data(RAW_DATA_PATH)
    X_train_raw, X_test_raw, y_train, y_test, metadata = prepare_data(df_raw, test_size=0.20)
    print(f"Dataset Size: {metadata['n_total']} rows | Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")
    print(f"Default Prevalence: Train={metadata['train_default_rate']:.1%}, Test={metadata['test_default_rate']:.1%}")

    # 2. Preprocessing & Encoding
    print("\n[Step 2/7] Preprocessing & Feature Engineering...")
    preprocessor = CreditDataPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.feature_names_
    print(f"Transformed Feature Matrix Shape: Train={X_train.shape}, Test={X_test.shape}")
    print(f"Total Encoded Features: {len(feature_names)}")

    # 3. Model Tuning & Cross-Validation (Class-Weighted)
    print("\n[Step 3/7] Tuning & Training Models (Stratified 5-Fold CV)...")
    models_dict = tune_and_train_models(X_train, y_train, cv_splits=5, scoring="roc_auc")

    # Extract best estimators
    trained_estimators = {name: res["best_estimator"] for name, res in models_dict.items()}

    # Also evaluate SMOTE variant on best tree model architecture
    print("\n--> Evaluating SMOTE Resampling vs Class-Weighted Loss on XGBoost...")
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    from xgboost import XGBClassifier
    xgb_smote = XGBClassifier(
        **{k: v for k, v in models_dict["XGBoost"]["best_params"].items()},
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_smote.fit(X_train_smote, y_train_smote)
    trained_estimators["XGBoost_SMOTE"] = xgb_smote

    # 4. Probability Calibration (Platt Scaling)
    print("\n[Step 4/7] Applying Platt Scaling (Sigmoid Probability Calibration)...")
    calibrated_estimators = {}
    calibration_stats = {}

    for name, model in trained_estimators.items():
        cal_model = calibrate_model(model, X_train, y_train, method="sigmoid", cv=5)
        calibrated_estimators[name] = cal_model

        # Compute Brier score improvement on test set
        p_uncal = model.predict_proba(X_test)[:, 1]
        p_cal = cal_model.predict_proba(X_test)[:, 1]
        stats = compute_calibration_metrics(y_test.values, p_uncal, p_cal)
        calibration_stats[name] = stats
        print(f"    {name}: Brier Uncal={stats['brier_score_uncalibrated']:.4f} -> Cal={stats['brier_score_calibrated']:.4f} (Imp: {stats['improvement_pct']:+.1f}%)")

    # Select Champion Model based on ROC-AUC & Calibration
    champion_name = "XGBoost"
    champion_uncal = trained_estimators[champion_name]
    champion_cal = calibrated_estimators[champion_name]

    # Save Reliability Diagram
    plot_calibration_curves(
        champion_uncal,
        champion_cal,
        X_test,
        y_test,
        model_name=champion_name,
        save_path=str(FIGURES_DIR / "calibration_curves.png")
    )
    print(f"    Calibration plot saved to: {FIGURES_DIR / 'calibration_curves.png'}")

    # 5. Comprehensive Model Evaluation
    print("\n[Step 5/7] Evaluating Test-Set Metrics across all architectures...")
    # Evaluate calibrated versions for final operational metrics
    eval_calibrated = {f"{k} (Calibrated)": v for k, v in calibrated_estimators.items()}
    # Also include uncalibrated champion for comparison
    eval_calibrated[f"{champion_name} (Uncalibrated)"] = champion_uncal

    df_eval = evaluate_models(eval_calibrated, X_test, y_test, threshold=0.50)
    print("\nTest Evaluation Results Table:")
    print(df_eval[["ROC-AUC", "PR-AUC", "F1-Score", "False_Positive_Rate (Denying Good)", "Brier_Score"]].to_string())

    # Save plots
    plot_roc_and_pr_curves(
        trained_estimators,
        X_test,
        y_test,
        save_path=str(FIGURES_DIR / "roc_pr_curves.png")
    )
    plot_metrics_comparison_bar(
        df_eval.loc[[f"{k} (Calibrated)" for k in trained_estimators.keys()]],
        save_path=str(FIGURES_DIR / "model_comparison.png")
    )
    print(f"    ROC/PR and comparison charts saved to: {FIGURES_DIR}")

    # 6. Explainability Quality: SHAP Consistency across CV folds
    print("\n[Step 6/7] Computing Explainability Quality (SHAP Consistency across CV Splits)...")
    from xgboost import XGBClassifier
    mean_rho, imp_matrix, rhos = evaluate_shap_consistency(
        XGBClassifier,
        model_params={**models_dict["XGBoost"]["best_params"], "eval_metric": "logloss", "random_state": RANDOM_STATE},
        X=X_train,
        y=y_train,
        n_splits=5
    )
    print(f"    Explainability Quality Metric: Mean Spearman Rank Correlation = {mean_rho:.4f}")
    print(f"    Pairwise Fold Correlations: {[round(r, 3) for r in rhos]}")

    # Save SHAP stability plot
    plt.figure(figsize=(8, 4))
    plt.boxplot(rhos, vert=False, patch_artist=True, boxprops=dict(facecolor="#5bc0de"))
    plt.scatter(rhos, [1] * len(rhos), color="#0275d8", zorder=3, alpha=0.8)
    plt.title(f"Explainability Quality: SHAP Stability Across 5 Folds (Mean Rho = {mean_rho:.3f})", fontsize=11, fontweight="bold")
    plt.xlabel("Spearman Rank Correlation (Pairwise Folds)", fontsize=10)
    plt.yticks([])
    plt.xlim(0.70, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "shap_cross_val_stability.png"), dpi=300)
    plt.close()

    # 7. Model Explainability Engine (SHAP)
    print("\n[Step 7/7] Generating SHAP Global and Local Explanations...")
    explainer = CreditRiskExplainer(champion_uncal, feature_names=feature_names)
    explainer.compute_shap_values(X_test)

    # Global plots
    explainer.plot_global_summary_bar(str(FIGURES_DIR / "shap_summary_bar.png"), max_display=12)
    explainer.plot_global_beeswarm(str(FIGURES_DIR / "shap_beeswarm.png"), max_display=12)
    df_global_imp = explainer.get_global_feature_importance()
    print("\nTop 8 Global Macro Risk Drivers:")
    print(df_global_imp.head(8).to_string())

    # Triage and Case Selection for Local Explanations
    y_test_probs = champion_cal.predict_proba(X_test)[:, 1]
    df_triage = triage_loan_applications(y_test_probs)
    df_triage["true_label"] = y_test.values

    # Uncertainty Distribution Plot
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df_triage,
        x="default_probability",
        hue="triage_decision",
        palette={"Auto-Approve": "#28a745", "Human Underwriter Review": "#ffc107", "Auto-Decline": "#dc3545"},
        bins=20,
        multiple="stack"
    )
    plt.axvline(AUTO_APPROVE_THRESHOLD, color="green", linestyle="--", label=f"Approve Cutoff ({AUTO_APPROVE_THRESHOLD})")
    plt.axvline(AUTO_DECLINE_THRESHOLD, color="red", linestyle="--", label=f"Decline Cutoff ({AUTO_DECLINE_THRESHOLD})")
    plt.title("Applicant Risk Distribution & Multi-Tier Underwriting Triage", fontsize=12, fontweight="bold")
    plt.xlabel("Calibrated Default Probability P(Default)", fontsize=10)
    plt.ylabel("Number of Applicants", fontsize=10)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "uncertainty_distribution.png"), dpi=300)
    plt.close()

    # Select 3 representative applicants:
    # 1. Approved (low risk)
    approved_indices = np.where(y_test_probs < 0.20)[0]
    idx_approved = int(approved_indices[0]) if len(approved_indices) > 0 else 0

    # 2. Declined (high risk)
    declined_indices = np.where(y_test_probs > 0.70)[0]
    idx_declined = int(declined_indices[0]) if len(declined_indices) > 0 else 1

    # 3. Uncertain / Borderline (Underwriter Review)
    uncertain_indices = np.where((y_test_probs >= 0.40) & (y_test_probs <= 0.60))[0]
    idx_uncertain = int(uncertain_indices[0]) if len(uncertain_indices) > 0 else 2

    print(f"\nRepresentative Applicant Cases for Local Audits:")
    print(f"  Approved Applicant (Idx {idx_approved}): P(Default) = {y_test_probs[idx_approved]:.1%}")
    print(f"  Declined Applicant (Idx {idx_declined}): P(Default) = {y_test_probs[idx_declined]:.1%}")
    print(f"  Borderline Applicant (Idx {idx_uncertain}): P(Default) = {y_test_probs[idx_uncertain]:.1%}")

    # Generate Waterfall Plots
    explainer.plot_local_waterfall(
        idx_approved,
        str(FIGURES_DIR / "shap_waterfall_applicant_approved.png"),
        applicant_label=f"Approved Case (P={y_test_probs[idx_approved]:.1%})"
    )
    explainer.plot_local_waterfall(
        idx_declined,
        str(FIGURES_DIR / "shap_waterfall_applicant_rejected.png"),
        applicant_label=f"Declined Case (P={y_test_probs[idx_declined]:.1%})"
    )
    explainer.plot_local_waterfall(
        idx_uncertain,
        str(FIGURES_DIR / "shap_waterfall_applicant_uncertain.png"),
        applicant_label=f"Underwriter Review Case (P={y_test_probs[idx_uncertain]:.1%})"
    )

    # Generate Adverse Action Notices for Declined & Uncertain Cases
    declined_notice = explainer.generate_adverse_action_notice(
        idx_declined,
        X_test.iloc[idx_declined],
        calibrated_prob=y_test_probs[idx_declined]
    )
    uncertain_notice = explainer.generate_adverse_action_notice(
        idx_uncertain,
        X_test.iloc[idx_uncertain],
        calibrated_prob=y_test_probs[idx_uncertain]
    )

    with open(REPORTS_DIR / "adverse_action_sample.json", "w") as f:
        json.dump({"declined_applicant": declined_notice, "borderline_applicant": uncertain_notice}, f, indent=2)

    # Save test predictions DataFrame
    df_test_out = X_test_raw.copy()
    df_test_out["true_class"] = y_test.values
    df_test_out["calibrated_default_prob"] = y_test_probs
    df_test_out["triage_decision"] = df_triage["triage_decision"].values
    df_test_out.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    # Save summary evaluation metrics to CSV
    df_eval.to_csv(REPORTS_DIR / "evaluation_metrics_summary.csv")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Reports & predictions saved to: {REPORTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()

