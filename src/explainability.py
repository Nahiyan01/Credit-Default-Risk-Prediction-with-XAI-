"""
Model Explainability Module using SHAP (SHapley Additive exPlanations).
Generates global feature importance plots, beeswarm distribution charts,
local waterfall explanation plots, and FCRA-compliant adverse action reasons.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import FEATURE_ADVERSE_ACTION_MAP


class CreditRiskExplainer:
    """
    Wraps SHAP TreeExplainer for credit risk decision audits,
    providing global macro insights and local applicant-level explanations.
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self.shap_values_raw = None
        self.shap_values_pos = None  # Class 1 (Default risk)
        self.expected_value_pos = None

    def compute_shap_values(self, X: pd.DataFrame):
        """Computes SHAP values on dataset X."""
        self.shap_values_raw = self.explainer(X)

        # Handle different SHAP output formats across model types / versions
        if len(self.shap_values_raw.shape) == 3:
            # Multi-class output (samples, features, classes) -> extract class 1 (default)
            self.shap_values_pos = self.shap_values_raw[:, :, 1]
            self.expected_value_pos = self.explainer.expected_value[1]
        else:
            self.shap_values_pos = self.shap_values_raw
            ev = self.explainer.expected_value
            self.expected_value_pos = ev[1] if isinstance(ev, (list, np.ndarray)) else ev

        return self.shap_values_pos

    def get_global_feature_importance(self) -> pd.DataFrame:
        """
        Returns a sorted DataFrame of global mean absolute SHAP values per feature.
        """
        if self.shap_values_pos is None:
            raise RuntimeError("SHAP values have not been computed yet. Call compute_shap_values().")

        mean_abs = np.mean(np.abs(self.shap_values_pos.values), axis=0)
        df_importance = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        return df_importance

    def plot_global_summary_bar(self, save_path: str, max_display: int = 15):
        """Plots global feature importance bar chart."""
        plt.figure(figsize=(10, 7))
        shap.plots.bar(self.shap_values_pos, max_display=max_display, show=False)
        plt.title("Global Macro Default Risk Drivers (Mean |SHAP Value|)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_global_beeswarm(self, save_path: str, max_display: int = 15):
        """Plots beeswarm distribution chart showing feature value impact on default risk."""
        plt.figure(figsize=(11, 8))
        shap.plots.beeswarm(self.shap_values_pos, max_display=max_display, show=False)
        plt.title("SHAP Beeswarm: Feature Value Impact on Credit Default Risk", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_local_waterfall(
        self,
        applicant_idx: int,
        save_path: str,
        applicant_label: str = "",
        max_display: int = 10
    ):
        """
        Generates and saves a local waterfall plot explaining a single loan applicant.
        """
        plt.figure(figsize=(10, 6))
        explanation_item = self.shap_values_pos[applicant_idx]
        shap.plots.waterfall(explanation_item, max_display=max_display, show=False)
        title_suffix = f" - {applicant_label}" if applicant_label else f" (Applicant #{applicant_idx})"
        plt.title(f"Underwriting Rationale: SHAP Waterfall Plot{title_suffix}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def generate_adverse_action_notice(
        self,
        applicant_idx: int,
        applicant_features: pd.Series,
        calibrated_prob: float,
        top_n: int = 4
    ) -> Dict[str, Any]:
        """
        Generates a regulatory-compliant Adverse Action Notice for a declined applicant
        by extracting the top positive SHAP contributors pushing default log-odds upwards.
        """
        shap_vals = self.shap_values_pos[applicant_idx].values
        feat_names = self.feature_names

        # Filter features that increased default risk (SHAP > 0)
        risk_drivers = []
        for name, val, f_val in zip(feat_names, shap_vals, applicant_features.values):
            if val > 0:
                risk_drivers.append((name, float(val), f_val))

        # Sort by largest positive SHAP impact
        risk_drivers.sort(key=lambda x: x[1], reverse=True)
        top_factors = risk_drivers[:top_n]

        disclosed_reasons = []
        for feat, shap_val, feat_val in top_factors:
            # Map root feature to regulatory disclosure notice
            root_feat = feat.split("_")[0]
            for mapped_key, disclosure in FEATURE_ADVERSE_ACTION_MAP.items():
                if mapped_key in feat:
                    root_feat = mapped_key
                    break
            
            reason_text = FEATURE_ADVERSE_ACTION_MAP.get(
                root_feat,
                f"Adverse indicator associated with factor '{feat}'."
            )
            disclosed_reasons.append({
                "factor_code": feat,
                "feature_value": float(feat_val) if isinstance(feat_val, (int, float, np.number)) else str(feat_val),
                "shap_impact": float(shap_val),
                "regulatory_disclosure_statement": reason_text
            })

        notice = {
            "applicant_id": int(applicant_idx),
            "calibrated_default_risk": float(calibrated_prob),
            "underwriting_disposition": "DECLINED" if calibrated_prob >= 0.50 else "UNDERWRITER REVIEW",
            "top_adverse_reasons": disclosed_reasons
        }
        return notice

