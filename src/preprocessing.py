"""
Preprocessing and Imbalance Handling Module.
Constructs scikit-learn preprocessing transformers (imputation, scaling, one-hot encoding)
and handles class imbalance using SMOTE and class weighting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, RANDOM_STATE


class CreditDataPreprocessor:
    """
    Transforms raw tabular applicant data into standardized, encoded numeric representations
    suitable for linear models and tree ensembles, preserving interpretable column names for SHAP.
    """

    def __init__(
        self,
        numerical_cols: List[str] = NUMERICAL_FEATURES,
        categorical_cols: List[str] = CATEGORICAL_FEATURES,
        scale_numeric: bool = True
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scale_numeric = scale_numeric
        self.transformer: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []

    def build_transformer(self) -> ColumnTransformer:
        """Constructs the ColumnTransformer pipeline."""
        # Numerical pipeline
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(num_steps)

        # Categorical pipeline
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numerical_cols),
                ("cat", cat_pipeline, self.categorical_cols)
            ],
            remainder="drop"
        )
        return transformer

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditDataPreprocessor":
        """Fits the transformers on training data."""
        self.transformer = self.build_transformer()
        self.transformer.fit(X, y)

        # Extract feature names
        num_feature_names = self.numerical_cols
        cat_encoder = self.transformer.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_cols).tolist()

        # Clean and sanitize feature names for XGBoost/LightGBM compatibility (no [, ], <, >, /)
        def sanitize_name(col: str) -> str:
            clean = col.replace("<=", "_le_").replace(">=", "_ge_")
            clean = clean.replace("<", "_lt_").replace(">", "_gt_")
            clean = clean.replace("[", "_").replace("]", "_")
            clean = clean.replace("/", "_").replace(" ", "_")
            clean = clean.replace("-", "_").replace(".", "_")
            # Remove any double underscores
            while "__" in clean:
                clean = clean.replace("__", "_")
            return clean.strip("_")

        cleaned_cat_names = [sanitize_name(f) for f in cat_feature_names]
        self.feature_names_ = num_feature_names + cleaned_cat_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms data and returns a DataFrame with labeled columns."""
        if self.transformer is None:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")
        
        arr = self.transformer.transform(X)
        df_out = pd.DataFrame(arr, columns=self.feature_names_, index=X.index)
        return df_out

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fits and transforms training data."""
        return self.fit(X, y).transform(X)


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: float = 1.0,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) on processed training data.

    Args:
        X: Processed features DataFrame
        y: Binary target Series
        sampling_strategy: Ratio of minority to majority class after resampling (1.0 = balanced 50:50)
        random_state: Seed for reproducibility

    Returns:
        X_resampled, y_resampled
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name=y.name)
    return X_res, y_res


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """
    Computes scale_pos_weight for gradient boosted trees (negative / positive ratio).
    E.g. for 700 good and 300 bad, scale_pos_weight = 700 / 300 ~ 2.333.
    """
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    return float(num_neg / max(1, num_pos))
