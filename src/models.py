"""
Model definitions, hyperparameter tuning, and cross-validation module.
Defines Baseline Logistic Regression, Random Forest, XGBoost, and LightGBM models,
with support for class weighting and SMOTE training.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from src.config import RANDOM_STATE
from src.preprocessing import calculate_scale_pos_weight


def get_model_definitions(y_train: np.ndarray) -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """
    Returns dictionary of models and their respective hyperparameter search grids.
    Incorporates class-weighted loss to counter target imbalance.
    """
    scale_pos = calculate_scale_pos_weight(y_train)

    models_and_grids = {
        "Logistic_Regression": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            {
                "C": [0.01, 0.1, 0.5, 1.0, 2.0, 10.0],
                "solver": ["lbfgs"]
            }
        ),
        "Random_Forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 8, 12, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        ),
        "XGBoost": (
            XGBClassifier(
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 4, 6],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        ),
        "LightGBM": (
            LGBMClassifier(
                scale_pos_weight=scale_pos,
                random_state=RANDOM_STATE,
                verbosity=-1,
                n_jobs=-1
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "subsample": [0.8, 1.0]
            }
        )
    }

    return models_and_grids


def tune_and_train_models(
    X_train,
    y_train,
    cv_splits: int = 5,
    scoring: str = "roc_auc"
) -> Dict[str, Dict[str, Any]]:
    """
    Executes GridSearchCV across all candidate architectures using Stratified K-Fold CV.

    Returns:
        Dictionary mapping model_name -> {
            'best_estimator': fitted_model,
            'best_params': dict,
            'best_cv_score': float,
            'cv_results': dict
        }
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    models_and_grids = get_model_definitions(y_train)
    results = {}

    for name, (base_model, param_grid) in models_and_grids.items():
        print(f"--> Tuning and training: {name} (scoring={scoring})...")
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_train, y_train)

        results[name] = {
            "best_estimator": grid.best_estimator_,
            "best_params": grid.best_params_,
            "best_cv_score": float(grid.best_score_),
            "cv_results": grid.cv_results_
        }
        print(f"    Completed {name}: Best CV {scoring.upper()} = {grid.best_score_:.4f}")

    return results
