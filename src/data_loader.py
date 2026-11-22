"""
Data loading and validation module.
Responsible for loading the German Credit dataset, mapping the target variable,
validating types, and performing stratified train/test splitting.
"""

import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    POSITIVE_LABEL,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE
)


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    df = pd.read_csv(filepath)
    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Validates data, encodes the binary target, and performs stratified split.

    Returns:
        X_train, X_test, y_train, y_test, metadata
    """
    df = df.copy()

    # Target Encoding: 1 = Default (bad), 0 = Creditworthy (good)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data.")

    y = (df[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()

    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    metadata = {
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_default_rate": float(y_train.mean()),
        "test_default_rate": float(y_test.mean()),
        "num_features": NUMERICAL_FEATURES,
        "cat_features": CATEGORICAL_FEATURES
    }

    return X_train, X_test, y_train, y_test, metadata


if __name__ == "__main__":
    df = load_raw_data()
    X_tr, X_te, y_tr, y_te, meta = prepare_data(df)
    print("Dataset loaded successfully!")
    print(f"Total samples: {meta['n_total']}, Train: {meta['n_train']}, Test: {meta['n_test']}")
    print(f"Default rate: Train={meta['train_default_rate']:.2%}, Test={meta['test_default_rate']:.2%}")

