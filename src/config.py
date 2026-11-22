"""
Configuration module for Credit Default Risk Prediction project.
Defines dataset paths, feature groupings, random seeds, and regulatory reason mappings.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR)

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "german_credit_data.csv"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for p in [DATA_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Target Definition: 1 = Default (bad), 0 = Non-default (good)
TARGET_COLUMN = "class"
POSITIVE_LABEL = "bad"
NEGATIVE_LABEL = "good"

# Feature Categorization
NUMERICAL_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents"
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker"
]

# Triage Thresholds for Calibration & Uncertainty
AUTO_APPROVE_THRESHOLD = 0.30       # Calibrated P(Default) < 0.30 -> Auto-Approve
AUTO_DECLINE_THRESHOLD = 0.65       # Calibrated P(Default) > 0.65 -> Auto-Decline
# 0.30 <= P(Default) <= 0.65 -> High Uncertainty: Triage to Human Underwriter

# Adverse Action Reason Mapping (Compliant with FCRA / ECOA regulations)
# Maps one-hot encoded or raw feature patterns to clear human-readable disclosures.
FEATURE_ADVERSE_ACTION_MAP = {
    "checking_status": "Status of checking account is low, overdrawn, or absent.",
    "duration": "Duration of requested loan term is excessively long relative to risk profile.",
    "credit_history": "Adverse credit history or past delinquency on existing obligations.",
    "credit_amount": "Requested credit amount is high relative to borrower repayment capacity.",
    "savings_status": "Insufficient liquid reserves or low balance in savings/investment accounts.",
    "employment": "Short tenure with present employer or unstable employment duration.",
    "installment_commitment": "High installment rate relative to disposable income.",
    "property_magnitude": "Lack of collateral or unverified property ownership.",
    "age": "Limited credit longevity or insufficient financial seasoning for requested tier.",
    "other_payment_plans": "Existing installment obligations at banks or retail stores.",
    "housing": "Unstable or high-expense residential living arrangement.",
    "existing_credits": "Multiple active credit lines leading to over-leverage risk.",
    "other_parties": "Absence of creditworthy co-applicant or guarantor.",
    "job": "Employment status classification indicates elevated income variance."
}
