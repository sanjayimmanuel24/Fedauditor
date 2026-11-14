import os
import secrets

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - listed in requirements.txt
    load_dotenv = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load local environment overrides before reading any secret or port.
if load_dotenv is not None:
    load_dotenv(os.path.join(BASE_DIR, ".env"))

DATA_DIR = os.path.join(BASE_DIR, "data")

# Primary dataset: the public PIMA Indians Diabetes dataset (768 records).
# If this file is missing the loader falls back to a documented synthetic
# generator so the application still boots.
DATA_PATH = os.path.join(DATA_DIR, "pima_diabetes.csv")

# Second public dataset used for external validation. It is a different
# questionnaire (binary symptom items + age + sex), not a re-split of PIMA.
# The file is downloaded from the UCI Machine Learning Repository; the loader
# refuses to fall back to synthetic data for this dataset.
UCI_EARLY_PATH = os.path.join(DATA_DIR, "uci_early_diabetes.csv")

# Legacy artefacts produced by notebooks/patients.ipynb live in
# notebooks/artifacts/. The federated engine no longer reads them.

# Database
DB_PATH = os.path.join(BASE_DIR, "users.db")

# Secret key for sessions. Set FEDHEALTH_SECRET_KEY in .env for stable sessions;
# otherwise a process-local random key is generated (logins reset on restart).
SECRET_KEY = os.environ.get("FEDHEALTH_SECRET_KEY") or secrets.token_hex(32)

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# Columns where a recorded 0 is physiologically impossible and is treated as
# missing (same convention as the exploratory notebook).
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood Pressure (mmHg)",
    "SkinThickness": "Skin Thickness (mm)",
    "Insulin": "Insulin (uU/mL)",
    "BMI": "BMI (kg/m2)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age (years)",
}

# UCI "Early Stage Diabetes Risk Prediction" features. Every symptom item is
# binary (Yes/No) and is encoded as 1/0; Gender is encoded Male=1, Female=0.
UCI_FEATURES = [
    "Age",
    "Gender",
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
]

UCI_FEATURE_LABELS = {
    "Age": "Age (years)",
    "Gender": "Gender (Male=1, Female=0)",
    "Polyuria": "Polyuria",
    "Polydipsia": "Polydipsia",
    "sudden weight loss": "Sudden weight loss",
    "weakness": "Weakness",
    "Polyphagia": "Polyphagia",
    "Genital thrush": "Genital thrush",
    "visual blurring": "Visual blurring",
    "Itching": "Itching",
    "Irritability": "Irritability",
    "delayed healing": "Delayed healing",
    "partial paresis": "Partial paresis",
    "muscle stiffness": "Muscle stiffness",
    "Alopecia": "Alopecia",
    "Obesity": "Obesity",
}

AGE_BANDS = [
    ("<30", 0, 30),
    ("30-45", 30, 45),
    ("45+", 45, 200),
]

# UCI's age distribution is shifted upward: fewer than 4% of records are under
# 30, which would leave the youngest cohort too small for a local calibration
# split. Its bands are therefore shifted so each client has enough rows and
# both classes.
UCI_AGE_BANDS = [
    ("<35", 0, 35),
    ("35-50", 35, 50),
    ("50+", 50, 200),
]

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Every dataset is normalised to the columns ``features + ["Outcome"]`` plus a
# derived ``AgeBand`` column. Only datasets with an ``Age`` column can use the
# age-cohort partition; both of the current datasets have one.
DATASETS = {
    "pima": {
        "name": "PIMA Indians Diabetes",
        "short_name": "PIMA",
        "path": DATA_PATH,
        "source_target": TARGET,
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "zero_as_missing": ZERO_AS_MISSING,
        "age_bands": AGE_BANDS,
        "allow_synthetic": True,
        "citation": "Smith et al., PIMA Indians Diabetes Database (UCI).",
    },
    "uci_early": {
        "name": "UCI Early Stage Diabetes Risk Prediction",
        "short_name": "UCI Early Stage",
        "path": UCI_EARLY_PATH,
        "source_target": "class",
        "features": UCI_FEATURES,
        "feature_labels": UCI_FEATURE_LABELS,
        "zero_as_missing": [],
        "age_bands": UCI_AGE_BANDS,
        "allow_synthetic": False,
        "binary_columns": [
            "Polyuria",
            "Polydipsia",
            "sudden weight loss",
            "weakness",
            "Polyphagia",
            "Genital thrush",
            "visual blurring",
            "Itching",
            "Irritability",
            "delayed healing",
            "partial paresis",
            "muscle stiffness",
            "Alopecia",
            "Obesity",
        ],
        "gender_column": "Gender",
        "citation": (
            "S. Islam et al., Early Stage Diabetes Risk Prediction Dataset "
            "(UCI Machine Learning Repository, 2020)."
        ),
    },
}

DEFAULT_DATASET = "pima"
SECOND_DATASET = "uci_early"

# ---------------------------------------------------------------------------
# Federated simulation
# ---------------------------------------------------------------------------
RANDOM_SEED = 20
TEST_SIZE = 0.2
N_CLIENTS = 3

# "age"  -> clinically motivated non-IID split: each simulated client serves a
#           distinct age cohort (young / mid / older). This is what creates the
#           calibration and fairness effects the project studies.
# "iid"  -> random shards (kept so the two regimes can be compared).
PARTITION_MODE = "age"

# Fraction of each client's data held out for local calibration / fairness
# statistics. Never shared; only aggregate counts leave the client.
CLIENT_CALIBRATION_FRACTION = 0.25

# Local optimiser
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.05
BATCH_SIZE = 32
L2_PENALTY = 1e-4

# Round budget used for client-valuation coalitions
VALUATION_ROUNDS = 15
DEFAULT_WARMUP_ROUNDS = 5

# How client updates are weighted during aggregation.
#   "sample"   -> weight by local sample count (standard FedAvg)
#   "uniform"  -> every client counts equally regardless of size
#   "positive" -> weight by the number of positive (diabetic) cases a client
#                 contributes, which shifts influence toward the high-prevalence
#                 cohort. This is the prevalence-aware variant.
AGGREGATION_WEIGHTING = "sample"
WEIGHTING_MODES = ("sample", "uniform", "positive")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
DECISION_THRESHOLD = 0.5
CALIBRATION_BINS = 10
FAIRNESS_THRESHOLD_STEPS = 19  # grid of candidate thresholds in [0.05, 0.95]
BOOTSTRAP_SAMPLES = 300
