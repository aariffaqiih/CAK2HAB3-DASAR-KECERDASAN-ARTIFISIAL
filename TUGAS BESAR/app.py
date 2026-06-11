import os
import json
from pathlib import Path
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from fuzzy_engine import (
    MF_CLASS,
    MF_OUTPUT,
    MF_SERVICE,
    MF_TRAVEL,
    RULES,
    SUGENO_OUTPUT,
    UNIVERSE_OUTPUT,
    calculate_fuzzy_scores,
    class_mapping,
    eval_mf,
    fuzzify,
    travel_mapping,
    trimf,
)


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# =========================================================
# Page Config
# =========================================================

st.set_page_config(
    page_title="Airline Satisfaction Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Optional UI Libraries
# =========================================================

try:
    from streamlit_extras.stylable_container import stylable_container
except Exception:
    @contextmanager
    def stylable_container(key=None, css_styles=None):
        yield

try:
    from streamlit_elements import elements, mui
    # Use the native HTML pipeline because the component frontend can fail in local browser proxies.
    ELEMENTS_AVAILABLE = False
except Exception:
    ELEMENTS_AVAILABLE = False


# =========================================================
# Global Styling
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600&display=swap');

    :root {
        --bg: #f7f9fc;
        --surface: rgba(255,255,255,0.88);
        --surface-solid: #ffffff;
        --surface-soft: #f8fafc;
        --text: #0f172a;
        --muted: #64748b;
        --line: #e5edf7;
        --primary: #2563eb;
        --primary-2: #0ea5e9;
        --success: #16a34a;
        --warning: #d97706;
        --danger: #dc2626;
        --shadow: 0 24px 70px rgba(15, 23, 42, 0.08);
        --shadow-sm: 0 12px 32px rgba(15, 23, 42, 0.06);
        --radius-xl: 28px;
        --radius-lg: 20px;
        --radius-md: 14px;
    }

    html, body, [class*="css"] {
        font-family: 'Google Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.13), transparent 34rem),
            radial-gradient(circle at 86% 6%, rgba(14, 165, 233, 0.10), transparent 30rem),
            linear-gradient(180deg, #f8fbff 0%, #f7f9fc 44%, #ffffff 100%);
    }

    .block-container {
        padding-top: 2.1rem !important;
        padding-bottom: 3rem !important;
        max-width: 1240px !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.76);
        backdrop-filter: blur(22px);
        border-right: 1px solid rgba(226, 232, 240, 0.84);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.35rem;
    }

    h1, h2, h3, h4 {
        letter-spacing: -0.04em;
        color: var(--text);
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(226, 232, 240, 0.95);
        border-radius: 34px;
        padding: 2.35rem 2.35rem 2rem;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.96), rgba(248,250,252,0.88)),
            radial-gradient(circle at top right, rgba(37,99,235,0.16), transparent 36%);
        box-shadow: var(--shadow);
    }

    .hero-card::after {
        content: "";
        position: absolute;
        width: 340px;
        height: 340px;
        top: -160px;
        right: -130px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(14,165,233,0.21), rgba(37,99,235,0.04), transparent 68%);
        pointer-events: none;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.42rem 0.78rem;
        border: 1px solid rgba(37, 99, 235, 0.16);
        background: rgba(239, 246, 255, 0.92);
        color: #1d4ed8;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 1rem 0 0.72rem;
        font-size: clamp(2.15rem, 4vw, 4.2rem);
        line-height: 0.96;
        font-weight: 600;
        max-width: 880px;
        color: #0f172a;
    }

    .hero-subtitle {
        font-size: 1.03rem;
        line-height: 1.7;
        max-width: 780px;
        color: var(--muted);
        margin: 0.25rem 0 0;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.82rem;
        margin-top: 1.55rem;
    }

    .mini-pill {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        border: 1px solid rgba(226, 232, 240, 0.95);
        background: rgba(255,255,255,0.78);
        border-radius: 999px;
        padding: 0.72rem 0.88rem;
        color: #334155;
        font-weight: 500;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.04);
    }

    .glass-card {
        border: 1px solid rgba(226, 232, 240, 0.92);
        background: rgba(255,255,255,0.78);
        backdrop-filter: blur(18px);
        border-radius: var(--radius-xl);
        padding: 1.35rem;
        box-shadow: var(--shadow-sm);
    }

    .soft-card {
        border: 1px solid rgba(226, 232, 240, 0.92);
        background: var(--surface-solid);
        border-radius: var(--radius-lg);
        padding: 1.15rem 1.18rem;
        box-shadow: 0 12px 36px rgba(15,23,42,0.045);
    }

    .section-label {
        display: flex;
        align-items: center;
        gap: 0.62rem;
        margin: 0.25rem 0 0.9rem;
        font-size: 1.12rem;
        font-weight: 600;
        letter-spacing: -0.025em;
        color: var(--text);
    }

    .section-label span {
        display: inline-flex;
        width: 34px;
        height: 34px;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: #eff6ff;
        color: var(--primary);
    }

    .metric-card {
        position: relative;
        overflow: hidden;
        min-height: 136px;
        border: 1px solid rgba(226, 232, 240, 0.94);
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.9));
        border-radius: 26px;
        padding: 1.08rem 1.12rem;
        box-shadow: 0 16px 38px rgba(15,23,42,0.06);
    }

    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2563eb, #06b6d4);
    }

    .metric-kicker {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 2.02rem;
        font-weight: 600;
        letter-spacing: -0.045em;
        line-height: 1;
        margin-bottom: 0.5rem;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .result-banner {
        border-radius: 30px;
        padding: 1.35rem 1.45rem;
        border: 1px solid rgba(226, 232, 240, 0.92);
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(240,249,255,0.88));
        box-shadow: var(--shadow-sm);
        margin: 1rem 0 1.2rem;
    }

    .result-title {
        font-size: 0.86rem;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.42rem;
    }

    .result-main {
        font-size: clamp(1.55rem, 3vw, 2.8rem);
        font-weight: 600;
        letter-spacing: -0.04em;
        color: var(--text);
        margin: 0;
    }

    .result-sub {
        color: var(--muted);
        margin-top: 0.5rem;
        font-size: 0.98rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.32rem 0.66rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.78rem;
        border: 1px solid transparent;
        white-space: nowrap;
    }

    .badge-success {
        color: #166534;
        background: #dcfce7;
        border-color: #bbf7d0;
    }

    .badge-warning {
        color: #92400e;
        background: #fef3c7;
        border-color: #fde68a;
    }

    .badge-blue {
        color: #1d4ed8;
        background: #dbeafe;
        border-color: #bfdbfe;
    }

    .badge-neutral {
        color: #475569;
        background: #f1f5f9;
        border-color: #e2e8f0;
    }

    .pretty-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 0.52rem;
        font-size: 0.92rem;
    }

    .pretty-table th {
        padding: 0.7rem 0.85rem;
        color: #64748b;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.055em;
        font-weight: 600;
        text-align: left;
    }

    .pretty-table td {
        padding: 0.9rem 0.85rem;
        border-top: 1px solid #e8eef7;
        border-bottom: 1px solid #e8eef7;
        background: rgba(255,255,255,0.86);
        color: #1e293b;
    }

    .pretty-table tr td:first-child {
        border-left: 1px solid #e8eef7;
        border-radius: 16px 0 0 16px;
    }

    .pretty-table tr td:last-child {
        border-right: 1px solid #e8eef7;
        border-radius: 0 16px 16px 0;
    }

    .pretty-table tbody tr:hover td {
        background: #f8fbff;
        border-color: #dbeafe;
    }

    .flow-wrap {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 1rem 0 0.2rem;
    }

    .flow-step {
        border: 1px solid rgba(226,232,240,0.92);
        border-radius: 22px;
        background: rgba(255,255,255,0.82);
        padding: 1rem;
        min-height: 104px;
        box-shadow: 0 12px 28px rgba(15,23,42,0.045);
    }

    .flow-num {
        display: inline-flex;
        width: 28px;
        height: 28px;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: #eff6ff;
        color: #2563eb;
        font-weight: 600;
        margin-bottom: 0.62rem;
    }

    .flow-title {
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.24rem;
    }

    .flow-desc {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.42;
    }

    .sidebar-card {
        padding: 0.9rem 0.25rem 0.3rem;
    }

    .sidebar-title {
        font-size: 1.28rem;
        font-weight: 600;
        letter-spacing: -0.03em;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }

    .sidebar-subtitle {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.45;
        margin-bottom: 1rem;
    }

    div.stButton > button:first-child {
        border: none !important;
        border-radius: 999px !important;
        background: linear-gradient(135deg, #2563eb, #0ea5e9) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.72rem 1.15rem !important;
        box-shadow: 0 14px 32px rgba(37,99,235,0.28) !important;
        transition: all 160ms ease !important;
        width: 100%;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 38px rgba(37,99,235,0.32) !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.74);
        border: 1px solid rgba(226,232,240,0.9);
        padding: 1rem;
        border-radius: 20px;
        box-shadow: 0 12px 28px rgba(15,23,42,0.045);
    }

    [data-testid="stMetricLabel"] p {
        color: #64748b !important;
        font-weight: 600 !important;
    }

    [data-testid="stTabs"] button {
        font-weight: 600;
        color: #475569;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(226,232,240,0.92) !important;
        border-radius: 20px !important;
        background: rgba(255,255,255,0.72) !important;
        box-shadow: 0 12px 26px rgba(15,23,42,0.035);
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #dbeafe, transparent);
        margin: 1.3rem 0;
    }

    @media (max-width: 980px) {
        .hero-grid, .flow-wrap {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 620px) {
        .hero-card {
            padding: 1.45rem;
            border-radius: 26px;
        }
        .hero-grid, .flow-wrap {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Model Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction"
ML_MODEL_DIRS = [
    BASE_DIR / "exported_ml_model",
    BASE_DIR / "exported_model",
    BASE_DIR,
]
DL_MODEL_DIR = BASE_DIR / "exported_dl_model"
DL_RESOURCE_DIRS = [
    DL_MODEL_DIR,
    BASE_DIR / "exported_ml_model",
    BASE_DIR / "exported_model",
    BASE_DIR,
]

FEATURES_ORIGINAL_DEFAULT = [
    "Type of Travel",
    "Class",
    "Inflight wifi service",
    "Online boarding",
    "Inflight entertainment",
]

FEATURES_HYBRID_DEFAULT = [
    "Type of Travel",
    "Class",
    "Inflight wifi service",
    "Online boarding",
    "Inflight entertainment",
    "mamdani_score",
    "sugeno_score",
]

MODEL_CONFIGS = {
    "Logistic Regression Without Fuzzy": {
        "display": "Logistic Regression",
        "file": "logistic_regression_without_fuzzy.pkl",
        "input_type": "original",
    },
    "Logistic Regression With Fuzzy": {
        "display": "Logistic Regression",
        "file": "logistic_regression_with_fuzzy.pkl",
        "input_type": "hybrid",
    },
    "Decision Tree Without Fuzzy": {
        "display": "Decision Tree",
        "file": "decision_tree_without_fuzzy.pkl",
        "input_type": "original",
    },
    "Decision Tree With Fuzzy": {
        "display": "Decision Tree",
        "file": "decision_tree_with_fuzzy.pkl",
        "input_type": "hybrid",
    },
    "Random Forest Without Fuzzy": {
        "display": "Random Forest",
        "file": "random_forest_without_fuzzy.pkl",
        "input_type": "original",
    },
    "Random Forest With Fuzzy": {
        "display": "Random Forest",
        "file": "random_forest_with_fuzzy.pkl",
        "input_type": "hybrid",
    },
    "XGBoost Without Fuzzy": {
        "display": "XGBoost",
        "file": "xgboost_without_fuzzy.pkl",
        "input_type": "original",
    },
    "XGBoost With Fuzzy": {
        "display": "XGBoost",
        "file": "xgboost_with_fuzzy.pkl",
        "input_type": "hybrid",
    },
}


DL_MODEL_CONFIGS = {
    "MLP Without Fuzzy": {
        "display": "MLP Deep Learning",
        "model_file": "mlp_without_fuzzy.keras",
        "preprocessor_file": "mlp_without_fuzzy_preprocessor.pkl",
        "input_type": "original",
    },
    "MLP With Fuzzy": {
        "display": "MLP Deep Learning",
        "model_file": "mlp_with_fuzzy.keras",
        "preprocessor_file": "mlp_with_fuzzy_preprocessor.pkl",
        "input_type": "hybrid",
    },
}


def resolve_existing_path(filename: str, search_dirs: list[Path]) -> Path:
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate

    return search_dirs[0] / filename


def resolve_ml_model_path(filename: str) -> Path:
    return resolve_existing_path(filename, ML_MODEL_DIRS)


def resolve_dl_resource_path(filename: str) -> Path:
    return resolve_existing_path(filename, DL_RESOURCE_DIRS)


def format_required_path(filename: str, search_dirs: list[Path]) -> str:
    locations = [str(directory / filename) for directory in search_dirs]
    return " atau ".join(locations)


def import_tensorflow():
    try:
        import tensorflow as tf

        return tf, None
    except Exception as exc:
        return None, str(exc)



@st.cache_resource
def load_resources():
    shared_resource_dirs = [
        BASE_DIR / "exported_ml_model",
        DL_MODEL_DIR,
        BASE_DIR / "exported_model",
        ROOT_DIR,
    ]
    label_path = resolve_existing_path("label_encoder.pkl", shared_resource_dirs)
    original_features_path = resolve_existing_path("features_original.pkl", shared_resource_dirs)
    hybrid_features_path = resolve_existing_path("features_hybrid.pkl", shared_resource_dirs)
    dl_info_path = resolve_dl_resource_path("dl_model_info.json")

    missing_files = []

    if not label_path.exists():
        missing_files.append(format_required_path("label_encoder.pkl", shared_resource_dirs))

    label_encoder_loaded = None
    if label_path.exists():
        label_encoder_loaded = joblib.load(label_path)

    if original_features_path.exists():
        features_original_loaded = joblib.load(original_features_path)
    else:
        features_original_loaded = FEATURES_ORIGINAL_DEFAULT

    if hybrid_features_path.exists():
        features_hybrid_loaded = joblib.load(hybrid_features_path)
    else:
        features_hybrid_loaded = FEATURES_HYBRID_DEFAULT

    dl_model_info_loaded = {}
    if dl_info_path.exists():
        with open(dl_info_path, "r", encoding="utf-8") as file:
            dl_model_info_loaded = json.load(file)

    models_loaded = {}

    for model_name, config in MODEL_CONFIGS.items():
        model_path = resolve_ml_model_path(config["file"])

        if model_path.exists():
            models_loaded[model_name] = joblib.load(model_path)
        else:
            missing_files.append(format_required_path(config["file"], ML_MODEL_DIRS))

    dl_models_loaded = {}
    tf, tf_error = import_tensorflow()

    if tf is None:
        missing_files.append(f"TensorFlow/Keras dependency: {tf_error}")
    else:
        for model_name, config in DL_MODEL_CONFIGS.items():
            model_path = resolve_dl_resource_path(config["model_file"])
            preprocessor_path = resolve_dl_resource_path(config["preprocessor_file"])
            model_exists = model_path.exists()
            preprocessor_exists = preprocessor_path.exists()

            if not model_exists:
                missing_files.append(format_required_path(config["model_file"], DL_RESOURCE_DIRS))

            if not preprocessor_exists:
                missing_files.append(format_required_path(config["preprocessor_file"], DL_RESOURCE_DIRS))

            if model_exists and preprocessor_exists:
                dl_models_loaded[model_name] = {
                    "model": tf.keras.models.load_model(model_path, compile=False),
                    "preprocessor": joblib.load(preprocessor_path),
                }

    return (
        models_loaded,
        dl_models_loaded,
        label_encoder_loaded,
        features_original_loaded,
        features_hybrid_loaded,
        dl_model_info_loaded,
        missing_files,
    )


# =========================================================
# Helper Functions
# =========================================================


def format_prediction_label(label: str) -> str:
    if label == "satisfied":
        return "Satisfied"
    return "Neutral or Dissatisfied"


FUZZY_BINARY_THRESHOLD = 50.0


def format_fuzzy_category(category: str | None) -> str:
    if category is None or pd.isna(category):
        return "—"

    return str(category).replace("_", " ").title()


def fuzzy_linguistic_category(score: float) -> str:
    memberships = {
        category: trimf(score, mf_spec)
        for category, mf_spec in MF_OUTPUT.items()
    }
    max_membership = max(memberships.values())
    dominant_categories = [
        category
        for category, membership in memberships.items()
        if np.isclose(membership, max_membership)
    ]

    if len(dominant_categories) == 1:
        return dominant_categories[0]

    if score >= 75 and "puas" in dominant_categories:
        return "puas"
    if score <= 25 and "tidak_puas" in dominant_categories:
        return "tidak_puas"
    if "netral" in dominant_categories:
        return "netral"

    return dominant_categories[0]


def fuzzy_score_to_dataset_label(score: float) -> str:
    return "satisfied" if float(score) >= FUZZY_BINARY_THRESHOLD else "neutral or dissatisfied"


def prediction_badge(label: str) -> str:
    label_display = format_prediction_label(label)
    badge_class = "badge-success" if label_display == "Satisfied" else "badge-warning"
    return f'<span class="badge {badge_class}">{label_display}</span>'


def fuzzy_category_badge(category: str | None) -> str:
    if category is None or pd.isna(category):
        return "—"

    category_display = format_fuzzy_category(category)
    if category == "puas":
        badge_class = "badge-success"
    elif category == "netral":
        badge_class = "badge-warning"
    else:
        badge_class = "badge-neutral"

    return f'<span class="badge {badge_class}">{category_display}</span>'


def voting_badge(value: str) -> str:
    if value == "Ya":
        return '<span class="badge badge-blue">Ya</span>'
    return '<span class="badge badge-neutral">Tidak</span>'


def scenario_badge(scenario: str) -> str:
    if scenario == "With Fuzzy":
        return '<span class="badge badge-blue">With Fuzzy</span>'
    if scenario == "Without Fuzzy":
        return '<span class="badge badge-neutral">Without Fuzzy</span>'
    return '<span class="badge badge-neutral">Fuzzy</span>'


def fmt_number(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def fmt_score(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.2f}"


def normalize_cell_for_constant_check(value) -> str:
    if isinstance(value, str) and value.strip() in {"", "—", "-", "None", "nan", "NaN"}:
        return "__empty__"
    if value is None or pd.isna(value):
        return "__empty__"
    return str(value).strip()


def drop_constant_columns(df: pd.DataFrame, always_keep: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    always_keep = always_keep or []
    columns_to_drop = []

    for column in df.columns:
        if column in always_keep:
            continue

        normalized_values = df[column].map(normalize_cell_for_constant_check)
        if normalized_values.nunique(dropna=False) <= 1:
            columns_to_drop.append(column)

    return df.drop(columns=columns_to_drop)


def get_prediction_probability(model, input_data):
    if not hasattr(model, "predict_proba"):
        return None, None

    proba = model.predict_proba(input_data)[0]
    class_labels = list(label_encoder.classes_)

    neutral_idx = class_labels.index("neutral or dissatisfied")
    satisfied_idx = class_labels.index("satisfied")

    return proba[neutral_idx], proba[satisfied_idx]


def get_dl_prediction(model_bundle, input_data: pd.DataFrame, threshold: float):
    transformed_input = model_bundle["preprocessor"].transform(input_data)
    prediction = model_bundle["model"].predict(transformed_input, verbose=0)
    prediction_array = np.asarray(prediction)

    if prediction_array.ndim == 2 and prediction_array.shape[1] == 1:
        prob_satisfied = float(np.clip(prediction_array[0][0], 0.0, 1.0))
        prob_neutral = 1.0 - prob_satisfied
        prediction_label_raw = "satisfied" if prob_satisfied >= threshold else "neutral or dissatisfied"
        return prediction_label_raw, prob_neutral, prob_satisfied

    class_probs = prediction_array[0]
    class_labels = list(label_encoder.classes_)
    neutral_idx = class_labels.index("neutral or dissatisfied")
    satisfied_idx = class_labels.index("satisfied")
    prediction_idx = int(np.argmax(class_probs))
    prediction_label_raw = label_encoder.inverse_transform([prediction_idx])[0]

    return (
        prediction_label_raw,
        float(class_probs[neutral_idx]),
        float(class_probs[satisfied_idx]),
    )


def render_pretty_table(df: pd.DataFrame, include_score: bool = True) -> None:
    display_rows = []

    for _, row in df.iterrows():
        dataset_label_raw = row.get("Label Dataset Raw", row.get("Raw Label", None))
        if dataset_label_raw is None:
            dataset_label_display = row.get("Label Dataset", "—")
            dataset_label_raw = "satisfied" if dataset_label_display == "Satisfied" else "neutral or dissatisfied"

        display_row = {
            "No": row.get("No", "—"),
            "Metode": row.get("Metode", "—"),
            "Jenis": row.get("Jenis", "Fuzzy"),
            "Output Asli": row.get("Output Asli", "—"),
            "Kategori Fuzzy": row.get("Kategori Fuzzy Raw"),
            "Label Dataset": dataset_label_raw,
            "Dipakai Voting": row.get("Dipakai Voting", "Ya"),
        }

        if include_score:
            display_row["Score"] = row.get("Score")

        display_row["Prob. Neutral/Dissatisfied"] = row.get("Prob. Neutral/Dissatisfied")
        display_row["Prob. Satisfied"] = row.get("Prob. Satisfied")
        display_rows.append(display_row)

    display_df = drop_constant_columns(pd.DataFrame(display_rows), always_keep=["No", "Metode"])
    columns = list(display_df.columns)
    header_html = "".join([f"<th>{col}</th>" for col in columns])
    rows_html = []

    for _, row in display_df.iterrows():
        cells = []

        for column in columns:
            value = row.get(column)

            if column == "Metode":
                display_value = "—" if normalize_cell_for_constant_check(value) == "__empty__" else value
                cells.append(f"<td><strong>{display_value}</strong></td>")
            elif column == "Jenis":
                cells.append(f"<td>{scenario_badge(value)}</td>")
            elif column == "Kategori Fuzzy":
                cells.append(f"<td>{fuzzy_category_badge(value)}</td>")
            elif column == "Label Dataset":
                cells.append(f"<td>{prediction_badge(value)}</td>")
            elif column == "Dipakai Voting":
                cells.append(f"<td>{voting_badge(value)}</td>")
            elif column == "Score":
                cells.append(f"<td>{fmt_score(value)}</td>")
            elif column in {"Prob. Neutral/Dissatisfied", "Prob. Satisfied"}:
                cells.append(f"<td>{fmt_number(value)}</td>")
            else:
                display_value = "—" if normalize_cell_for_constant_check(value) == "__empty__" else value
                cells.append(f"<td>{display_value}</td>")

        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
    <div class="glass-card" style="padding: 0.6rem 0.85rem 0.85rem; overflow-x:auto;">
        <table class="pretty-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-kicker">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline() -> None:
    if ELEMENTS_AVAILABLE:
        try:
            with elements("mui_pipeline_header"):
                mui.Typography(
                    "Input → Fuzzifikasi → Inferensi → Defuzzifikasi → Machine Learning → Deep Learning → Output",
                    variant="body1",
                    sx={
                        "padding": "14px 18px",
                        "borderRadius": "18px",
                        "background": "linear-gradient(135deg, #eff6ff, #f8fafc)",
                        "border": "1px solid #dbeafe",
                        "fontFamily": "Google Sans, Inter, sans-serif",
                        "fontWeight": 600,
                        "color": "#1e3a8a",
                    },
                )
            return
        except Exception:
            pass

    st.markdown(
        """
        <div class="flow-wrap">
            <div class="flow-step"><div class="flow-num">1</div><div class="flow-title">Input</div><div class="flow-desc">Data layanan penumpang.</div></div>
            <div class="flow-step"><div class="flow-num">2</div><div class="flow-title">Fuzzy</div><div class="flow-desc">Mamdani dan Sugeno dihitung.</div></div>
            <div class="flow-step"><div class="flow-num">3</div><div class="flow-title">Hybrid</div><div class="flow-desc">Fuzzy score menjadi fitur ML.</div></div>
            <div class="flow-step"><div class="flow-num">4</div><div class="flow-title">10 Model AI</div><div class="flow-desc">8 ML klasik dan 2 MLP deep learning.</div></div>
            <div class="flow-step"><div class="flow-num">5</div><div class="flow-title">Output</div><div class="flow-desc">Ringkasan 12 hasil prediksi.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


FUZZY_VAR_LABELS = {
    "ob": "Online boarding",
    "wifi": "Inflight wifi service",
    "ent": "Inflight entertainment",
    "cls": "Class",
    "trav": "Type of Travel",
}


def describe_mf_spec(spec) -> tuple[str, str]:
    if isinstance(spec, tuple):
        mf_type = spec[0]
        params = spec[1] if mf_type == "trimf" else spec[1:]
        return mf_type, str(params)

    return "trimf", str(spec)


def build_variable_linguistic_df() -> pd.DataFrame:
    rows = []
    service_variables = [
        ("Online boarding", "ob"),
        ("Inflight wifi service", "wifi"),
        ("Inflight entertainment", "ent"),
    ]

    for variable_name, code in service_variables:
        for label, spec in MF_SERVICE.items():
            mf_type, params = describe_mf_spec(spec)
            rows.append(
                {
                    "Variabel": variable_name,
                    "Kode": code,
                    "Universe": "0-5",
                    "Himpunan": label,
                    "Fungsi": mf_type,
                    "Parameter": params,
                }
            )

    for label, spec in MF_CLASS.items():
        mf_type, params = describe_mf_spec(spec)
        rows.append(
            {
                "Variabel": "Class",
                "Kode": "cls",
                "Universe": "0=Eco, 1=Eco Plus, 2=Business",
                "Himpunan": label,
                "Fungsi": mf_type,
                "Parameter": params,
            }
        )

    for label, spec in MF_TRAVEL.items():
        mf_type, params = describe_mf_spec(spec)
        rows.append(
            {
                "Variabel": "Type of Travel",
                "Kode": "trav",
                "Universe": "0=Personal, 1=Business",
                "Himpunan": label,
                "Fungsi": mf_type,
                "Parameter": params,
            }
        )

    for label, spec in MF_OUTPUT.items():
        mf_type, params = describe_mf_spec(spec)
        rows.append(
            {
                "Variabel": "Output Satisfaction",
                "Kode": "out",
                "Universe": "0-100",
                "Himpunan": label,
                "Fungsi": mf_type,
                "Parameter": params,
            }
        )

    return pd.DataFrame(rows)


def build_membership_curve_df(universe: np.ndarray, mf_specs: dict) -> pd.DataFrame:
    data = {"x": universe}

    for label, spec in mf_specs.items():
        if isinstance(spec, tuple):
            data[label] = [eval_mf(x, spec) for x in universe]
        else:
            data[label] = [trimf(x, spec) for x in universe]

    return pd.DataFrame(data)


def get_membership_chart_configs() -> list[dict]:
    return [
        {
            "title": "Online Boarding",
            "x_label": "Nilai layanan 0-5",
            "universe": np.linspace(0, 5, 101),
            "specs": MF_SERVICE,
        },
        {
            "title": "Inflight Wifi Service",
            "x_label": "Nilai layanan 0-5",
            "universe": np.linspace(0, 5, 101),
            "specs": MF_SERVICE,
        },
        {
            "title": "Inflight Entertainment",
            "x_label": "Nilai layanan 0-5",
            "universe": np.linspace(0, 5, 101),
            "specs": MF_SERVICE,
        },
        {
            "title": "Class",
            "x_label": "0=Eco, 1=Eco Plus, 2=Business",
            "universe": np.linspace(0, 2, 101),
            "specs": MF_CLASS,
        },
        {
            "title": "Type of Travel",
            "x_label": "0=Personal, 1=Business",
            "universe": np.linspace(0, 1, 101),
            "specs": MF_TRAVEL,
        },
        {
            "title": "Output Satisfaction",
            "x_label": "Skor kepuasan 0-100",
            "universe": np.linspace(0, 100, 101),
            "specs": MF_OUTPUT,
        },
    ]


def format_rule_antecedent(rule: dict) -> str:
    return " AND ".join(
        f"{FUZZY_VAR_LABELS.get(var, var)} = {label}"
        for var, label in rule["ant"]
    )


def build_rule_base_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Rule": rule["id"],
                "IF": format_rule_antecedent(rule),
                "THEN": f"Satisfaction = {rule['con']}",
                "Kategori Output": rule["con"],
            }
            for rule in RULES
        ]
    )


def build_normalization_mapping_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Layer": "Fuzzy internal",
                "Output": "Tidak Puas / Netral / Puas",
                "Label Dataset": "Tidak langsung",
                "Dipakai Voting Final": "Tidak",
                "Catatan": "Mamdani dan Sugeno tetap menampilkan 3 kategori linguistik.",
            },
            {
                "Layer": "Konversi biner",
                "Output": f"Skor fuzzy < {FUZZY_BINARY_THRESHOLD:.0f}",
                "Label Dataset": format_prediction_label("neutral or dissatisfied"),
                "Dipakai Voting Final": "Ya",
                "Catatan": "Mengikuti label negatif dataset asli.",
            },
            {
                "Layer": "Konversi biner",
                "Output": f"Skor fuzzy >= {FUZZY_BINARY_THRESHOLD:.0f}",
                "Label Dataset": format_prediction_label("satisfied"),
                "Dipakai Voting Final": "Ya",
                "Catatan": "Disamakan dengan evaluasi fuzzy pada notebook Mamdani-Sugeno.",
            },
        ]
    )


def render_normalization_mapping() -> None:
    st.markdown('<div class="section-label"><span>NL</span>Normalization Layer</div>', unsafe_allow_html=True)
    st.caption(
        "Fuzzy tetap menghasilkan 3 kategori linguistik internal sesuai notebook: tidak puas, netral, dan puas. "
        "Karena model ML/DL memakai label asli dataset yang biner, voting final memakai konversi skor fuzzy yang sama "
        f"dengan evaluasi notebook fuzzy: skor < {FUZZY_BINARY_THRESHOLD:.0f} menjadi neutral or dissatisfied, "
        f"sedangkan skor >= {FUZZY_BINARY_THRESHOLD:.0f} menjadi satisfied."
    )
    st.dataframe(
        drop_constant_columns(build_normalization_mapping_df()),
        use_container_width=True,
        hide_index=True,
    )


def build_fuzzification_df(fuzzified_result: dict) -> pd.DataFrame:
    rows = []

    for variable_code, memberships in fuzzified_result.items():
        for label, degree in memberships.items():
            rows.append(
                {
                    "Variabel": FUZZY_VAR_LABELS.get(variable_code, variable_code),
                    "Kode": variable_code,
                    "Himpunan": label,
                    "Derajat Keanggotaan": float(degree),
                }
            )

    return pd.DataFrame(rows)


def build_inference_df(inference_result: list[tuple[int, float, str]]) -> pd.DataFrame:
    rules_by_id = {rule["id"]: rule for rule in RULES}
    rows = []

    for rule_id, strength, conclusion in inference_result:
        rule = rules_by_id[rule_id]
        rows.append(
            {
                "Rule": rule_id,
                "IF": format_rule_antecedent(rule),
                "Firing Strength": float(strength),
                "THEN": f"Satisfaction = {conclusion}",
                "Status": "Aktif" if strength > 0 else "Tidak aktif",
            }
        )

    return pd.DataFrame(rows)


def build_sugeno_weight_df(inference_result: list[tuple[int, float, str]]) -> pd.DataFrame:
    rows = []

    for rule_id, strength, conclusion in inference_result:
        if strength <= 0:
            continue

        singleton = SUGENO_OUTPUT[conclusion]
        rows.append(
            {
                "Rule": rule_id,
                "Firing Strength": float(strength),
                "Output Singleton": singleton,
                "Strength x Singleton": float(strength * singleton),
                "Conclusion": conclusion,
            }
        )

    return pd.DataFrame(rows)


def build_mamdani_aggregation_df(inference_result: list[tuple[int, float, str]]) -> pd.DataFrame:
    aggregated = []

    for x in UNIVERSE_OUTPUT:
        max_membership = 0.0
        for _, firing_strength, conclusion in inference_result:
            output_membership = trimf(x, MF_OUTPUT[conclusion])
            clipped = min(float(firing_strength), output_membership)
            max_membership = max(max_membership, clipped)
        aggregated.append(max_membership)

    return pd.DataFrame(
        {
            "Skor": UNIVERSE_OUTPUT,
            "Aggregated Membership": aggregated,
        }
    )


def render_fuzzy_knowledge_base() -> None:
    st.markdown('<div class="section-label"><span>KB</span>Knowledge Base Fuzzy</div>', unsafe_allow_html=True)
    tab_variable, tab_membership, tab_rules = st.tabs(
        [
            "Variable linguistic",
            "Fungsi keanggotaan + visualisasi",
            "Rule base",
        ]
    )

    with tab_variable:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            render_metric_card("Input Variables", "5", "Travel, class, wifi, boarding, entertainment.")
        with col_b:
            render_metric_card("Output Variable", "1", "Satisfaction score 0-100.")
        with col_c:
            render_metric_card("Rule Base", str(len(RULES)), "Memenuhi minimal 15 rule pada ketentuan.")

        st.dataframe(
            drop_constant_columns(build_variable_linguistic_df(), always_keep=["Variabel", "Himpunan"]),
            use_container_width=True,
            hide_index=True,
        )

    with tab_membership:
        st.caption("Semua fungsi keanggotaan diimplementasikan from scratch memakai trimf, zmf, dan smf sesuai file Colab.")
        st.code(
            "trimf(x, [a,b,c]) = segitiga\n"
            "zmf(x, a,b) = turun linear dari 1 ke 0\n"
            "smf(x, a,b) = naik linear dari 0 ke 1",
            language="text",
        )

        chart_configs = get_membership_chart_configs()
        for start in range(0, len(chart_configs), 2):
            cols = st.columns(2)
            for col, chart_config in zip(cols, chart_configs[start:start + 2]):
                with col:
                    st.write(f"**{chart_config['title']}**")
                    curve_df = build_membership_curve_df(
                        universe=chart_config["universe"],
                        mf_specs=chart_config["specs"],
                    )
                    st.line_chart(curve_df.set_index("x"), height=220, use_container_width=True)
                    st.caption(chart_config["x_label"])

    with tab_rules:
        st.caption("Operator AND pada antecedent dihitung dengan nilai minimum derajat keanggotaan.")
        st.dataframe(
            drop_constant_columns(build_rule_base_df(), always_keep=["Rule", "IF", "THEN"]),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    render_normalization_mapping()


def render_fuzzy_process(
    encoded_fuzzy_input: dict | None,
    fuzzified_result: dict | None,
    inference_result: list[tuple[int, float, str]] | None,
    mamdani_score: float | None,
    sugeno_score: float | None,
) -> None:
    st.markdown('<div class="section-label"><span>FX</span>Proses Perhitungan Fuzzy</div>', unsafe_allow_html=True)

    tab_fuzz, tab_infer, tab_defuzz = st.tabs(
        [
            "Fuzzifikasi",
            "Inferensi",
            "Defuzzifikasi",
        ]
    )

    with tab_fuzz:
        if encoded_fuzzy_input is None or fuzzified_result is None:
            st.info("Jalankan prediksi dari sidebar untuk melihat hasil fuzzifikasi berdasarkan input user.")
        else:
            st.write("Input kategorikal di-encode sebelum masuk ke fungsi keanggotaan.")
            encoded_df = pd.DataFrame([encoded_fuzzy_input])
            st.dataframe(encoded_df, use_container_width=True, hide_index=True)

            st.write("Derajat keanggotaan setiap input:")
            fuzzification_df = build_fuzzification_df(fuzzified_result)
            st.dataframe(
                drop_constant_columns(fuzzification_df, always_keep=["Variabel", "Himpunan"]),
                use_container_width=True,
                hide_index=True,
            )

    with tab_infer:
        if inference_result is None:
            st.info("Jalankan prediksi dari sidebar untuk melihat firing strength semua rule.")
        else:
            inference_df = build_inference_df(inference_result)
            active_rule_count = int((inference_df["Firing Strength"] > 0).sum())
            col_a, col_b = st.columns(2)
            with col_a:
                render_metric_card("Active Rules", str(active_rule_count), "Rule dengan firing strength lebih dari 0.")
            with col_b:
                render_metric_card("Inference Operator", "MIN", "Firing strength = minimum antecedent.")

            st.dataframe(
                drop_constant_columns(inference_df, always_keep=["Rule", "IF", "THEN"]),
                use_container_width=True,
                hide_index=True,
            )

    with tab_defuzz:
        if inference_result is None or mamdani_score is None or sugeno_score is None:
            st.info("Jalankan prediksi dari sidebar untuk melihat agregasi Mamdani dan bobot Sugeno.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                render_metric_card("Mamdani Defuzz", f"{mamdani_score:.2f}", "Centroid dari agregasi output fuzzy.")
            with col_b:
                render_metric_card("Sugeno Defuzz", f"{sugeno_score:.2f}", "Weighted average dari singleton output.")

            left, right = st.columns(2)
            with left:
                st.write("**Agregasi output Mamdani**")
                mamdani_df = build_mamdani_aggregation_df(inference_result)
                st.line_chart(mamdani_df.set_index("Skor"), height=260, use_container_width=True)
                st.caption("Setiap output rule dipotong oleh firing strength, lalu digabung dengan operator maksimum.")

            with right:
                st.write("**Bobot output Sugeno**")
                sugeno_weight_df = build_sugeno_weight_df(inference_result)
                if sugeno_weight_df.empty:
                    st.info("Tidak ada rule aktif. Nilai default defuzzifikasi adalah 50.")
                else:
                    st.dataframe(
                        drop_constant_columns(sugeno_weight_df, always_keep=["Rule"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    numerator = sugeno_weight_df["Strength x Singleton"].sum()
                    denominator = sugeno_weight_df["Firing Strength"].sum()
                    st.caption(f"Sugeno = {numerator:.3f} / {denominator:.3f} = {sugeno_score:.2f}")


# =========================================================
# Load Resources
# =========================================================

(
    models,
    dl_models,
    label_encoder,
    features_original,
    features_hybrid,
    dl_model_info,
    missing_files,
) = load_resources()


# =========================================================
# Sidebar Input
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">✈️ Passenger Input</div>
            <div class="sidebar-subtitle">
                Atur profil perjalanan dan nilai layanan untuk menjalankan Fuzzy Mamdani, Sugeno, 8 model ML, dan 2 model deep learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        type_of_travel = st.selectbox(
            "Type of Travel",
            ["Personal Travel", "Business travel"],
        )

        flight_class = st.selectbox(
            "Class",
            ["Eco", "Eco Plus", "Business"],
        )

        inflight_wifi_service = st.slider(
            "Inflight Wifi Service",
            min_value=0,
            max_value=5,
            value=3,
        )

        online_boarding = st.slider(
            "Online Boarding",
            min_value=0,
            max_value=5,
            value=3,
        )

        inflight_entertainment = st.slider(
            "Inflight Entertainment",
            min_value=0,
            max_value=5,
            value=3,
        )

        submitted = st.form_submit_button("Prediksi Kepuasan")

    st.markdown("---")
    st.caption("Light theme | Minimalist dashboard | Fuzzy + ML + Deep Learning comparison")
    st.caption(f"Dataset source: {DATASET_SOURCE_URL}")


# =========================================================
# Hero Section
# =========================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">✈️ Airline Satisfaction Intelligence</div>
        <div class="hero-title">Fuzzy Mamdani, Sugeno, Machine Learning, dan Deep Learning dalam satu dashboard.</div>
        <p class="hero-subtitle">
            Aplikasi ini membandingkan 12 output: Mamdani, Sugeno, 4 model ML tanpa fuzzy, 4 model ML dengan fuzzy score, dan 2 MLP deep learning. 
            Desain dibuat ringan, bersih, dan profesional untuk demo akademik.
        </p>
        <div class="hero-grid">
            <div class="mini-pill">🧠 Fuzzy Mamdani</div>
            <div class="mini-pill">📐 Fuzzy Sugeno</div>
            <div class="mini-pill">🤖 8 ML + 2 MLP</div>
            <div class="mini-pill">📊 12 Output</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Dataset source: [Airline Passenger Satisfaction - Kaggle]({DATASET_SOURCE_URL})"
)

st.write("")
render_pipeline()


# =========================================================
# Missing Model Guard
# =========================================================

if missing_files:
    st.markdown("<br>", unsafe_allow_html=True)
    with stylable_container(
        key="missing_model_box",
        css_styles="""
        {
            border: 1px solid #fecaca;
            background: #fff7f7;
            border-radius: 24px;
            padding: 1rem 1.1rem;
        }
        """,
    ):
        st.error("Beberapa file model belum ditemukan. Ekspor semua model terlebih dahulu agar dashboard 12 output bisa berjalan.")
        missing_df = pd.DataFrame({"File yang dibutuhkan": missing_files})
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
        st.info(
            "Letakkan model ML di exported_ml_model/ dan model MLP .keras beserta preprocessor .pkl di exported_dl_model/. "
            "Aplikasi membutuhkan 8 model ML, 2 model deep learning, label_encoder.pkl, dan fitur original/hybrid."
        )
    st.stop()

if label_encoder is None:
    st.error("label_encoder.pkl belum ditemukan.")
    st.stop()


# =========================================================
# Initial State
# =========================================================

if not submitted:
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        render_metric_card("Fuzzy Engine", "Ready", "Mamdani dan Sugeno dihitung dari input pengguna.")
    with col_b:
        render_metric_card("ML Pipelines", "8", "4 model without fuzzy dan 4 model with fuzzy.")
    with col_c:
        render_metric_card("Deep Learning", "2", "MLP without fuzzy dan with fuzzy.")
    with col_d:
        render_metric_card("Dashboard Output", "12", "2 fuzzy + 8 ML + 2 deep learning output.")

    st.markdown("<br>", unsafe_allow_html=True)
    render_fuzzy_knowledge_base()
    st.markdown("<br>", unsafe_allow_html=True)
    render_fuzzy_process(
        encoded_fuzzy_input=None,
        fuzzified_result=None,
        inference_result=None,
        mamdani_score=None,
        sugeno_score=None,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Isi parameter di sidebar, lalu tekan **Prediksi Kepuasan** untuk melihat hasil.")
    st.stop()


# =========================================================
# Prediction Pipeline
# =========================================================

encoded_fuzzy_input = {
    "Type of Travel": travel_mapping[type_of_travel],
    "Class": class_mapping[flight_class],
    "Inflight wifi service": inflight_wifi_service,
    "Online boarding": online_boarding,
    "Inflight entertainment": inflight_entertainment,
}

mamdani_score, sugeno_score, inference_result = calculate_fuzzy_scores(
    type_of_travel=type_of_travel,
    flight_class=flight_class,
    inflight_wifi_service=inflight_wifi_service,
    online_boarding=online_boarding,
    inflight_entertainment=inflight_entertainment,
)

mamdani_category = fuzzy_linguistic_category(mamdani_score)
sugeno_category = fuzzy_linguistic_category(sugeno_score)
mamdani_label = fuzzy_score_to_dataset_label(mamdani_score)
sugeno_label = fuzzy_score_to_dataset_label(sugeno_score)
fuzzified_result = fuzzify(encoded_fuzzy_input)

input_original = pd.DataFrame(
    [
        {
            "Type of Travel": type_of_travel,
            "Class": flight_class,
            "Inflight wifi service": inflight_wifi_service,
            "Online boarding": online_boarding,
            "Inflight entertainment": inflight_entertainment,
        }
    ]
)

input_hybrid = pd.DataFrame(
    [
        {
            "Type of Travel": type_of_travel,
            "Class": flight_class,
            "Inflight wifi service": inflight_wifi_service,
            "Online boarding": online_boarding,
            "Inflight entertainment": inflight_entertainment,
            "mamdani_score": mamdani_score,
            "sugeno_score": sugeno_score,
        }
    ]
)

input_original = input_original[features_original]
input_hybrid = input_hybrid[features_hybrid]

ml_results = []

for model_name, config in MODEL_CONFIGS.items():
    model = models[model_name]

    if config["input_type"] == "original":
        model_input = input_original
        scenario = "Without Fuzzy"
    else:
        model_input = input_hybrid
        scenario = "With Fuzzy"

    prediction_encoded = model.predict(model_input)[0]
    prediction_label_raw = label_encoder.inverse_transform([prediction_encoded])[0]
    prediction_label = format_prediction_label(prediction_label_raw)

    prob_neutral, prob_satisfied = get_prediction_probability(model=model, input_data=model_input)

    ml_results.append(
        {
            "Metode": config["display"],
            "Jenis": scenario,
            "Raw Label": prediction_label_raw,
            "Prediksi": prediction_label,
            "Output Asli": prediction_label,
            "Kategori Fuzzy Raw": None,
            "Kategori Fuzzy": "—",
            "Label Dataset Raw": prediction_label_raw,
            "Label Dataset": prediction_label,
            "Dipakai Voting": "Ya",
            "Voting Label Raw": prediction_label_raw,
            "Score": None,
            "Prob. Neutral/Dissatisfied": prob_neutral,
            "Prob. Satisfied": prob_satisfied,
        }
    )

ml_results_df = pd.DataFrame(ml_results)

dl_threshold = float(dl_model_info.get("threshold", 0.5))
dl_results = []

for model_name, config in DL_MODEL_CONFIGS.items():
    model_bundle = dl_models[model_name]

    if config["input_type"] == "original":
        model_input = input_original
        scenario = "Without Fuzzy"
    else:
        model_input = input_hybrid
        scenario = "With Fuzzy"

    prediction_label_raw, prob_neutral, prob_satisfied = get_dl_prediction(
        model_bundle=model_bundle,
        input_data=model_input,
        threshold=dl_threshold,
    )
    prediction_label = format_prediction_label(prediction_label_raw)

    dl_results.append(
        {
            "Metode": config["display"],
            "Jenis": scenario,
            "Raw Label": prediction_label_raw,
            "Prediksi": prediction_label,
            "Output Asli": prediction_label,
            "Kategori Fuzzy Raw": None,
            "Kategori Fuzzy": "—",
            "Label Dataset Raw": prediction_label_raw,
            "Label Dataset": prediction_label,
            "Dipakai Voting": "Ya",
            "Voting Label Raw": prediction_label_raw,
            "Score": None,
            "Prob. Neutral/Dissatisfied": prob_neutral,
            "Prob. Satisfied": prob_satisfied,
        }
    )

dl_results_df = pd.DataFrame(dl_results)

fuzzy_results_df = pd.DataFrame(
    [
        {
            "Metode": "Mamdani",
            "Jenis": "Fuzzy",
            "Raw Label": mamdani_label,
            "Prediksi": format_prediction_label(mamdani_label),
            "Output Asli": f"Mamdani score {mamdani_score:.2f}",
            "Kategori Fuzzy Raw": mamdani_category,
            "Kategori Fuzzy": format_fuzzy_category(mamdani_category),
            "Label Dataset Raw": mamdani_label,
            "Label Dataset": format_prediction_label(mamdani_label),
            "Dipakai Voting": "Ya",
            "Voting Label Raw": mamdani_label,
            "Score": mamdani_score,
            "Prob. Neutral/Dissatisfied": None,
            "Prob. Satisfied": None,
        },
        {
            "Metode": "Sugeno",
            "Jenis": "Fuzzy",
            "Raw Label": sugeno_label,
            "Prediksi": format_prediction_label(sugeno_label),
            "Output Asli": f"Sugeno score {sugeno_score:.2f}",
            "Kategori Fuzzy Raw": sugeno_category,
            "Kategori Fuzzy": format_fuzzy_category(sugeno_category),
            "Label Dataset Raw": sugeno_label,
            "Label Dataset": format_prediction_label(sugeno_label),
            "Dipakai Voting": "Ya",
            "Voting Label Raw": sugeno_label,
            "Score": sugeno_score,
            "Prob. Neutral/Dissatisfied": None,
            "Prob. Satisfied": None,
        },
    ]
)

summary_df = pd.concat([fuzzy_results_df, ml_results_df, dl_results_df], ignore_index=True)
summary_df.insert(0, "No", range(1, len(summary_df) + 1))
total_outputs = len(summary_df)

vote_counts = summary_df["Voting Label Raw"].value_counts()
satisfied_votes = int(vote_counts.get("satisfied", 0))
neutral_votes = int(vote_counts.get("neutral or dissatisfied", 0))

if satisfied_votes > neutral_votes:
    majority_label = "Satisfied"
    majority_subtitle = f"{satisfied_votes} dari {total_outputs} output normalized mengarah ke satisfied."
    majority_raw = "satisfied"
elif neutral_votes > satisfied_votes:
    majority_label = "Neutral or Dissatisfied"
    majority_subtitle = f"{neutral_votes} dari {total_outputs} output normalized mengarah ke neutral or dissatisfied."
    majority_raw = "neutral or dissatisfied"
else:
    majority_label = "Seimbang"
    majority_subtitle = f"Voting {total_outputs} output normalized memiliki jumlah yang sama."
    majority_raw = "neutral or dissatisfied"


# =========================================================
# Result Summary
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="result-banner">
        <div class="result-title">Final Majority Result</div>
        <p class="result-main">{majority_label}</p>
        <div class="result-sub">{majority_subtitle}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_1, col_2, col_3, col_4 = st.columns(4)

with col_1:
    render_metric_card(
        "Mamdani Score",
        f"{mamdani_score:.2f}",
        f"{format_fuzzy_category(mamdani_category)} -> {format_prediction_label(mamdani_label)}",
    )
with col_2:
    render_metric_card(
        "Sugeno Score",
        f"{sugeno_score:.2f}",
        f"{format_fuzzy_category(sugeno_category)} -> {format_prediction_label(sugeno_label)}",
    )
with col_3:
    render_metric_card("Vote Satisfied", str(satisfied_votes), f"Dihitung dari {total_outputs} output.")
with col_4:
    render_metric_card("Vote Neutral", str(neutral_votes), f"Dihitung dari {total_outputs} output.")

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# Tabs
# =========================================================

tab_summary, tab_ml, tab_dl, tab_fuzzy, tab_input = st.tabs(
    [
        "Ringkasan 12 Output",
        "Detail Machine Learning",
        "Detail Deep Learning",
        "Detail Fuzzy Logic",
        "Data Input Model",
    ]
)

with tab_summary:
    st.markdown('<div class="section-label"><span>AI</span>Ringkasan 12 Output</div>', unsafe_allow_html=True)
    render_pretty_table(summary_df, include_score=True)
    st.caption(
        "Voting final memakai Label Dataset yang sudah dinormalisasi. "
        "Untuk Mamdani dan Sugeno, kategori linguistik internal tetap ditampilkan sebagai Tidak Puas, Netral, atau Puas. "
        f"Konversi ke label biner mengikuti evaluasi notebook fuzzy: skor < {FUZZY_BINARY_THRESHOLD:.0f} menjadi neutral or dissatisfied, "
        f"sedangkan skor >= {FUZZY_BINARY_THRESHOLD:.0f} menjadi satisfied."
    )
    st.markdown("<br>", unsafe_allow_html=True)
    render_normalization_mapping()

with tab_ml:
    st.markdown('<div class="section-label"><span>🤖</span>Perbandingan 8 Output Machine Learning</div>', unsafe_allow_html=True)
    ml_show_df = ml_results_df.copy()
    ml_show_df.insert(0, "No", range(1, len(ml_show_df) + 1))
    render_pretty_table(ml_show_df, include_score=False)

    st.markdown("<br>", unsafe_allow_html=True)
    with stylable_container(
        key="ml_dataframe_container",
        css_styles="""
        {
            border: 1px solid #e5edf7;
            background: rgba(255,255,255,0.80);
            border-radius: 24px;
            padding: 1rem;
            box-shadow: 0 12px 28px rgba(15,23,42,0.04);
        }
        """,
    ):
        st.write("Versi dataframe untuk inspeksi detail:")
        st.dataframe(
            drop_constant_columns(
                ml_results_df.drop(
                    columns=["Raw Label", "Prediksi", "Kategori Fuzzy Raw", "Voting Label Raw"],
                    errors="ignore",
                ),
                always_keep=["Metode"],
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_dl:
    st.markdown('<div class="section-label"><span>DL</span>Perbandingan 2 Output Deep Learning</div>', unsafe_allow_html=True)
    dl_show_df = dl_results_df.copy()
    dl_show_df.insert(0, "No", range(1, len(dl_show_df) + 1))
    render_pretty_table(dl_show_df, include_score=False)
    st.caption(f"MLP memakai threshold probabilitas {dl_threshold:.2f} untuk menentukan label satisfied.")

    st.markdown("<br>", unsafe_allow_html=True)
    with stylable_container(
        key="dl_dataframe_container",
        css_styles="""
        {
            border: 1px solid #e5edf7;
            background: rgba(255,255,255,0.80);
            border-radius: 24px;
            padding: 1rem;
            box-shadow: 0 12px 28px rgba(15,23,42,0.04);
        }
        """,
    ):
        st.write("Versi dataframe untuk inspeksi detail:")
        st.dataframe(
            drop_constant_columns(
                dl_results_df.drop(
                    columns=["Raw Label", "Prediksi", "Kategori Fuzzy Raw", "Voting Label Raw"],
                    errors="ignore",
                ),
                always_keep=["Metode"],
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_fuzzy:
    render_fuzzy_knowledge_base()
    st.markdown("<br>", unsafe_allow_html=True)
    render_fuzzy_process(
        encoded_fuzzy_input=encoded_fuzzy_input,
        fuzzified_result=fuzzified_result,
        inference_result=inference_result,
        mamdani_score=mamdani_score,
        sugeno_score=sugeno_score,
    )

with tab_input:
    st.markdown('<div class="section-label"><span>📦</span>Data yang Masuk ke Model</div>', unsafe_allow_html=True)

    left_input, right_input = st.columns(2)
    with left_input:
        st.write("Input untuk model **without fuzzy**")
        st.dataframe(input_original, use_container_width=True, hide_index=True)
    with right_input:
        st.write("Input untuk model **with fuzzy**")
        st.dataframe(input_hybrid, use_container_width=True, hide_index=True)

st.caption(
    "Catatan: Fuzzy Mamdani-Sugeno tetap menampilkan 3 kategori internal, sedangkan ML dan DL memakai label asli dataset yang biner. "
    f"Untuk voting final, output fuzzy dikonversi dengan threshold skor {FUZZY_BINARY_THRESHOLD:.0f} agar konsisten dengan evaluasi notebook fuzzy."
)
