import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path BEFORE importing from src.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS_DIR, RAW_DIR, REPORTS_DIR  # noqa: E402

@dataclass(frozen=True)
class PolicyConfig:
    ead: float
    annual_margin_rate: float
    lgd: float
    acquisition_cost: float

    @property
    def gain_if_non_default(self) -> float:
        return self.ead * self.annual_margin_rate

    @property
    def loss_if_default(self) -> float:
        return self.ead * self.lgd


def load_policy_config() -> PolicyConfig:
    cfg_path = REPORTS_DIR.parents[0] / "policy_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return PolicyConfig(
        ead=float(cfg["ead"]),
        annual_margin_rate=float(cfg["annual_margin_rate"]),
        lgd=float(cfg["lgd"]),
        acquisition_cost=float(cfg["acquisition_cost"]),
    )


def expected_profit(pd_value: float, cfg: PolicyConfig) -> float:
    return (1.0 - pd_value) * cfg.gain_if_non_default - pd_value * cfg.loss_if_default - cfg.acquisition_cost


def load_cutpoints() -> dict:
    cp_path = REPORTS_DIR / "score_band_cutpoints.json"
    return json.loads(cp_path.read_text(encoding="utf-8"))


def assign_band(pd_value: float, cutpoints: dict) -> str:
    q25, q50, q75 = cutpoints["q25"], cutpoints["q50"], cutpoints["q75"]
    if pd_value <= q25:
        return "A"
    if pd_value <= q50:
        return "B"
    if pd_value <= q75:
        return "C"
    return "D"


def band_decision(band: str) -> str:

    return {"A": "APPROVE", "B": "APPROVE", "C": "REVIEW", "D": "REJECT"}[band]


@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS_DIR / "calibrated_isotonic.joblib")


@st.cache_data
def load_example_row() -> pd.Series:
    df = pd.read_parquet(RAW_DIR / "openml_give_me_some_credit_raw.parquet").drop_duplicates()
    X = df.drop(columns=["target"])
    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)
    return X.iloc[0]


def main() -> None:
    st.title("Credit Risk Scoring Demo (Calibrated PD + Score Bands)")

    st.write(
        "This demo uses a calibrated probability of default (PD) model and assigns score bands (A-D). "
        "It then provides a decision and expected profit under configurable assumptions."
    )

    cfg = load_policy_config()
    cutpoints = load_cutpoints()
    model = load_model()

    example = load_example_row()

    st.subheader("1) Input features")
    st.caption("Edit values and click 'Score'.")

    inputs = {}
    for col, val in example.items():
        if pd.api.types.is_number(val):
            inputs[col] = st.number_input(col, value=float(val))
        else:
            inputs[col] = st.text_input(col, value=str(val))

    if st.button("Score"):
        X_input = pd.DataFrame([inputs])

        pd_value = float(model.predict_proba(X_input)[:, 1][0])

        band = assign_band(pd_value, cutpoints)
        decision = band_decision(band)
        exp_prof = expected_profit(pd_value, cfg)

        best_threshold = 0.23733668341708547
        decision_threshold = "APPROVE" if pd_value < best_threshold else "REJECT"

        st.subheader("2) Results")
        st.metric("Calibrated PD", f"{pd_value:.4f}")
        st.metric("Score Band", band)
        st.metric("Decision", decision)
        st.metric("Expected Profit (per loan)", f"{exp_prof:.2f}")
        st.metric("Best Threshold (calibrated)", f"{best_threshold:.4f}")
        st.metric("Decision (threshold rule)", decision_threshold)

        st.subheader("3) Policy assumptions")
        st.json(
            {
                "ead": cfg.ead,
                "annual_margin_rate": cfg.annual_margin_rate,
                "lgd": cfg.lgd,
                "acquisition_cost": cfg.acquisition_cost,
                "cutpoints": cutpoints,
            }
        )


if __name__ == "__main__":
    main()
