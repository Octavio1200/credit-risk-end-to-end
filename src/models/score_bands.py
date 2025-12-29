import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ARTIFACTS_DIR, RAW_DIR, REPORTS_DIR


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


def expected_profit_per_loan(pd: np.ndarray, cfg: PolicyConfig) -> np.ndarray:
    return (1.0 - pd) * cfg.gain_if_non_default - pd * cfg.loss_if_default - cfg.acquisition_cost


def main() -> None:
    cfg = load_policy_config()

    # 1) Cargamos data y preprocesamos igual que en train_baseline
    df = pd.read_parquet(RAW_DIR / "openml_give_me_some_credit_raw.parquet").drop_duplicates()
    y = df["target"].astype(int).to_numpy()
    X = df.drop(columns=["target"])

    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2) Cargamos el modelo calibrado isotónico
    model = joblib.load(ARTIFACTS_DIR / "calibrated_isotonic.joblib")

    # 3) Probabilidades calibradas en test
    pd_test = model.predict_proba(X_test)[:, 1]

    # 4) Creamos los score bands según cuantiles
    # A = riesgo bajo, D = riesgo alto
    q = np.quantile(pd_test, [0.25, 0.50, 0.75])
    # bandas: A: <=q1, B: (q1,q2], C: (q2,q3], D: >q3
    band = np.where(
        pd_test <= q[0], "A",
        np.where(pd_test <= q[1], "B",
                 np.where(pd_test <= q[2], "C", "D"))
    )

    # 5) Construimos el reporte por bandas
    rep = pd.DataFrame({
        "pd_calibrated": pd_test,
        "y": y_test,
        "band": band
    })

    # Para cada banda, calculamos métricas
    out_rows = []
    for b in ["A", "B", "C", "D"]:
        sub = rep[rep["band"] == b]
        n = len(sub)
        share = n / len(rep)
        default_rate = float(sub["y"].mean()) if n > 0 else float("nan")
        avg_pd = float(sub["pd_calibrated"].mean()) if n > 0 else float("nan")

        exp_profit_loan = float(np.mean(expected_profit_per_loan(sub["pd_calibrated"].to_numpy(), cfg))) if n > 0 else float("nan")
        exp_profit_total = float(np.sum(expected_profit_per_loan(sub["pd_calibrated"].to_numpy(), cfg))) if n > 0 else float("nan")

        out_rows.append({
            "band": b,
            "n": int(n),
            "share": float(share),
            "avg_pd": avg_pd,
            "realized_default_rate": default_rate,
            "expected_profit_per_loan": exp_profit_loan,
            "expected_profit_total": exp_profit_total,
        })

    bands_table = pd.DataFrame(out_rows)

    # 6) Guardamos el reporte
    csv_path = REPORTS_DIR / "score_bands_test.csv"
    bands_table.to_csv(csv_path, index=False)

    json_path = REPORTS_DIR / "score_bands_test.json"
    json_path.write_text(bands_table.to_json(orient="records", indent=2), encoding="utf-8")

    print(f"[OK] Saved bands table CSV to: {csv_path}")
    print(f"[OK] Saved bands table JSON to: {json_path}")

    print("\n[RESULTS] Score bands (Test):")
    print(bands_table.to_string(index=False))

    # 7) También guardamos los cutpoints
    cutpoints = {
        "definition": "Quantile-based bands on calibrated PD (test set). A=lowest risk, D=highest risk.",
        "q25": float(q[0]),
        "q50": float(q[1]),
        "q75": float(q[2]),
    }
    cp_path = REPORTS_DIR / "score_band_cutpoints.json"
    cp_path.write_text(json.dumps(cutpoints, indent=2), encoding="utf-8")
    print(f"\n[OK] Saved cutpoints to: {cp_path}")


if __name__ == "__main__":
    main()
