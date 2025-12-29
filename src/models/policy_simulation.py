import json
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ARTIFACTS_DIR, FIGURES_DIR, RAW_DIR, REPORTS_DIR


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
    cfg_path = REPORTS_DIR.parents[0] / "policy_config.json"  # root/policy_config.json
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return PolicyConfig(
        ead=float(cfg["ead"]),
        annual_margin_rate=float(cfg["annual_margin_rate"]),
        lgd=float(cfg["lgd"]),
        acquisition_cost=float(cfg["acquisition_cost"]),
    )


def expected_profit_per_loan(pd: np.ndarray, cfg: PolicyConfig) -> np.ndarray:
    # E[profit] = (1 - PD)*gain - PD*loss - cost
    return (1.0 - pd) * cfg.gain_if_non_default - pd * cfg.loss_if_default - cfg.acquisition_cost


def realized_profit_per_loan(y_true: np.ndarray, cfg: PolicyConfig) -> np.ndarray:
    # y=0 -> no default -> gain - cost
    # y=1 -> default -> -loss - cost
    return np.where(
        y_true == 0,
        cfg.gain_if_non_default - cfg.acquisition_cost,
        -cfg.loss_if_default - cfg.acquisition_cost,
    )


def main() -> None:
    cfg = load_policy_config()
    print("[INFO] Policy config:", cfg)

    # 1) Cargar datos igual que en train_baseline
    path = RAW_DIR / "openml_give_me_some_credit_raw.parquet"
    df = pd.read_parquet(path).drop_duplicates()

    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    # Quitar columnas constantes
    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)

    # 2) Rehacer el mismo split determinístico
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
    )

    # 3) Cargar el modelo entrenado
    model_path = ARTIFACTS_DIR / "baseline_logreg_pd.joblib"
    model = joblib.load(model_path)

    # 4) Probabilidades PD
    pd_val = model.predict_proba(X_val)[:, 1]
    pd_test = model.predict_proba(X_test)[:, 1]

    # 5) Probar muchos umbrales y calcular:
    # - aprobación
    # - ganancia esperada
    # - ganancia realizada
    thresholds = np.linspace(0.01, 0.50, 200)

    val_exp_profit = []
    val_real_profit = []
    val_approve_rate = []

    test_exp_profit = []
    test_real_profit = []
    test_approve_rate = []

    for t in thresholds:
        approve_val = pd_val < t
        approve_test = pd_test < t

        val_approve_rate.append(float(np.mean(approve_val)))
        test_approve_rate.append(float(np.mean(approve_test)))

        # esperado
        val_exp_profit.append(float(np.sum(expected_profit_per_loan(pd_val[approve_val], cfg))))
        test_exp_profit.append(float(np.sum(expected_profit_per_loan(pd_test[approve_test], cfg))))

        # realizado
        val_real_profit.append(float(np.sum(realized_profit_per_loan(y_val.to_numpy()[approve_val], cfg))))
        test_real_profit.append(float(np.sum(realized_profit_per_loan(y_test.to_numpy()[approve_test], cfg))))

    # 6) Elegir el umbral óptimo EN VALIDACIÓN
    best_idx = int(np.argmax(val_exp_profit))
    best_t = float(thresholds[best_idx])

    summary = {
        "policy_config": {
            "ead": cfg.ead,
            "annual_margin_rate": cfg.annual_margin_rate,
            "lgd": cfg.lgd,
            "acquisition_cost": cfg.acquisition_cost,
        },
        "best_threshold_chosen_on": "validation_expected_profit",
        "best_threshold": best_t,
        "validation": {
            "approve_rate": val_approve_rate[best_idx],
            "expected_profit_total": val_exp_profit[best_idx],
            "realized_profit_total": val_real_profit[best_idx],
        },
        "test_at_best_threshold": {
            "approve_rate": test_approve_rate[best_idx],
            "expected_profit_total": test_exp_profit[best_idx],
            "realized_profit_total": test_real_profit[best_idx],
            "base_rate_test": float(y_test.mean()),
        },
    }

    # 7) Guardar summary
    out_path = REPORTS_DIR / "policy_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Saved policy summary to: {out_path}")

    # 8) Gráficos
    # (A) Profit vs threshold
    plt.figure()
    plt.plot(thresholds, val_exp_profit, label="Val Expected Profit")
    plt.plot(thresholds, test_exp_profit, label="Test Expected Profit")
    plt.plot(thresholds, val_real_profit, linestyle="--", label="Val Realized Profit")
    plt.plot(thresholds, test_real_profit, linestyle="--", label="Test Realized Profit")
    plt.axvline(best_t, linestyle=":", label=f"Best t={best_t:.3f}")
    plt.xlabel("Threshold (approve if PD < threshold)")
    plt.ylabel("Total Profit")
    plt.title("Profit vs Threshold")
    plt.legend()
    fig1 = FIGURES_DIR / "profit_vs_threshold.png"
    plt.savefig(fig1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved figure to: {fig1}")

    # (B) Approval rate vs threshold
    plt.figure()
    plt.plot(thresholds, val_approve_rate, label="Val Approve Rate")
    plt.plot(thresholds, test_approve_rate, label="Test Approve Rate")
    plt.axvline(best_t, linestyle=":", label=f"Best t={best_t:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Approval Rate")
    plt.title("Approval Rate vs Threshold")
    plt.legend()
    fig2 = FIGURES_DIR / "approval_rate_vs_threshold.png"
    plt.savefig(fig2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved figure to: {fig2}")

    print("\n[RESULTS] Policy summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
