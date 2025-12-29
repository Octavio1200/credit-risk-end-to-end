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


def realized_profit_per_loan(y_true: np.ndarray, cfg: PolicyConfig) -> np.ndarray:
    return np.where(
        y_true == 0,
        cfg.gain_if_non_default - cfg.acquisition_cost,
        -cfg.loss_if_default - cfg.acquisition_cost,
    )


def main() -> None:
    cfg = load_policy_config()
    print("[INFO] Policy config:", cfg)

    # 1) cargamos data y preprocesamos igual que en train_baseline
    df = pd.read_parquet(RAW_DIR / "openml_give_me_some_credit_raw.parquet").drop_duplicates()
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
    )

    # 2) cargamos el modelo calibrado isotónico
    model = joblib.load(ARTIFACTS_DIR / "calibrated_isotonic.joblib")

    # 3) calibramos probabilidades en test
    pd_val = model.predict_proba(X_val)[:, 1]
    pd_test = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.01, 0.40, 200)

    val_exp_profit, val_real_profit, val_approve = [], [], []
    test_exp_profit, test_real_profit, test_approve = [], [], []

    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()

    for t in thresholds:
        approve_val = pd_val < t
        approve_test = pd_test < t

        val_approve.append(float(np.mean(approve_val)))
        test_approve.append(float(np.mean(approve_test)))

        val_exp_profit.append(float(np.sum(expected_profit_per_loan(pd_val[approve_val], cfg))))
        test_exp_profit.append(float(np.sum(expected_profit_per_loan(pd_test[approve_test], cfg))))

        val_real_profit.append(float(np.sum(realized_profit_per_loan(y_val_np[approve_val], cfg))))
        test_real_profit.append(float(np.sum(realized_profit_per_loan(y_test_np[approve_test], cfg))))

    best_idx = int(np.argmax(val_exp_profit))
    best_t = float(thresholds[best_idx])

    summary = {
        "model": "calibrated_isotonic",
        "best_threshold_chosen_on": "validation_expected_profit",
        "best_threshold": best_t,
        "validation": {
            "approve_rate": val_approve[best_idx],
            "expected_profit_total": val_exp_profit[best_idx],
            "realized_profit_total": val_real_profit[best_idx],
        },
        "test_at_best_threshold": {
            "approve_rate": test_approve[best_idx],
            "expected_profit_total": test_exp_profit[best_idx],
            "realized_profit_total": test_real_profit[best_idx],
            "base_rate_test": float(y_test.mean()),
        },
        "policy_config": {
            "ead": cfg.ead,
            "annual_margin_rate": cfg.annual_margin_rate,
            "lgd": cfg.lgd,
            "acquisition_cost": cfg.acquisition_cost,
        },
    }

    out_path = REPORTS_DIR / "policy_summary_calibrated.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Saved calibrated policy summary to: {out_path}")

    # Figures
    plt.figure()
    plt.plot(thresholds, val_exp_profit, label="Val Expected Profit")
    plt.plot(thresholds, test_exp_profit, label="Test Expected Profit")
    plt.axvline(best_t, linestyle=":", label=f"Best t={best_t:.3f}")
    plt.xlabel("Threshold (approve if PD < threshold)")
    plt.ylabel("Total Expected Profit")
    plt.title("Expected Profit vs Threshold (Calibrated PD)")
    plt.legend()
    fig1 = FIGURES_DIR / "profit_vs_threshold_calibrated.png"
    plt.savefig(fig1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved figure to: {fig1}")

    plt.figure()
    plt.plot(thresholds, val_approve, label="Val Approve Rate")
    plt.plot(thresholds, test_approve, label="Test Approve Rate")
    plt.axvline(best_t, linestyle=":", label=f"Best t={best_t:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Approval Rate")
    plt.title("Approval Rate vs Threshold (Calibrated PD)")
    plt.legend()
    fig2 = FIGURES_DIR / "approval_rate_vs_threshold_calibrated.png"
    plt.savefig(fig2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved figure to: {fig2}")

    print("\n[RESULTS] Calibrated policy summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
