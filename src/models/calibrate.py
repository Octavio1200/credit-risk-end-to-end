import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from src.config import ARTIFACTS_DIR, FIGURES_DIR, RAW_DIR, REPORTS_DIR


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "mean_pred": float(np.mean(proba)),
        "base_rate": float(np.mean(y_true)),
    }


def main() -> None:
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
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
    )

    # 2) Carcamos el modelo base entrenado
    base_model = joblib.load(ARTIFACTS_DIR / "baseline_logreg_pd.joblib")

    # 3) Probabilidades del modelo base
    val_proba_base = base_model.predict_proba(X_val)[:, 1]
    test_proba_base = base_model.predict_proba(X_test)[:, 1]

    base_metrics = {
        "val": evaluate(y_val, val_proba_base),
        "test": evaluate(y_test, test_proba_base),
    }

    # 4) Calibración del modelo (base_model) usando validación

    cal_sigmoid = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
    cal_sigmoid.fit(X_trainval, y_trainval)

    cal_isotonic = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=3)
    cal_isotonic.fit(X_trainval, y_trainval)

    test_proba_sigmoid = cal_sigmoid.predict_proba(X_test)[:, 1]
    test_proba_isotonic = cal_isotonic.predict_proba(X_test)[:, 1]

    cal_metrics = {
        "test_sigmoid": evaluate(y_test, test_proba_sigmoid),
        "test_isotonic": evaluate(y_test, test_proba_isotonic),
    }

    # 5) Guardamos los modelos calibrados
    joblib.dump(cal_sigmoid, ARTIFACTS_DIR / "calibrated_sigmoid.joblib")
    joblib.dump(cal_isotonic, ARTIFACTS_DIR / "calibrated_isotonic.joblib")
    print("[OK] Saved calibrated models in artifacts/")

    # 6) Diagramas de calibración
    plt.figure()
    for name, proba in [
        ("Baseline", test_proba_base),
        ("Sigmoid", test_proba_sigmoid),
        ("Isotonic", test_proba_isotonic),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (Test)")
    plt.legend()
    fig_path = FIGURES_DIR / "calibration_curve_test.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved calibration curve to: {fig_path}")

    # 7) Informe resumen
    report = {
        "baseline": base_metrics,
        "calibrated": cal_metrics,
        "recommended": "sigmoid" if cal_metrics["test_sigmoid"]["brier"] <= cal_metrics["test_isotonic"]["brier"] else "isotonic",
    }
    out_path = REPORTS_DIR / "calibration_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Saved calibration report to: {out_path}")

    print("\n[RESULTS] Calibration report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
