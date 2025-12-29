import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import ARTIFACTS_DIR, FIGURES_DIR, RAW_DIR, REPORTS_DIR


def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def main() -> None:
    # 1) Carga de datos
    path = RAW_DIR / "openml_give_me_some_credit_raw.parquet"
    df = pd.read_parquet(path)

    # 2) Limpieza de datos: quitamos duplicados
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"[INFO] Dropped duplicates: {before - after} rows removed.")

    # 3) Diferenciar X / Y
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    # 4) Quitar columnas  con constantes
    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)
        print(f"[INFO] Dropped constant columns: {constant_cols}")

    # 5) División de base: train/val/test (estratificado)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
    )
    print(f"[INFO] Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # 6) Columnas numéricas y categóricas
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    # 7) Preprocesamiento
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    # 8) Modelo baseline: Regresión Logística (PD)
    # class_weight="balanced" ayuda cuando hay desbalance (común en default).
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", clf),
        ]
    )

    # 9) Entrenar
    pipe.fit(X_train, y_train)

    # 10) Predicciones (probabilidades)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]

    # 11) Métricas
    metrics = {
        "val": {
            "roc_auc": float(roc_auc_score(y_val, val_proba)),
            "pr_auc": float(average_precision_score(y_val, val_proba)),
            "brier": float(brier_score_loss(y_val, val_proba)),
            "ks": float(ks_statistic(y_val.to_numpy(), val_proba)),
        },
        "test": {
            "roc_auc": float(roc_auc_score(y_test, test_proba)),
            "pr_auc": float(average_precision_score(y_test, test_proba)),
            "brier": float(brier_score_loss(y_test, test_proba)),
            "ks": float(ks_statistic(y_test.to_numpy(), test_proba)),
        },
        "base_rate_test": float(y_test.mean()),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }

    # 12) Guardar modelo
    model_path = ARTIFACTS_DIR / "baseline_logreg_pd.joblib"
    joblib.dump(pipe, model_path)
    print(f"[OK] Saved model to: {model_path}")

    # 13) Guardar métricas
    metrics_path = REPORTS_DIR / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] Saved metrics to: {metrics_path}")

    # 14) Gráfico ROC (test)
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    roc_path = FIGURES_DIR / "roc_test.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved ROC plot to: {roc_path}")

    # 15) Gráfico Precision-Recall (test)
    precision, recall, _ = precision_recall_curve(y_test, test_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test)")
    pr_path = FIGURES_DIR / "pr_test.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved PR plot to: {pr_path}")

    # 16) Imprimir resumen
    print("\n[RESULTS] Baseline metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
