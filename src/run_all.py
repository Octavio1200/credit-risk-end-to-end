import os
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    # Asegurar que el directorio raíz del proyecto esté en PYTHONPATH
    os.environ.setdefault("PYTHONPATH", ".")

    steps = [
        ["python", "-m", "src.data.ingest"],
        ["python", "-m", "src.data.validate"],
        ["python", "-m", "src.models.train_baseline"],
        ["python", "-m", "src.models.calibrate"],
        ["python", "-m", "src.models.score_bands"],
        ["python", "-m", "src.models.policy_simulation_calibrated"],
    ]

    for cmd in steps:
        _run(cmd)

    print("\n[OK] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
