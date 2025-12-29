import json
from datetime import datetime, timezone

import openml
import pandas as pd

from src.config import RAW_DIR

OPENML_DATASET_ID = 45577  # Give-Me-Some-Credit de OPENML


def main() -> None:
    dataset = openml.datasets.get_dataset(OPENML_DATASET_ID)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    df = X.copy()
    df["target"] = y

    out_path = RAW_DIR / "openml_give_me_some_credit_raw.parquet"
    df.to_parquet(out_path, index=False)

    meta = {
        "openml_dataset_id": OPENML_DATASET_ID,
        "openml_name": dataset.name,
        "default_target_attribute": dataset.default_target_attribute,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "attribute_names": attribute_names,
        "categorical_indicator": list(map(bool, categorical_indicator)),
    }
    (RAW_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Saved raw data to: {out_path}")
    print(f"[OK] Saved metadata to: {RAW_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()