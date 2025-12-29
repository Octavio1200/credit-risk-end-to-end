import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

from src.config import RAW_DIR


def build_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        columns={
            "target": Column(
                pa.Int64,
                checks=[Check.isin([0, 1])],
                nullable=False,
                required=True,
            ),
        },
        checks=[
            Check(lambda df: df.shape[0] > 0, error="Dataset has 0 rows."),
            Check(lambda df: df.shape[1] > 1, error="Dataset must have features + target."),
        ],
        strict=False,
        coerce=True,
    )


def main() -> None:
    path = RAW_DIR / "openml_give_me_some_credit_raw.parquet"
    df = pd.read_parquet(path)

    schema = build_schema()
    schema.validate(df)

    dup_rate = df.duplicated().mean()
    null_rate = df.isna().mean().sort_values(ascending=False).head(10)

    print("[OK] Schema validation passed.")
    print(f"[INFO] Duplicate row rate: {dup_rate:.4f}")
    print("[INFO] Top-10 columns by null rate:")
    print(null_rate.to_string())


if __name__ == "__main__":
    main()