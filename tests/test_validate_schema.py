import pandas as pd
import pandera.pandas as pa

from src.data.validate import build_schema


def test_schema_accepts_binary_target():
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        }
    )
    schema = build_schema()
    schema.validate(df)


def test_schema_rejects_non_binary_target():
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "target": [0, 2, 0],
        }
    )
    schema = build_schema()
    try:
        schema.validate(df)
        assert False, "Expected schema to fail for non-binary target"
    except pa.errors.SchemaError:
        assert True
