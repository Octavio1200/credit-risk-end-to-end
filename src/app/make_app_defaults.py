import json
import pandas as pd
from src.config import RAW_DIR, REPORTS_DIR

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

def main():
    df = pd.read_parquet(RAW_DIR / "openml_give_me_some_credit_raw.parquet").drop_duplicates()
    X = df.drop(columns=["target"])
    defaults = {c: float(X[c].median()) for c in FEATURES if c in X.columns}
    out = REPORTS_DIR / "app_feature_defaults.json"
    out.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {out}")

if __name__ == "__main__":
    main()
