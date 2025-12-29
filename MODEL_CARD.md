# Model Card — Credit Risk PD Model (Calibrated)

## Overview
This project implements a probability of default (PD) model for consumer credit risk. The model outputs calibrated PD estimates and supports business decisioning via score bands (A–D) and a profit-based approval threshold.

## Intended Use
- **Primary use:** Demonstration / portfolio project for end-to-end credit risk modeling.
- **Outputs:** Calibrated PD (`P(default)`), score band, and decision suggestions (band rule and threshold rule).
- **Users:** Data scientists / analysts exploring credit risk workflows.

## Data
- **Source:** OpenML “Give Me Some Credit”.
- **Target:** Binary indicator of financial distress within 2 years (as provided by the dataset).
- **Splits:** Train / validation / test with stratification.

## Model
- **Type:** Logistic regression baseline with preprocessing (as implemented in this repo).
- **Calibration:** Sigmoid and Isotonic calibration evaluated; **Isotonic recommended** based on calibration quality (Brier).

## Performance (from this repo run)
### Discrimination (Test)
- ROC-AUC: ~0.803
- PR-AUC: ~0.325
- KS: ~0.465
- Base rate (test): ~0.067

### Calibration (Test)
Baseline probabilities were strongly miscalibrated (mean predicted PD far above base rate).
After calibration:
- **Isotonic (recommended):** Brier ~0.052, mean predicted PD ~0.069
- Sigmoid: Brier ~0.062

## Decisioning
### Score Bands
Applicants are segmented into four bands (A–D) using cutpoints derived from calibrated PD quantiles for demonstration.
- A: lowest PD
- D: highest PD

**Note:** In production, band cutpoints should be derived on training data and monitored for drift and population changes.

### Profit-based Policy Simulation
The repo includes a simple expected-profit model:
- Profit if non-default: `EAD * annual_margin_rate`
- Loss if default: `EAD * LGD`
- Net expected profit per loan: `(1-PD)*gain - PD*loss - acquisition_cost`

A threshold is chosen on validation to maximize expected profit, then evaluated on test.

## Limitations
- This is not a production credit scorecard.
- No fairness / bias assessment is included.
- Dataset may not represent modern lending portfolios or underwriting policies.
- Policy simulation uses simplified assumptions (single-period margin, fixed LGD/EAD, etc.).

## Monitoring Suggestions (non-exhaustive)
- Monitor prediction calibration (Brier / reliability plots) on recent cohorts.
- Track feature missingness and distribution shifts.
- Re-evaluate band cutpoints and policy thresholds periodically.
- Add alerting for significant changes in approval rates, default rates, or realized profitability.

## How to Reproduce
See `README.md` for commands. The full end-to-end run is available via:
```powershell
$env:PYTHONPATH="."
python -m src.run_all
