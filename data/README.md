# Insurance Pricing Model Using Negative Binomial and Gamma GLMs

## Overview
This project builds an actuarial-style insurance pricing model using a two-part framework:

- Negative Binomial GLM for claim frequency
- Gamma GLM for claim severity
- Pure premium calculated as predicted frequency × predicted severity

The project uses automobile insurance data and focuses on estimating expected loss for each policyholder.

---

## Objective
The goal is to model insurance risk more realistically by separating:

1. **Claim Frequency**: how often claims occur
2. **Claim Severity**: how large claims are when they occur

This is a common actuarial pricing approach because the drivers of frequency and severity are often different.

---

## Variables Used
The final model uses the following predictors:

- Age
- Income
- Car Age
- MVR Points
- Home Value
- Travel Time

Target variables:
- `clm_freq`: claim frequency
- `clm_amt`: claim severity

---

## Methodology

### 1. Data Cleaning
- Standardized column names
- Converted numeric-looking strings to numeric format
- Removed missing and infinite values
- Removed inconsistent rows where claim frequency was positive but claim amount was zero

### 2. Frequency Model
A **Negative Binomial GLM** was used instead of Poisson because the frequency data showed signs of overdispersion.

### 3. Severity Model
A **Gamma GLM with log link** was used for positive claim amounts only.

### 4. Pure Premium
Pure premium was calculated as:

`Predicted Frequency × Predicted Severity`

---

## Key Results
- MVR points strongly increase expected premium
- Frequency is more sensitive to driving behavior
- Severity is harder to predict and shows more variability
- The Negative Binomial model improves on Poisson when claim counts are overdispersed

---

## Visualizations
The project includes:
- Actual vs predicted claim frequency
- Actual vs predicted claim severity
- Average pure premium by MVR points

---

## Files
- `src/insurance_pricing_project.py`: main modeling script
- `outputs/insurance_pricing_predictions_nb.csv`: predicted premiums
- `outputs/*.png`: saved visualizations

---

## Tools Used
- Python
- pandas
- numpy
- statsmodels
- matplotlib

---

## How to Run
```bash
pip install -r requirements.txt
python src/insurance_pricing_project.py
