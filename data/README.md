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

## Why Negative Binomial?

The standard Poisson model assumes that the mean and variance of claim counts are equal.  
However, this assumption was violated in the dataset.

Observed:
- The Pearson Chi-square statistic was significantly larger than the degrees of freedom  
- This indicates that the claim frequency data is **overdispersed**  

Implication:
- The Poisson model understates variability and may produce biased estimates  

Solution:
- A **Negative Binomial GLM** was used instead  

Benefits:
- Captures extra variability in claim counts  
- Provides a more realistic representation of insurance risk  
- Improves overall model fit  

---

## Key Results

- Modeled over **8,000+ policyholders**  
- MVR points show a strong positive relationship with expected premium  
- Claim frequency is more responsive to risk factors than claim severity  
- Negative Binomial provides a better fit than Poisson for overdispersed data  
- Claim severity remains difficult to predict due to heavy-tailed loss distribution  

---

## Business Interpretation

- Policyholders with higher MVR points should be charged higher premiums  
- Claim frequency and severity should be modeled separately due to different underlying drivers  
- Overdispersion must be accounted for when modeling claim counts  
- Pure premium enables effective **risk-based pricing segmentation**  


## Visualizations

### Claim Frequency
![Frequency](outputs/freq_actual_vs_pred.png)

### Claim Severity
![Severity](outputs/sev_actual_vs_pred.png)

### Premium by MVR Points
![MVR](outputs/premium_by_mvr.png)



## How to Run
```bash
pip install -r requirements.txt
python src/insurance_pricing_project.py






