import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD DATA
# =========================================================
file_path = "Car Insurance Claim Data.csv"   # change path if needed
df = pd.read_csv(file_path)

# =========================================================
# 2. CLEAN COLUMN NAMES
# =========================================================
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Cleaned columns:")
print(df.columns.tolist())

# =========================================================
# 3. CLEAN NUMERIC COLUMNS
# =========================================================
numeric_cols = [
    "clm_freq", "clm_amt",
    "age", "income", "car_age", "mvr_pts",
    "home_val", "travtime"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================================================
# 4. DEFINE FEATURES
# =========================================================
features = [
    "age", "income", "car_age", "mvr_pts",
    "home_val", "travtime"
]

target_freq = "clm_freq"
target_sev = "clm_amt"

needed_cols = features + [target_freq, target_sev]
df_model = df[needed_cols].copy()

# =========================================================
# 5. CLEAN MODEL DATA
# =========================================================
df_model = df_model.replace([np.inf, -np.inf], np.nan)
df_model = df_model.dropna()

# optional: remove inconsistent rows
df_model = df_model[~((df_model["clm_freq"] > 0) & (df_model["clm_amt"] == 0))]

print("\nDtypes after cleaning:")
print(df_model.dtypes)

print("\nFirst 5 cleaned rows:")
print(df_model.head())

# =========================================================
# 6. FREQUENCY MODEL (NEGATIVE BINOMIAL GLM)
# =========================================================
X_freq = df_model[features].copy()
y_freq = df_model[target_freq].copy()

X_freq = X_freq.apply(pd.to_numeric, errors="coerce")
y_freq = pd.to_numeric(y_freq, errors="coerce")

freq_data = pd.concat([X_freq, y_freq], axis=1)
freq_data = freq_data.replace([np.inf, -np.inf], np.nan).dropna()

X_freq = freq_data[features].copy()
y_freq = freq_data[target_freq].copy()

X_freq = sm.add_constant(X_freq, has_constant="add")

print("\nFrequency model X dtypes:")
print(X_freq.dtypes)
print("\nFrequency model y dtype:")
print(y_freq.dtype)

nb_model = sm.GLM(
    y_freq,
    X_freq,
    family=sm.families.NegativeBinomial()
).fit()

print("\n================ NEGATIVE BINOMIAL MODEL SUMMARY ================\n")
print(nb_model.summary())

# =========================================================
# 7. SEVERITY MODEL (GAMMA GLM)
#    only use rows with positive claim amount
# =========================================================
df_sev = df_model[df_model[target_sev] > 0].copy()

X_sev = df_sev[features].copy()
y_sev = df_sev[target_sev].copy()

X_sev = X_sev.apply(pd.to_numeric, errors="coerce")
y_sev = pd.to_numeric(y_sev, errors="coerce")

sev_data = pd.concat([X_sev, y_sev], axis=1)
sev_data = sev_data.replace([np.inf, -np.inf], np.nan).dropna()

X_sev = sev_data[features].copy()
y_sev = sev_data[target_sev].copy()

X_sev = sm.add_constant(X_sev, has_constant="add")

print("\nSeverity model X dtypes:")
print(X_sev.dtypes)
print("\nSeverity model y dtype:")
print(y_sev.dtype)

gamma_model = sm.GLM(
    y_sev,
    X_sev,
    family=sm.families.Gamma(sm.families.links.Log())
).fit()

print("\n================ GAMMA MODEL SUMMARY ================\n")
print(gamma_model.summary())

# =========================================================
# 8. PREDICTIONS
# =========================================================
X_pred = df_model[features].copy()
X_pred = X_pred.apply(pd.to_numeric, errors="coerce")
X_pred = X_pred.replace([np.inf, -np.inf], np.nan).dropna()
X_pred = sm.add_constant(X_pred, has_constant="add")

df_pred = df_model.loc[X_pred.index].copy()

df_pred["pred_freq"] = nb_model.predict(X_pred)
df_pred["pred_sev"] = gamma_model.predict(X_pred)
df_pred["pure_premium"] = df_pred["pred_freq"] * df_pred["pred_sev"]

print("\n================ PREDICTIONS ================\n")
print(df_pred[[target_freq, target_sev, "pred_freq", "pred_sev", "pure_premium"]].head())

# =========================================================
# 9. SAVE OUTPUT
# =========================================================
df_pred.to_csv("insurance_pricing_predictions_nb.csv", index=False)
print("\nPredictions saved to: insurance_pricing_predictions_nb.csv")

# =========================================================
# 10. MODEL EVALUATION
# =========================================================
print("\n===== MODEL EVALUATION =====")

print("\n--- Frequency Model (Negative Binomial) ---")
print("Deviance:", nb_model.deviance)
print("Pearson Chi2:", nb_model.pearson_chi2)
print("AIC:", nb_model.aic)

print("\n--- Severity Model ---")
print("Deviance:", gamma_model.deviance)
print("Pearson Chi2:", gamma_model.pearson_chi2)
print("AIC:", gamma_model.aic)


# =========================================================
# 11. VISUALIZATIONS
# =========================================================

# ---------------------------
# Visualization 1:
# Actual vs Predicted Frequency
# ---------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df_pred["clm_freq"], df_pred["pred_freq"], alpha=0.4)
plt.xlabel("Actual Claim Frequency")
plt.ylabel("Predicted Claim Frequency")
plt.title("Actual vs Predicted Claim Frequency")
plt.grid(True)
plt.show()

# ---------------------------
# Visualization 2:
# Actual vs Predicted Severity
# only for policies with positive claim amount
# ---------------------------
df_pred_sev = df_pred[df_pred["clm_amt"] > 0].copy()

plt.figure(figsize=(8, 6))
plt.scatter(df_pred_sev["clm_amt"], df_pred_sev["pred_sev"], alpha=0.4)
plt.xlabel("Actual Claim Severity")
plt.ylabel("Predicted Claim Severity")
plt.title("Actual vs Predicted Claim Severity")
plt.grid(True)
plt.show()

# ---------------------------
# Visualization 3:
# Average Pure Premium by MVR Points
# ---------------------------
avg_premium_by_mvr = df_pred.groupby("mvr_pts")["pure_premium"].mean()

plt.figure(figsize=(8, 6))
avg_premium_by_mvr.plot(kind="bar")
plt.xlabel("MVR Points")
plt.ylabel("Average Pure Premium")
plt.title("Average Pure Premium by MVR Points")
plt.grid(True, axis="y")
