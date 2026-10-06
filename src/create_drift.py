import pandas as pd
import numpy as np

REFERENCE_PATH = "data/reference.csv"
CURRENT_PATH = "data/current.csv"

# Load original/reference data
df = pd.read_csv(REFERENCE_PATH)

np.random.seed(42)

# ------------------------------------------------
# Create controlled numerical drift
# ------------------------------------------------

if "Heart Rate (bpm)" in df.columns:
    df["Heart Rate (bpm)"] = (
        df["Heart Rate (bpm)"] + np.random.normal(15, 5, len(df))
    )

if "Systolic Blood Pressure (mmHg)" in df.columns:
    df["Systolic Blood Pressure (mmHg)"] = (
        df["Systolic Blood Pressure (mmHg)"] + np.random.normal(15, 5, len(df))
    )

if "SpO2 Level (%)" in df.columns:
    df["SpO2 Level (%)"] = (
        df["SpO2 Level (%)"] - np.random.normal(2, 1, len(df))
    )

# ------------------------------------------------
# Create categorical drift
# ------------------------------------------------

if "Smoking" in df.columns:
    values = df["Smoking"].dropna().unique()

    if len(values) > 1:
        df.loc[:int(len(df) * 0.3), "Smoking"] = values[-1]

# Save modified dataset
df.to_csv(CURRENT_PATH, index=False)

print("✅ Drifted current.csv created successfully!")