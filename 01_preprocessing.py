"""
STEP 1 - DATA PREPROCESSING
Input : data/global_inflation_post_covid.csv  (raw, 100,000 rows)
        data/country_metadata.csv             (second dataset for blending)
Output: data/inflation_cleaned.csv            (cleaned + blended, used by both models and Power BI)
        outputs/preprocessing_report.txt      (before -> steps -> after, for your report)
"""
import pandas as pd
import numpy as np

RAW = "data/global_inflation_post_covid.csv"
META = "data/country_metadata.csv"
OUT = "data/inflation_cleaned.csv"

NUM_COLS = ["inflation_rate", "interest_rate", "oil_price", "gdp_growth",
            "unemployment_rate", "money_supply_m2", "exchange_rate_usd",
            "food_price_index", "supply_chain_index"]

report = []
def log(msg):
    print(msg)
    report.append(str(msg))

# ---------- 1. Load ----------
df = pd.read_csv(RAW)
log("=== BEFORE PREPROCESSING ===")
log(f"Shape: {df.shape}")
log(f"Columns: {list(df.columns)}")
log("Missing values per column:\n" + df.isna().sum().to_string())
log(f"Duplicate rows: {df.duplicated().sum()}")
log("Summary statistics:\n" + df[NUM_COLS].describe().round(2).to_string())

# ---------- 2. Clean ----------
log("\n=== PREPROCESSING STEPS ===")
before = len(df)
df = df.drop_duplicates()
log(f"1. Removed duplicate rows: {before - len(df)}")

df["country"] = df["country"].astype(str).str.strip().str.upper()
log("2. Standardised country codes (trimmed spaces, upper case)")

df["date"] = pd.to_datetime(df["date"], format="%Y-%m", errors="coerce")
bad_dates = df["date"].isna().sum()
df = df.dropna(subset=["date"])
log(f"3. Converted 'date' (YYYY-MM text) to a real date; dropped {bad_dates} unparseable rows")

for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
missing = int(df[NUM_COLS].isna().sum().sum())
df[NUM_COLS] = df.groupby("country")[NUM_COLS].transform(lambda s: s.fillna(s.median()))
log(f"4. Coerced numeric columns; filled {missing} missing values with the country median")

# Outliers: cap at 1st/99th percentile (winsorising) instead of deleting rows
capped = 0
for c in NUM_COLS:
    lo, hi = df[c].quantile([0.01, 0.99])
    capped += int(((df[c] < lo) | (df[c] > hi)).sum())
    df[c] = df[c].clip(lo, hi)
log(f"5. Capped extreme outliers at the 1st/99th percentile ({capped} values adjusted)")

# Feature engineering
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["quarter"] = "Q" + df["date"].dt.quarter.astype(str)
df["real_interest_rate"] = (df["interest_rate"] - df["inflation_rate"]).round(2)
df["inflation_band"] = pd.cut(df["inflation_rate"], bins=[-np.inf, 2, 4, 6, np.inf],
                              labels=["Low (<2%)", "Moderate (2-4%)", "High (4-6%)", "Very High (>6%)"])
df["covid_phase"] = np.select(
    [df["date"] < "2021-01-01", df["date"] < "2023-01-01"],
    ["2020 Shock", "2021-22 Surge"], default="2023-24 Cooling")
log("6. Added features: year, month, quarter, real_interest_rate, inflation_band, covid_phase")

# ---------- 3. Blend with second dataset ----------
meta = pd.read_csv(META)
df = df.merge(meta, on="country", how="left")
log(f"7. Blended with country_metadata.csv (name, region, economy type, lat/long); "
    f"unmatched countries: {df['country_name'].isna().sum()}")

# ---------- 4. Save ----------
df["date"] = df["date"].dt.strftime("%Y-%m-%d")
df.to_csv(OUT, index=False)
log("\n=== AFTER PREPROCESSING ===")
log(f"Shape: {df.shape}")
log(f"Columns: {list(df.columns)}")
log(f"Saved -> {OUT}")

with open("outputs/preprocessing_report.txt", "w") as f:
    f.write("\n".join(report))
