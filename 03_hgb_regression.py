"""
STEP 3 - HISTGRADIENTBOOSTING REGRESSOR  (supervised: predict a number)

Question answered: "Given a country's economic indicators for a month, what inflation rate do we expect,
                    and which indicators drive inflation the most?"

Target  : inflation_rate
Features: country (categorical), year, month, interest_rate, oil_price, gdp_growth, unemployment_rate,
          money_supply_m2, exchange_rate_usd, food_price_index, supply_chain_index

Input : data/inflation_cleaned.csv
Output: outputs/hgb_predictions.csv        -> test rows: actual, predicted, error          (Power BI)
        outputs/hgb_metrics.csv            -> MAE, RMSE, R2 (train, test, 5-fold CV)      (Power BI KPI cards)
        outputs/hgb_feature_importance.csv -> permutation importance per feature          (Power BI bar chart)
        outputs/hgb_country_error.csv      -> MAE / RMSE per country                       (Power BI map)
        models/hgb_model.joblib                                                           (web app)
        outputs/plots/*.png                                                               (report)
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

TARGET = "inflation_rate"
NUM_FEATURES = ["year", "month", "interest_rate", "oil_price", "gdp_growth", "unemployment_rate",
                "money_supply_m2", "exchange_rate_usd", "food_price_index", "supply_chain_index"]
FEATURES = ["country"] + NUM_FEATURES
RANDOM_STATE = 42

# ---------- 1. Load ----------
df = pd.read_csv("data/inflation_cleaned.csv")
X = df[FEATURES].copy()
X["country"] = X["country"].astype("category")   # HGB handles categories natively, no one-hot needed
y = df[TARGET]

# ---------- 2. Train / test split (80 / 20) ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("Train:", X_train.shape, " Test:", X_test.shape)

# ---------- 3. Model ----------
model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.1,
    max_iter=300,             # number of boosting trees
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    categorical_features="from_dtype",
    early_stopping=True,      # stops adding trees when validation score stops improving
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)
print("Trees used (after early stopping):", model.n_iter_)

# ---------- 4. Evaluate ----------
def scores(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)}

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
train_s, test_s = scores(y_train, pred_train), scores(y_test, pred_test)
cv = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2")

metrics = pd.DataFrame([
    {"metric": "MAE",  "train": train_s["MAE"],  "test": test_s["MAE"]},
    {"metric": "RMSE", "train": train_s["RMSE"], "test": test_s["RMSE"]},
    {"metric": "R2",   "train": train_s["R2"],   "test": test_s["R2"]},
    {"metric": "CV_R2_mean (5-fold)", "train": np.nan, "test": cv.mean()},
    {"metric": "CV_R2_std (5-fold)",  "train": np.nan, "test": cv.std()},
]).round(4)
# Baseline: always predicting the training mean (shows how much the model adds)
baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
metrics.loc[len(metrics)] = {"metric": "Baseline_MAE (predict mean)", "train": np.nan, "test": round(baseline_mae, 4)}
metrics.to_csv("outputs/hgb_metrics.csv", index=False)
print(metrics.to_string(index=False))

# ---------- 5. Predictions table for Power BI ----------
out = df.loc[X_test.index, ["country", "country_name", "region", "economy_type", "latitude", "longitude",
                            "date", "year", "quarter", "covid_phase"]].copy()
out["actual_inflation"] = y_test.round(3)
out["predicted_inflation"] = pred_test.round(3)
out["error"] = (out["actual_inflation"] - out["predicted_inflation"]).round(3)   # + = model under-predicted
out["abs_error"] = out["error"].abs()
out["error_band"] = pd.cut(out["abs_error"], [-0.001, 0.25, 0.5, 1, np.inf],
                           labels=["< 0.25", "0.25-0.5", "0.5-1", "> 1"])
out.to_csv("outputs/hgb_predictions.csv", index=False)

country_err = (out.groupby(["country", "country_name", "latitude", "longitude"])
                  .agg(MAE=("abs_error", "mean"),
                       RMSE=("error", lambda e: np.sqrt((e ** 2).mean())),
                       avg_actual=("actual_inflation", "mean"),
                       avg_predicted=("predicted_inflation", "mean"),
                       n_test_rows=("error", "size"))
                  .round(3).reset_index())
country_err.to_csv("outputs/hgb_country_error.csv", index=False)

# ---------- 6. Feature importance (permutation: how much R2 drops when a feature is shuffled) ----------
sample = X_test.sample(10000, random_state=RANDOM_STATE)
pi = permutation_importance(model, sample, y_test.loc[sample.index], n_repeats=5,
                            random_state=RANDOM_STATE, scoring="r2")
imp = (pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean, "std": pi.importances_std})
         .sort_values("importance", ascending=False).round(4))
imp.to_csv("outputs/hgb_feature_importance.csv", index=False)
print("\nFeature importance:\n", imp.to_string(index=False))

joblib.dump(model, "models/hgb_model.joblib")
joblib.dump(list(X["country"].cat.categories), "models/hgb_country_categories.joblib")

# ---------- 7. Python visualisations (for the report) ----------
s = out.sample(3000, random_state=RANDOM_STATE)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(s.actual_inflation, s.predicted_inflation, s=6, alpha=.4)
lim = [out.actual_inflation.min(), out.actual_inflation.max()]
ax.plot(lim, lim, "r--", label="Perfect prediction")
ax.set(title=f"Actual vs Predicted (test R² = {test_s['R2']:.3f})", xlabel="Actual inflation %", ylabel="Predicted inflation %")
ax.legend(); plt.tight_layout(); plt.savefig("outputs/plots/hgb_actual_vs_predicted.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(out.error, bins=60, color="tab:purple")
ax.set(title=f"Prediction error distribution (MAE = {test_s['MAE']:.3f})", xlabel="Actual − Predicted", ylabel="Count")
plt.tight_layout(); plt.savefig("outputs/plots/hgb_error_histogram.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(imp.feature[::-1], imp.importance[::-1], xerr=imp["std"][::-1], color="tab:green")
ax.set(title="Permutation feature importance", xlabel="Drop in R² when shuffled")
plt.tight_layout(); plt.savefig("outputs/plots/hgb_feature_importance.png", dpi=150); plt.close()

trend = out.groupby("date")[["actual_inflation", "predicted_inflation"]].mean()
ax = trend.plot(figsize=(10, 4), title="Average actual vs predicted inflation over time (test set)")
ax.set_ylabel("Inflation %"); plt.tight_layout(); plt.savefig("outputs/plots/hgb_trend.png", dpi=150); plt.close()
print("\nDone. Files written to outputs/ and models/")
