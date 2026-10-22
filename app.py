"""
STEP 4 - WEB SERVER (Flask)
Serves both trained models in a browser form and as a JSON API.

Run locally :  python app.py          -> open http://127.0.0.1:5000
Run in prod :  gunicorn app:app       (used by Render / Railway / PythonAnywhere)

The user enters a country, month and 8 economic indicators:
  1. HistGradientBoostingRegressor predicts the inflation rate.
  2. K-Means assigns the economic "regime" cluster using the indicators + predicted inflation.
"""
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))
M = lambda f: os.path.join(BASE, "models", f)

hgb = joblib.load(M("hgb_model.joblib"))
countries = joblib.load(M("hgb_country_categories.joblib"))
kmeans = joblib.load(M("kmeans_model.joblib"))
scaler = joblib.load(M("kmeans_scaler.joblib"))
cluster_names = joblib.load(M("kmeans_cluster_names.joblib"))
metrics = pd.read_csv(os.path.join(BASE, "outputs", "hgb_metrics.csv")).set_index("metric")["test"]
meta = pd.read_csv(os.path.join(BASE, "data", "country_metadata.csv")).set_index("country")

INDICATORS = {   # field: (label, default value = dataset median)
    "interest_rate": ("Interest rate (%)", 2.59),
    "oil_price": ("Oil price (USD/barrel)", 75.2),
    "gdp_growth": ("GDP growth (%)", 2.52),
    "unemployment_rate": ("Unemployment rate (%)", 5.5),
    "money_supply_m2": ("Money supply M2", 11788),
    "exchange_rate_usd": ("Exchange rate vs USD", 42.36),
    "food_price_index": ("Food price index", 119.83),
    "supply_chain_index": ("Supply-chain stress index", 1.67),
}
KMEANS_FEATURES = ["inflation_rate"] + list(INDICATORS)
HGB_FEATURES = ["country", "year", "month"] + list(INDICATORS)

app = Flask(__name__)


def run_models(data: dict) -> dict:
    missing = [f for f in HGB_FEATURES if f not in data or data[f] in ("", None)]
    if missing:
        raise ValueError(f"Missing field(s): {', '.join(missing)}")
    row ={"country": data["country"].upper(), "year": int(data["year"]), "month": int(data["month"])}
    for f in INDICATORS:
        row[f] = float(data[f])
    if row["country"] not in countries:
        raise ValueError(f"Unknown country '{row['country']}'. Use one of: {', '.join(countries)}")

    X = pd.DataFrame([row])[HGB_FEATURES]
    X["country"] = pd.Categorical(X["country"], categories=countries)
    pred = float(hgb.predict(X)[0])

    k_in = pd.DataFrame([{**row, "inflation_rate": pred}])[KMEANS_FEATURES]
    cid = int(kmeans.predict(scaler.transform(k_in))[0])
    return {"country": row["country"], "country_name": meta.loc[row["country"], "country_name"],
            "predicted_inflation": round(pred, 2),
            "expected_range": [round(float(pred - metrics["RMSE"]), 2), round(float(pred + metrics["RMSE"]), 2)],
            "cluster_id": cid, "cluster_name": cluster_names[cid]}


@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    form = {"country": "IND", "year": 2024, "month": 6, **{k: v[1] for k, v in INDICATORS.items()}}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            result = run_models(form)
        except (ValueError, KeyError) as e:
            error = str(e)
    return render_template("index.html", countries=countries, meta=meta, indicators=INDICATORS,
                           form=form, result=result, error=error,
                           r2=metrics["R2"], mae=metrics["MAE"], rmse=metrics["RMSE"])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        return jsonify(run_models(request.get_json(force=True)))
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
