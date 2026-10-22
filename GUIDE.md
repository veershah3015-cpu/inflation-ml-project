# Post-COVID Global Inflation: Step-by-Step Implementation Guide
**K-Means Clustering + HistGradientBoosting Regressor → Power BI Dashboard → Web Server**

---

## 0. What's in this folder

```
inflation_ml_project/
├── data/
│   ├── global_inflation_post_covid.csv   ← ORIGINAL dataset (100,000 rows)
│   ├── country_metadata.csv              ← 2nd dataset for BLENDING (name, region, lat/long)
│   └── inflation_cleaned.csv             ← PREPROCESSED dataset (created by step 1)
├── 01_preprocessing.py                   ← Step 1: clean + feature engineering + blend
├── 02_kmeans_clustering.py               ← Step 2: K-Means
├── 03_hgb_regression.py                  ← Step 3: HistGradientBoosting Regressor
├── inflation_ml_notebook.ipynb           ← Same 3 steps as a Jupyter notebook (already run)
├── app.py + templates/index.html         ← Step 4: Flask web server
├── models/                               ← saved trained models (.joblib) used by the web app
├── outputs/                              ← CSVs you import into Power BI + plots/ for the report
├── powerbi_dax_measures.txt              ← all DAX to copy-paste into Power BI
├── requirements.txt, Procfile            ← for installing and deploying
└── GUIDE.md                              ← this file
```

**Order of work:** Preprocess → K-Means → HGB → check the Python plots → build Power BI → run the web server → deploy → write the report.

**About the dataset:** 100,000 rows · 20 economies (incl. EU) · 60 months (Jan 2020 – Dec 2024) · 9 numeric indicators · no missing values, no duplicates. Each country-month has ~83 records.

> ⚠️ **Mention this in your report's "Limitations":** the dataset looks *synthetic*. For example Argentina and Turkey show ~3–3.6% average inflation, while real figures in 2022–24 were far higher. Your conclusions are about **this dataset's patterns**, not real-world policy advice.

---

## 1. Setup (one time)

1. Install **Python 3.10+** (python.org). On Windows, tick **"Add Python to PATH"**.
2. Open a terminal in the `inflation_ml_project` folder (VS Code → File → Open Folder → Terminal → New Terminal).
3. Create a virtual environment and install the libraries:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> Re-run steps 2 and 3 below on your own machine even though the `models/` folder already exists. Saved scikit-learn models only load reliably with the **same scikit-learn version** that trained them.

---

## 2. STEP 1: Preprocessing

```bash
python 01_preprocessing.py
```

**What it does (this goes in your "Data Preprocessing Documentation"):**

| # | Step | Why |
|---|------|-----|
| 1 | Remove duplicate rows (0 found) | Duplicates bias averages |
| 2 | Standardise country codes | Clean joins |
| 3 | Convert `date` text "2020-01" → real date | Power BI time axis & time intelligence |
| 4 | Fill missing numbers with the country median (0 found) | Robust, keeps rows |
| 5 | Cap outliers at the 1st/99th percentile (winsorising) | Stops extreme values dominating K-Means distances |
| 6 | New features: `year`, `month`, `quarter`, `real_interest_rate` (= interest − inflation), `inflation_band`, `covid_phase` | Calculated fields for grouping & analysis |
| 7 | **Blend** with `country_metadata.csv` → `country_name`, `region`, `economy_type`, `latitude`, `longitude` | Data-integration requirement + maps |

**Output:** `data/inflation_cleaned.csv` (100,000 × 22) and `outputs/preprocessing_report.txt` (the *before → steps → after* text, ready to paste into the report).

---

## 3. STEP 2: K-Means Clustering

```bash
python 02_kmeans_clustering.py
```

### How it works
1. **Aggregate** to one row per *country-month* (20 × 60 = **1,200 points**). Each point means "India in June 2022".
2. **Scale** all 9 indicators with `StandardScaler`. K-Means measures distance, so without this `money_supply_m2` (≈12,000) would swamp `inflation_rate` (≈4).
3. **Choose k** using the **Elbow method** (inertia) and the **Silhouette score** for k = 2…8 → `kmeans_k_selection.csv` and `plots/kmeans_elbow_silhouette.png`.
4. **Fit the final model with k = 4.** k=2 has the highest silhouette (0.25), but 2 groups are too coarse to interpret. The elbow bends around k = 3–4, and k = 4 gives 4 clusters that each mean something clear. *Say this in the viva.* To try another k, change `FINAL_K` at the top of the script.
5. **Name clusters automatically:** they're ranked by average inflation, then labelled with each cluster's most distinctive indicator.
6. **PCA** squeezes the 9 dimensions into 2 (`pca_x`, `pca_y`) so the clusters can be drawn on a scatter plot.

### Output: what you get

| File | One row = | Key columns | Use in Power BI |
|---|---|---|---|
| `kmeans_country_month.csv` | country-month | `cluster_id`, `cluster_name`, `pca_x`, `pca_y`, 9 indicators, lat/long | scatter, ribbon, matrix |
| `kmeans_country_summary.csv` | country | `dominant_cluster`, `share_in_dominant`, lat/long | **cluster map** |
| `kmeans_cluster_profiles.csv` | cluster | average of every indicator | profile chart / heat map |
| `kmeans_k_selection.csv` | k value | `inertia`, `silhouette` | elbow line chart |

### Your results (k = 4)

| Cluster | Avg inflation | What defines it | When it happens |
|---|---|---|---|
| **C1: Lowest Inflation, High Interest Rate** | 2.97% | interest rate 3.94% (highest) | spread across 2021–24; dominant in USA, GBR, TUR, PHL, ARG |
| **C2: Below-avg Inflation, High Supply-Chain Stress** | 4.15% | oil $40.7, supply-chain index 3.47, unemployment 6.4% | **almost entirely 2020**: the COVID shock |
| **C3: Above-avg Inflation, High Food Prices** | 4.35% | GDP growth 3.7%, food index 135.7 | mostly **2023–24**: the recovery phase |
| **C4: Highest Inflation, High Oil Price** | 5.08% | oil $93.4, lowest interest rate 1.77% | mostly **2021–22**: the energy-price surge |

**Interpretation:** The clusters recover the post-COVID story *without being told about dates*: shock (2020) → oil-driven surge (2021–22) → recovery (2023–24). Countries that kept interest rates high (C1) had the lowest inflation.

---

## 4. STEP 3: HistGradientBoosting Regressor

```bash
python 03_hgb_regression.py      # ~30 seconds
```

### How it works
- **Target:** `inflation_rate`
- **Features (11):** `country` (categorical, handled natively, no one-hot needed), `year`, `month`, `interest_rate`, `oil_price`, `gdp_growth`, `unemployment_rate`, `money_supply_m2`, `exchange_rate_usd`, `food_price_index`, `supply_chain_index`
- **Split:** 80% train (80,000 rows) / 20% test (20,000 rows), `random_state=42`
- **Model:** boosted decision trees on histogram-binned features (fast on 100k rows). Early stopping is on. Hyper-parameters are listed in the script.
- **Evaluation:** MAE, RMSE, R² on train & test; 5-fold cross-validation; a "predict the mean" baseline for comparison.
- **Feature importance:** permutation importance, i.e. how much R² drops when a feature's values are shuffled.

### Your results

| Metric | Train | Test |
|---|---|---|
| MAE | 0.388 | **0.387** percentage points |
| RMSE | 0.489 | **0.489** |
| R² | 0.885 | **0.883** |
| 5-fold CV R² | | 0.882 ± 0.002 |
| Baseline MAE (always predict the mean) | | 1.164 |

**Interpretation:** Predictions are on average **±0.39 percentage points** from the true inflation rate, which is **3× better than the baseline**. Train ≈ test, so there is **no overfitting**. The model explains **88%** of the variation in inflation.

**Feature importance (drop in R²):** country 0.86 ≫ oil price 0.24 > interest rate 0.20 > supply-chain index 0.16 > others < 0.07.
→ *Country-specific structure matters most. Among the economic drivers, **oil prices**, **interest rates** (inverse) and **supply-chain stress** drive post-COVID inflation, which matches the K-Means story.*

### Output: what you get

| File | One row = | Key columns | Use in Power BI |
|---|---|---|---|
| `hgb_predictions.csv` | test record (20,000) | `actual_inflation`, `predicted_inflation`, `error` (actual − predicted), `abs_error`, `error_band`, country, date, lat/long | actual-vs-predicted, error histogram, decomposition tree |
| `hgb_metrics.csv` | metric | `train`, `test` | KPI cards (or use the DAX measures) |
| `hgb_feature_importance.csv` | feature | `importance`, `std` | bar chart |
| `hgb_country_error.csv` | country | `MAE`, `RMSE`, avg actual/predicted | error map |
| `models/hgb_model.joblib` | the trained model | | web app |

---

## 5. Python visualisations (for the report)

The scripts already save these to `outputs/plots/`. Paste them into your report under "ML Results":

| Plot | What it shows |
|---|---|
| `kmeans_elbow_silhouette.png` | Why k = 4 |
| `kmeans_pca_scatter.png` | Clusters separated in 2-D |
| `kmeans_profile_heatmap.png` | What each cluster means |
| `hgb_actual_vs_predicted.png` | Points hug the red diagonal → good fit |
| `hgb_error_histogram.png` | Errors centred at 0, symmetric → no bias |
| `hgb_feature_importance.png` | Drivers of inflation |
| `hgb_trend.png` | Predicted line tracks actual over time |

The interactive versions of all of these go in Power BI (next section).

---

## 6. Power BI: Step by Step

### 6.1 Import data
**Home → Get data → Text/CSV**, and load each of these (click **Load**, not Transform, unless a type is wrong):

1. `data/inflation_cleaned.csv`
2. `data/country_metadata.csv`
3. `outputs/kmeans_country_month.csv`
4. `outputs/kmeans_country_summary.csv`
5. `outputs/kmeans_cluster_profiles.csv`
6. `outputs/kmeans_k_selection.csv`
7. `outputs/hgb_predictions.csv`
8. `outputs/hgb_metrics.csv`
9. `outputs/hgb_feature_importance.csv`
10. `outputs/hgb_country_error.csv`

In Power Query (Transform data), check that every `date` column is type **Date**, `latitude`/`longitude` are **Decimal**, and numbers are **Decimal number**.

**Data category for maps:** select `country_metadata[latitude]` → Column tools → Data category **Latitude**; do the same for longitude → **Longitude**, and `country_name` → **Country/Region**.

### 6.2 Create the Calendar table and the Top-N parameter
Copy them from `powerbi_dax_measures.txt` (top section). Mark `Calendar` as the date table.

### 6.3 Relationships (Model view)
Drag to create these. All are **single direction, 1 → many** except where noted:

| From (1 side) | To (many side) |
|---|---|
| `country_metadata[country]` | `inflation_cleaned[country]` |
| `country_metadata[country]` | `kmeans_country_month[country]` |
| `country_metadata[country]` | `hgb_predictions[country]` |
| `country_metadata[country]` | `kmeans_country_summary[country]` (1:1) |
| `country_metadata[country]` | `hgb_country_error[country]` (1:1) |
| `Calendar[Date]` | `inflation_cleaned[date]` |
| `Calendar[Date]` | `kmeans_country_month[date]` |
| `Calendar[Date]` | `hgb_predictions[date]` |

This is the **data blending** part: a single Country slicer or Year slicer filters every page.
**Always use `country_metadata` columns (country_name, region, lat/long) and `Calendar` columns in slicers and axes**, not the copies inside the fact tables. Otherwise cross-filtering breaks.

### 6.4 Measures and calculated columns
Paste everything from `powerbi_dax_measures.txt`. Put measures in their own folder or a blank "_Measures" table to keep things tidy.

### 6.5 Groups, Bins, Sets, Top-N (requirement 4)
- **Bins (grouping a number):** right-click `inflation_cleaned[inflation_rate]` → New group → Bin, size **0.5** → used for the histogram.
- **Groups (category grouping):** right-click `country_metadata[country_name]` → New group → make "G7-like" (USA, GBR, CAN, JPN, EU) vs "BRICS+" (BRA, RUS, IND, CHN, ZAF, SAU) vs Other.
- **Set:** calculated column `High Inflation Set` (in DAX file). Power BI has no "sets" like Tableau, so this in/out column is the equivalent. Say that in the viva.
- **Top-N:** (a) Filters pane → `country_name` → Filter type **Top N** → Top 5 by `Avg Inflation %`; and (b) dynamic: `Show In Top N` = 1 with the **Top N slicer** parameter.
- **Calculated fields:** `Oil Price Band`, `Policy Stance`, `real_interest_rate`.

---

### 6.6 Dashboard pages and which visual to pick

Use a consistent theme (View → Themes) and the same cluster colours on every page (Format → Colors → set C1…C4 once).

#### PAGE 1: Overview (Interactive Dashboard)
| Visual (Visualizations pane) | Fields |
|---|---|
| **Card ×4** (KPIs) | `Avg Inflation %`, `Max Inflation %`, `Avg Real Interest Rate %`, `Inflation YoY Change (pp)` |
| **Slicers** | `Calendar[Year]` (tile style), `country_metadata[region]` (dropdown), `economy_type`, `Top N` parameter |
| **Line chart** | X: `Calendar[Date]` (month level) · Y: `Avg Inflation %` · Secondary Y: `Avg Oil Price`. Inflation follows oil. |
| **Clustered bar chart (Top-N)** | Y: `country_name` · X: `Avg Inflation %` · Filter: `Show In Top N = 1` · Title: fx → `Dynamic Title` |
| **Donut chart** | Legend: `inflation_band` · Values: `Records` |
| **Stacked column chart** | X: `Calendar[Year]` · Y: `Records` · Legend: `inflation_band` |
| **Treemap** | Category: `region` → `country_name` · Values: `Avg Inflation %` |
Turn on **drill-down** on the line chart (Year → Quarter → Month) and the treemap (Region → Country).

#### PAGE 2: Exploratory and Advanced
| Visual | Fields | Why |
|---|---|---|
| **Scatter chart** | X: `Avg Interest Rate %`, Y: `Avg Inflation %`, Values/Details: `country_name`, Size: `Avg GDP Growth %`, Play axis: `Calendar[Year]` | Advanced scatter: higher rates go with lower inflation, animated over time |
| **Matrix (heat map)** | Rows: `country_name` · Columns: `Calendar[Year]` · Values: `Avg Inflation %` → Format → Cell elements → **Background color** (gradient) | Advanced viz: heat map |
| **Ribbon chart** | X: `Calendar[Year]` · Y: `Avg Inflation %` · Legend: `region` | Advanced viz: rank changes |
| **Waterfall chart** | Category: `Calendar[Year]` · Y: `Inflation YoY Change (pp)` | Advanced viz: year-on-year change |
| **Column chart (Histogram)** | X: `inflation_rate (bins)` · Y: `Records` | Distribution |
| **Box plot** *(optional)* | Get more visuals → "Box and Whisker chart by MAQ" · Category `region`, Value `inflation_rate` | Advanced viz: spread |

#### PAGE 3: Geo-Spatial
| Visual | Fields |
|---|---|
| **Map (bubble map)** | Latitude/Longitude: `country_metadata` lat/long · Bubble size: `Avg Inflation %` · Legend: `region` · Tooltips: `Avg Interest Rate %`, `Avg Oil Price` |
| **Filled map** | Location: `country_name` · Color saturation (Format → Fill colors → fx gradient): `Avg Inflation %` *(EU won't fill since it isn't a country; mention this or rely on the bubble map)* |
| **Slicer** | `Calendar[Year]` to watch the map change over 2020 → 2022 → 2024 |

**Two spatial observations you can write:**
1. Highest average inflation: **India (5.5%)**, **South Africa (5.2%)**, **South Korea (5.1%)**, **Saudi Arabia (5.1%)**. Africa is the highest region (5.15%).
2. Lowest: **Philippines (2.4%)**, **Thailand (2.9%)**, **Argentina (3.2%)**, **USA (3.3%)**, and South America is the lowest region (3.6%). Several of these (USA, PHL, ARG) are high-interest-rate economies.
3. *(bonus)* Advanced (4.23%) vs Emerging (4.15%) economies differ very little. Geography and policy matter more than development level.

#### PAGE 4: K-Means Results (ML visualisation)
| Visual | Fields | What it shows |
|---|---|---|
| **Map** ⭐ | Lat/Long from `country_metadata` · **Legend: `kmeans_country_summary[dominant_cluster]`** · Size: `share_in_dominant` | "Geographical clusters using cluster categories" (exactly what the instructions ask for) |
| **Scatter chart** | X: `pca_x` · Y: `pca_y` · Legend: `cluster_name` · Values: `kmeans_country_month[country]` + `date` (so every point is plotted) | Cluster separation |
| **Ribbon / 100% stacked column** | X: `Calendar[Year]` · Y: `Cluster Country-Months` · Legend: `cluster_name` | Clusters shifting over time: C2 (2020) → C4 (2021–22) → C3 (2023–24) |
| **Matrix heat map** | Rows: `kmeans_cluster_profiles[cluster_name]` · Values: each indicator · conditional background colour | Cluster profiles |
| **Line chart** | X: `kmeans_k_selection[k]` (set to *Categorical*) · Y: `inertia`; second line chart with `silhouette` | Elbow method |
| **Donut** | Legend: `cluster_name` · Values: `Cluster Country-Months` | Cluster sizes |
| **Card** | `Number of Clusters`, `Best Silhouette` | |
| **Slicer** | `cluster_name` | Click a cluster → map & scatter cross-filter |

#### PAGE 5: HistGradientBoosting Results (ML visualisation)
| Visual | Fields | What it shows |
|---|---|---|
| **Cards ×4** | `R2 Score`, `MAE`, `RMSE`, `Within 0.5pp %` | Model accuracy KPIs |
| **Gauge** | Value: `R2 Score` · Min 0 · Max 1 · Target 0.8 | Advanced viz |
| **Scatter: Actual vs Predicted** ⭐ | X: `actual_inflation` · Y: `predicted_inflation` · Values: `hgb_predictions[date]` + `country` · Analytics pane → **Y = X reference line** (or add a constant line) | Points on the diagonal = correct |
| **Line chart** | X: `Calendar[Date]` · Y: `Avg Actual %` and `Avg Predicted %` | Two lines overlapping |
| **Column chart (error histogram)** | Right-click `error` → New group → Bin size 0.1 → X: `error (bins)` · Y: `Test Rows` | Prediction error centred at 0 |
| **Bar chart** | Y: `hgb_feature_importance[feature]` · X: `importance` (sort desc) | Drivers of inflation |
| **Map** | Lat/Long · Size/colour: `hgb_country_error[MAE]` | Where the model is least accurate |
| **Decomposition tree** | Analyze: `MAE` · Explain by: `region`, `country_name`, `covid_phase`, `error_band` | Advanced viz: drill into the error |
Clicking a country anywhere will **recalculate R²/MAE/RMSE for that country** (dynamic DAX).

### 6.7 Interactivity checklist (requirement 5)
- **Cross-filtering:** on by default. Format → Edit interactions to choose filter vs highlight.
- **Drill-down:** enable on hierarchies (Region → Country, Year → Month).
- **Drill-through:** create a page "Country Detail" → add `country_name` to the *Drill-through* well → right-click any country on another page → Drill through.
- **Tooltips:** add extra fields to every visual's Tooltips well. Optionally build a *Tooltip page* (Page info → Allow use as tooltip).
- **Bookmarks + buttons:** "Reset filters" button. Page navigator for the 5 pages.
- Meaningful **titles**, **legends** on, axis titles with **units (%)**.

### 6.8 Publish (deliverable 7)
1. **Home → Publish** → sign in with your college Microsoft account → choose *My workspace*.
2. In app.powerbi.com: open the report → **File → Embed report → Publish to web (public)** → copy the link. If your college tenant blocks this, use **Share** (link) or **Export → PDF**.
3. Also save the `.pbix` (Deliverable 4: the project file).

---

## 7. STEP 4: Run the Model on a Web Server

### 7.1 Run locally
```bash
python app.py
```
Open **http://127.0.0.1:5000**. Pick a country, year, month, and enter the 8 indicators, then click **Predict inflation**.
You get:
- **Predicted inflation %** (HistGradientBoosting)
- a likely range (± RMSE)
- the **economic regime cluster** (K-Means, using the indicators + predicted inflation)
- the model's test R², MAE, RMSE

(See `outputs/plots/web_app_screenshot.png`.)

**Try a "2022 energy shock" for India:** oil 110, interest 1.5, supply-chain 2.5 → ≈ 6.6% → *C4: Highest Inflation, High Oil Price*.

### 7.2 JSON API (show this in the demo)
```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d "{\"country\":\"IND\",\"year\":2022,\"month\":6,\"interest_rate\":1.5,\"oil_price\":110,\"gdp_growth\":3,\"unemployment_rate\":5.5,\"money_supply_m2\":12000,\"exchange_rate_usd\":75,\"food_price_index\":125,\"supply_chain_index\":2.5}"
```
Response:
```json
{"country":"IND","country_name":"India","predicted_inflation":6.63,"expected_range":[6.14,7.12],
 "cluster_id":0,"cluster_name":"C4: Highest Inflation, High Oil Price"}
```
Or use Postman / Thunder Client (VS Code extension).

### 7.3 Deploy online for free (Render.com)
1. Make a **GitHub** repo and upload the whole folder (including `models/`, `outputs/hgb_metrics.csv`, `data/country_metadata.csv`). If a file is >100 MB GitHub rejects it, but none here are that big.
   ```bash
   git init && git add . && git commit -m "inflation ML app"
   git branch -M main
   git remote add origin https://github.com/<you>/inflation-ml.git
   git push -u origin main
   ```
2. **Pin versions:** run `pip freeze > requirements.txt` on the machine where you trained the models, so the server uses the same scikit-learn version.
3. Go to **render.com** → sign in with GitHub → **New → Web Service** → select the repo.
   - Runtime: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
   - Instance: **Free**
4. Wait for "Live", then open `https://<your-app>.onrender.com`. Put this link in the report.
   (The free tier sleeps after inactivity, so the first load takes ~30–50 s. Open it before your demo.)

**Alternative: PythonAnywhere** (free): upload the folder → Web tab → Add new web app → Flask → set the source path, and in the WSGI file use `from app import app as application` → install requirements in a Bash console → Reload.

---

## 8. Report: Insights and Interpretation (section 10)

1. **Three post-COVID phases appear without dates being given to the model:** the 2020 shock (oil ~$40, supply-chain stress 3.5, unemployment up), the 2021–22 surge (oil ~$110 in 2022, avg inflation peaks at 4.7%), and 2023–24 recovery (GDP growth 3.7%, inflation back to ~4.1%).
2. **Monetary policy matters:** the cluster with the highest interest rates (C1, 3.9%) has the lowest inflation (3.0%); correlation interest ↔ inflation = −0.21; interest rate is the #2 economic driver in the regressor.
3. **Energy prices are the biggest external driver:** oil price is the top economic feature (permutation importance 0.24) and defines the highest-inflation cluster.
4. **The regressor is reliable:** test R² 0.883, MAE 0.39 pp, 3× better than a naive baseline, no overfitting, and errors are similar across all 20 countries (MAE 0.37–0.40).
5. **Spatial:** see the two observations in 6.6 / Page 3.
6. **Limitations:** synthetic-looking data; random split (a time-based split, e.g. train 2020–23 and test 2024, is a good "future work" point); K-Means assumes round clusters (silhouette is modest at 0.14 for k = 4, so the clusters overlap).

---

## 9. Likely viva questions

- **Why K-Means?** Unlabelled data. We want natural groups of economic conditions. Simple and interpretable.
- **Why scale before K-Means?** It uses Euclidean distance, so large-unit features would dominate.
- **How did you choose k?** Elbow + silhouette. k = 2 scores best but is too coarse; k = 4 sits at the elbow and is interpretable.
- **Why HistGradientBoosting and not Random Forest?** It's much faster on 100k rows (bins features into 256 histograms), handles categorical features natively, has built-in early stopping, and boosting usually beats bagging on tabular data.
- **What is boosting?** Trees are added one after another, each fitting the *errors* (residuals) of the previous ones; the learning rate shrinks each tree's contribution.
- **MAE vs RMSE?** MAE is the average absolute miss. RMSE squares errors, so it punishes big misses more. RMSE ≥ MAE always.
- **What does R² = 0.88 mean?** 88% of the variance in inflation is explained by the features.
- **Permutation importance?** Shuffle one feature's values and measure how much R² drops. A bigger drop means a more important feature.
- **How is the model served?** It's saved with joblib, loaded once by Flask, and predicts via an HTML form and a JSON endpoint, run by gunicorn on Render.
