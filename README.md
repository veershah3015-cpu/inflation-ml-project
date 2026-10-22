# Post-COVID Global Inflation Analysis

**Trends, clusters and predictions using Power BI, K-Means and HistGradientBoosting**

Data Visualization Mini Project · B.Tech Artificial Intelligence · MPSTME, NMIMS
Team: [Name 1] · [Name 2] · [Name 3]  |  Guide: [Faculty name]

- **Live web app:** https://inflation-ml-project.vercel.app
- **Power BI dashboard:** [paste Power BI "Publish to web" link here]

---

## Problem statement
After COVID-19, inflation rose sharply worldwide, but not equally. This project answers:
1. Which economies and months behaved alike? → **K-Means clustering**
2. What drives inflation the most, and can it be predicted? → **HistGradientBoosting regressor**
3. How can the results be shared? → **Power BI dashboard** + **deployed web app**

## Dataset
[Global Inflation Post-COVID (Kaggle)](https://www.kaggle.com/): 100,000 records · 20 economies (incl. EU) · Jan 2020 – Dec 2024 · 9 indicators (inflation, interest rate, oil price, GDP growth, unemployment, money supply, exchange rate, food price index, supply-chain stress).
Blended with `country_metadata.csv` (country name, region, economy type, latitude, longitude).

## Project structure
```
├── data/
│   ├── global_inflation_post_covid.csv   original dataset
│   ├── country_metadata.csv              second dataset (blending)
│   └── inflation_cleaned.csv             preprocessed dataset
├── 01_preprocessing.py                   cleaning + feature engineering + blending
├── 02_kmeans_clustering.py               K-Means (k = 4)
├── 03_hgb_regression.py                  HistGradientBoosting regressor
├── inflation_ml_notebook.ipynb           same pipeline as a notebook
├── app.py, templates/index.html          Flask web app
├── models/                               trained models (.joblib)
├── outputs/                              result CSVs for Power BI + plots/
├── requirements.txt                      Python libraries (pinned)
└── .vercelignore                         files excluded from deployment
```

## Results
| Model | Result |
|---|---|
| K-Means (k = 4, chosen by elbow + silhouette) | 4 economic regimes; clusters recover the COVID timeline (2020 shock → 2021–22 oil surge → 2023–24 recovery) without using dates |
| HistGradientBoosting | Test R² **0.883** · MAE **0.39** pp · RMSE **0.49** · 3× better than mean baseline · no overfitting (train R² 0.885) |
| Top drivers | Oil price, interest rate, supply-chain stress |

## How to run locally
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python 01_preprocessing.py
python 02_kmeans_clustering.py
python 03_hgb_regression.py
python app.py                   # open http://127.0.0.1:5000
```

## Deployment
Deployed on Vercel with the Vercel CLI (`vercel --prod`). Flask is detected automatically from `app.py`.

## Tools
Python (pandas, scikit-learn, matplotlib, Flask) · Power BI Desktop · Vercel · VS Code
