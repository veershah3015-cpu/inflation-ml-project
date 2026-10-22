"""
STEP 2 - K-MEANS CLUSTERING  (unsupervised: group similar economic situations)

Question answered: "Which countries/months share a similar post-COVID economic profile?"

Unit of clustering: one row per COUNTRY-MONTH (20 countries x 60 months = 1,200 rows),
made by averaging the raw records, so each point is "country X in month Y".

Input : data/inflation_cleaned.csv
Output: outputs/kmeans_country_month.csv   -> each country-month + its cluster + PCA x/y   (Power BI)
        outputs/kmeans_country_summary.csv -> each country's dominant cluster + lat/long  (Power BI map)
        outputs/kmeans_cluster_profiles.csv-> average indicator values per cluster        (Power BI)
        outputs/kmeans_k_selection.csv     -> elbow (inertia) + silhouette for k=2..8      (Power BI / report)
        models/kmeans_model.joblib, models/kmeans_scaler.joblib                           (web app)
        outputs/plots/*.png                                                               (report)
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

FEATURES = ["inflation_rate", "interest_rate", "oil_price", "gdp_growth",
            "unemployment_rate", "money_supply_m2", "exchange_rate_usd",
            "food_price_index", "supply_chain_index"]
FINAL_K = 4          # change after looking at the elbow / silhouette output
RANDOM_STATE = 42

# ---------- 1. Load & aggregate ----------
df = pd.read_csv("data/inflation_cleaned.csv")
meta_cols = ["country", "country_name", "region", "economy_type", "latitude", "longitude"]
cm = (df.groupby(["country", "date"], as_index=False)[FEATURES].mean()
        .merge(df[meta_cols].drop_duplicates(), on="country"))
print("Country-month rows:", cm.shape)

# ---------- 2. Scale (K-Means uses distance, so all features must be on the same scale) ----------
scaler = StandardScaler()
X = scaler.fit_transform(cm[FEATURES])

# ---------- 3. Choose k: elbow method + silhouette score ----------
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    rows.append({"k": k, "inertia": round(km.inertia_, 2),
                 "silhouette": round(silhouette_score(X, km.labels_), 4)})
ksel = pd.DataFrame(rows)
print(ksel.to_string(index=False))
ksel.to_csv("outputs/kmeans_k_selection.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ksel.k, ksel.inertia, "o-"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia (WCSS)")
ax[1].plot(ksel.k, ksel.silhouette, "o-", color="tab:orange"); ax[1].set(title="Silhouette score", xlabel="k", ylabel="Silhouette")
for a in ax: a.axvline(FINAL_K, ls="--", color="grey")
plt.tight_layout(); plt.savefig("outputs/plots/kmeans_elbow_silhouette.png", dpi=150); plt.close()

# ---------- 4. Final model ----------
kmeans = KMeans(n_clusters=FINAL_K, n_init=10, random_state=RANDOM_STATE).fit(X)
cm["cluster_id"] = kmeans.labels_
print(f"\nFinal k={FINAL_K}, silhouette={silhouette_score(X, kmeans.labels_):.3f}")

# ---------- 5. Give clusters readable names (ranked by average inflation) ----------
prof = cm.groupby("cluster_id")[FEATURES].mean()
order = prof["inflation_rate"].sort_values().index.tolist()      # lowest -> highest inflation
level = {2: ["Lower", "Higher"],
         3: ["Low", "Medium", "High"],
         4: ["Lowest", "Below-avg", "Above-avg", "Highest"]}.get(FINAL_K, [f"Rank {i+1}" for i in range(FINAL_K)])
pretty = {"interest_rate": "Interest Rate", "oil_price": "Oil Price", "gdp_growth": "GDP Growth",
          "unemployment_rate": "Unemployment", "money_supply_m2": "Money Supply",
          "exchange_rate_usd": "Exchange Rate", "food_price_index": "Food Prices",
          "supply_chain_index": "Supply-Chain Stress"}
# z-score of each cluster's mean vs. the overall mean -> the most distinctive indicator names the cluster
zprof = (prof - cm[FEATURES].mean()) / cm[FEATURES].std()
name_map = {}
for rank, cid in enumerate(order):
    z = zprof.loc[cid].drop("inflation_rate")
    top = z.abs().idxmax()
    trait = f"{'High' if z[top] > 0 else 'Low'} {pretty[top]}"
    name_map[cid] = f"C{rank + 1}: {level[rank]} Inflation, {trait}"
cm["cluster_name"] = cm["cluster_id"].map(name_map)

# ---------- 6. PCA to 2-D so clusters can be drawn on a scatter plot ----------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pcs = pca.fit_transform(X)
cm["pca_x"], cm["pca_y"] = pcs[:, 0].round(4), pcs[:, 1].round(4)
print("PCA variance explained:", pca.explained_variance_ratio_.round(3))

# ---------- 7. Save outputs ----------
cm[FEATURES] = cm[FEATURES].round(3)
cm.to_csv("outputs/kmeans_country_month.csv", index=False)

profiles = cm.groupby("cluster_name")[FEATURES].mean().round(2)
profiles.insert(0, "num_country_months", cm.groupby("cluster_name").size())
profiles.reset_index().to_csv("outputs/kmeans_cluster_profiles.csv", index=False)
print("\nCluster profiles:\n", profiles.to_string())

summary = (cm.groupby(meta_cols)
             .agg(dominant_cluster=("cluster_name", lambda s: s.mode()[0]),
                  share_in_dominant=("cluster_name", lambda s: round((s == s.mode()[0]).mean(), 3)),
                  avg_inflation=("inflation_rate", "mean"),
                  avg_interest=("interest_rate", "mean"))
             .round(3).reset_index())
summary.to_csv("outputs/kmeans_country_summary.csv", index=False)
print("\nCountry -> dominant cluster:\n", summary[["country", "dominant_cluster", "share_in_dominant"]].to_string(index=False))

joblib.dump(kmeans, "models/kmeans_model.joblib")
joblib.dump(scaler, "models/kmeans_scaler.joblib")
joblib.dump(name_map, "models/kmeans_cluster_names.joblib")

# ---------- 8. Python visualisations (for the report) ----------
fig, ax = plt.subplots(figsize=(8, 6))
for name, g in cm.groupby("cluster_name"):
    ax.scatter(g.pca_x, g.pca_y, s=14, alpha=.7, label=name)
ax.set(title="K-Means clusters (PCA projection)", xlabel="PC1", ylabel="PC2")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("outputs/plots/kmeans_pca_scatter.png", dpi=150); plt.close()

z = (profiles[FEATURES] - profiles[FEATURES].mean()) / profiles[FEATURES].std()
fig, ax = plt.subplots(figsize=(10, 3.5))
im = ax.imshow(z.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(z)), z.index, fontsize=8)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax.text(j, i, profiles[FEATURES].iloc[i, j], ha="center", va="center", fontsize=7)
plt.colorbar(im, label="z-score vs other clusters"); ax.set_title("Cluster profiles")
plt.tight_layout(); plt.savefig("outputs/plots/kmeans_profile_heatmap.png", dpi=150); plt.close()
print("\nDone. Files written to outputs/ and models/")
