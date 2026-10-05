"""
Stage 5 — feature ablation study (Sub-bab 5.2 / RQ2).

Tests the contribution of the two identity-attribute features to the final
K-Means partition:
  1. srcOrg_enc  (network organisation) — ADDED to the 12-feature final set.
  2. srcCountry_enc (country) — REMOVED from the 12-feature final set.

Recipe (must match exactly for numbers to be comparable):
  - Take the column list, re-fit a fresh StandardScaler on exactly those
    columns (never slice a pre-scaled matrix — that silently changes the
    per-feature mean/variance used for scaling and gives different numbers).
  - KMeans(n_clusters=6, random_state=42, n_init=20, max_iter=500) — same
    hyperparameters as the final model in tahap5_kmeans.py, NOT the k-sweep
    parameters in tahap5_clustering_prep.py (n_init=10, max_iter=300).
  - silhouette_score with sample_size=min(5000, n), random_state=42, to
    match how the elbow/silhouette figure was scored.
  - Compare cluster membership to the final 12-feature partition via
    adjusted_rand_score.

This script requires prep_objects.pkl, produced by tahap5_clustering_prep.py.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import scipy.stats as ss

with open('prep_objects.pkl', 'rb') as f:
    prep = pickle.load(f)

df_clust = prep['df_clust']
FEAT_COLS = prep['FEAT_COLS']   # the 12 final features


def fit(cols, n_init=20, max_iter=500):
    X = StandardScaler().fit_transform(df_clust[cols].values)
    km = KMeans(n_clusters=6, random_state=42, n_init=n_init, max_iter=max_iter)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
    return labels, sil


print("=" * 60)
print("ABLATION STUDY — feature contribution to K-Means partition")
print("=" * 60)

base_labels, base_sil = fit(FEAT_COLS)
print(f"\nBaseline (12 features, final model): silhouette = {base_sil:.4f}")

# 1. srcOrg_enc ablation: what happens if we ADD it back in (13 features)?
org_labels, org_sil = fit(FEAT_COLS + ['srcOrg_enc'])
ari_org = adjusted_rand_score(base_labels, org_labels)
print(f"\n+ srcOrg_enc (13 features): silhouette = {org_sil:.4f} "
      f"(baseline was {org_sil:.4f} -> {base_sil:.4f} when removed)")
print(f"  Adjusted Rand Index vs 12-feature partition: {ari_org:.4f}")

# 2. srcCountry_enc ablation: what happens if we REMOVE it (11 features)?
no_country_cols = [c for c in FEAT_COLS if c != 'srcCountry_enc']
cty_labels, cty_sil = fit(no_country_cols)
ari_cty = adjusted_rand_score(base_labels, cty_labels)
print(f"\n- srcCountry_enc (11 features): silhouette = {cty_sil:.4f}")
print(f"  Adjusted Rand Index vs 12-feature partition: {ari_cty:.4f}")

# 3. Cramer's V: cluster membership vs srcOrg_grouped (descriptive attribute)
ct = pd.crosstab(pd.Series(base_labels, name='cluster'), df_clust['srcOrg_grouped'])
chi2 = ss.chi2_contingency(ct)[0]
n = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
print(f"\nCramer's V (cluster ~ srcOrg_grouped): {cramers_v:.4f}")

others_share = (
    df_clust.assign(cluster=base_labels)
    .groupby('cluster')['srcOrg_grouped']
    .apply(lambda s: (s == 'others').mean() * 100)
)
print("\n'others' share of srcOrg_grouped per cluster:")
print(others_share.round(1).to_string())

print("\n" + "=" * 60)
print("SUMMARY FOR SUB-BAB 5.2")
print("=" * 60)
print(f"srcOrg_enc removed:     silhouette {org_sil:.3f} -> {base_sil:.3f}, "
      f"ARI = {ari_org:.3f}")
print(f"srcCountry_enc removed: silhouette {base_sil:.3f} -> {cty_sil:.3f}, "
      f"ARI = {ari_cty:.3f}")
print(f"Cramer's V (cluster ~ srcOrg_grouped): {cramers_v:.3f}")