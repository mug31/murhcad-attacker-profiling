"""
Stage 5 — K-Means clustering (k=6) and PCA projection.

CHANGES FROM PREVIOUS VERSION
  1. Typology labels are no longer derived by a heuristic inside this script.
     They are imported from cluster_labels.py, which mirrors Table 4.1 verbatim.
     The old heuristic produced 'Mixed/Others Scanner' / 'SIP Scanner' etc.,
     which is the root cause of the label conflict between Figures 4.4/4.5 and
     Table 4.1.
  2. A cluster identity check runs before labels are applied, so a permuted
     cluster ID cannot silently mislabel every figure.
  3. All figure text and console output is in English.
  4. Figures saved at 300 dpi (print requirement).

Clustering parameters are UNCHANGED (k=6, random_state=42, n_init=20,
max_iter=500) so all published numbers still reproduce.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')

from cluster_labels import CLUSTER_LABELS, verify_cluster_identity, canonicalise_cluster_ids

DPI = 300   # was 150; raised for print. Lower it back if file size matters.

# -- Load prep objects ------------------------------------------------------
with open('prep_objects.pkl', 'rb') as f:
    prep = pickle.load(f)

df_clust  = prep['df_clust']
X_scaled  = prep['X_scaled']
FEAT_COLS = prep['FEAT_COLS']
le_org    = prep['le_org']
le_cty    = prep['le_cty']

# -- K-Means k=6 ------------------------------------------------------------
# NOTE: K-Means runs on X_scaled — the 13 standardised features. The PCA below
# is used ONLY to project the result into 2D for plotting; it is never an input
# to the clustering. This is the ambiguity flagged in review item 4.2.
K = 6
km = KMeans(n_clusters=K, random_state=42, n_init=20, max_iter=500)
raw_ids = km.fit_predict(X_scaled)
df_clust['cluster_raw'] = raw_ids
# Remap arbitrary K-Means IDs to Table 4.1's canonical C0..C5 numbering.
df_clust['cluster'], _map = canonicalise_cluster_ids(
    df_clust.assign(cluster=raw_ids), X_scaled, km, cluster_col='cluster')

print("=" * 60)
print(f"K-Means  k={K}  |  n={len(df_clust)} IPs  |  inertia={km.inertia_:,.0f}")
print(f"Input space: {X_scaled.shape[1]} standardised features (srcOrg excluded per ablation; not PCA)")
print("=" * 60)

# -- 1. IP distribution per cluster ----------------------------------------
print("\n-- 1. IP distribution per cluster --")
dist = df_clust['cluster'].value_counts().sort_index()
for c, n in dist.items():
    print(f"  Cluster {c}: {n:4d} IPs  ({n/len(df_clust)*100:.1f}%)")

# -- 2. Mean profile per cluster -------------------------------------------
print("\n-- 2. Mean profile per cluster --")
PROFILE_COLS = [
    'total_events', 'pct_sip', 'pct_smbd', 'pct_ssh',
    'pct_telnet', 'dominant_hour', 'hour_entropy', 'active_hours'
]
profile = df_clust.groupby('cluster')[PROFILE_COLS].mean().round(3)
print(profile.to_string())

# -- 3. Top 3 countries and orgs per cluster -------------------------------
print("\n-- 3. Top 3 source countries and organisations per cluster --")
for c in range(K):
    sub = df_clust[df_clust['cluster'] == c]
    top_cty = sub['srcCountryName'].value_counts().head(3)
    top_org = sub['srcOrg_grouped'].value_counts().head(3)
    print(f"\n  Cluster {c} (n={len(sub)}):")
    print(f"    Countries : {', '.join([f'{n}({v})' for n, v in top_cty.items()])}")
    print(f"    Orgs      : {', '.join([f'{n}({v})' for n, v in top_org.items()])}")

# -- 4. Typology labels ----------------------------------------------------
# Labels come from Table 4.1 via cluster_labels.py. Verify first that cluster
# IDs still correspond to the groups those labels describe.
verify_cluster_identity(df_clust, cluster_col='cluster', strict=False)

df_clust['typology'] = df_clust['cluster'].map(CLUSTER_LABELS)

print("-- 4. Typology per cluster (source: Table 4.1) --")
proto_profile = df_clust.groupby('cluster')[
    ['pct_sip', 'pct_smbd', 'pct_ssh', 'pct_telnet', 'pct_others']
].mean()
for c in range(K):
    p, pp = profile.loc[c], proto_profile.loc[c]
    dom = pp.idxmax()
    print(f"  Cluster {c} -> [{CLUSTER_LABELS[c]}]  (dominant: {dom}={pp[dom]:.2f}, "
          f"mean_events={p['total_events']:.0f}, active_hrs={p['active_hours']:.1f})")

# -- 5. PCA 2D scatter -----------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

# Light-mode palette, matched to the Stage-4 charts for a consistent look
# across the manuscript. Colours chosen to stay distinguishable in greyscale
# print and for common forms of colour-vision deficiency.
PALETTE = ['#2a78d6',   # C0 blue
           '#1baf7a',   # C1 green
           '#d1495b',   # C2 red
           '#eda100',   # C3 amber
           '#8a5fbf',   # C4 purple
           '#0f8b8d']   # C5 teal
CLUSTER_COLORS = {c: PALETTE[c] for c in range(K)}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

for c in range(K):
    mask = df_clust['cluster'] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=CLUSTER_COLORS[c], label=f"C{c}: {CLUSTER_LABELS[c]}",
               alpha=0.55, s=26, edgecolors='none', zorder=3)

centroids_pca = pca.transform(km.cluster_centers_)
canon_to_raw = {canon: raw for raw, canon in _map.items()}
for c in range(K):
    raw = canon_to_raw[c]           # centroid rows are in RAW id order
    ax.scatter(centroids_pca[raw, 0], centroids_pca[raw, 1],
               c=CLUSTER_COLORS[c], s=260, marker='*',
               edgecolors='#333333', linewidths=1.0, zorder=6)
    # Per-cluster label offsets: C2 and C4 centroids sit almost on top of each
    # other, so a single fixed offset makes their labels collide ("C4C2").
    LABEL_OFFSET = {0: (8, 6), 1: (-20, 6), 2: (8, -14),
                    3: (8, 6), 4: (-20, 6), 5: (8, 6)}
    ax.annotate(f'C{c}', (centroids_pca[raw, 0], centroids_pca[raw, 1]),
                fontsize=10, fontweight='bold', color='#222222',
                xytext=LABEL_OFFSET[c], textcoords='offset points', zorder=7)

ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}% of variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}% of variance)', fontsize=11)
ax.set_title(
    'K-Means clusters (k=6) — 2D PCA projection\n'
    f'MURHCAD honeypot  |  n={len(df_clust)} IPs (total_events > 1)\n'
    'PCA for visualisation only; clustering ran on 12 standardised features',
    fontsize=12.5, fontweight='500', pad=12)
ax.tick_params(labelsize=10)
ax.grid(True, color='#e1e0d9', lw=0.8)
ax.set_axisbelow(True)

legend = ax.legend(loc='upper right', framealpha=0.95, facecolor='white',
                   edgecolor='#cccccc', fontsize=9,
                   title='Cluster', title_fontsize=10)
legend.get_title().set_fontweight('bold')

fig.tight_layout()
plt.savefig('pca_clusters.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print("\nFigure saved: pca_clusters.png")

# -- Save results ----------------------------------------------------------
# NOTE: column 'tipologi' has been renamed to 'typology' in this version.
out_cols = [
    'srcIp', 'total_events', 'unique_dst_ports', 'unique_dst_hosts',
    'pct_sip', 'pct_smbd', 'pct_ssh', 'pct_telnet', 'pct_others',
    'dominant_hour', 'hour_entropy', 'active_hours',
    'srcCountryName', 'srcOrg', 'srcOrg_grouped',
    'cluster', 'typology'
]
df_clust[out_cols].to_csv('clustered_ips.csv', index=False)
print("Results saved : clustered_ips.csv")
print(f"\nPCA variance explained: PC1={var_exp[0]*100:.1f}%  "
      f"PC2={var_exp[1]*100:.1f}%  total={sum(var_exp)*100:.1f}%")