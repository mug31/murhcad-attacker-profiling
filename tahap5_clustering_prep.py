"""
Stage 5 — preprocessing, one-shot split, and k selection (elbow + silhouette).

CHANGES FROM PREVIOUS VERSION
  1. All figure text and console output is in English.
  2. Figure saved at 300 dpi.
  3. Both the chosen k (=6) and the silhouette-optimal k are annotated, so the
     figure shows the trade-off the manuscript argues for in Section 3.5 rather
     than only the statistical optimum.
  4. The k-sweep parameters are printed explicitly, because they differ from
     the final fit (see note below).

NOTHING ELSE CHANGED. In particular the sweep still uses n_init=10,
max_iter=300 while the final model in tahap5_kmeans.py uses n_init=20,
max_iter=500. Do not "harmonise" these without re-checking the manuscript:
raising the sweep to 20/500 changes the k=10 silhouette from 0.323 to 0.317,
and 0.323 is the number quoted in Section 3.5.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

DPI = 300
CHOSEN_K = 6          # k adopted in the manuscript (Section 3.5)
ELBOW_K  = 5          # inflection point identified by the elbow method
SWEEP_N_INIT, SWEEP_MAX_ITER = 10, 300

# -- 1. Load per-IP features ------------------------------------------------
df = pd.read_csv("features_per_ip.csv")
print(f"Total IPs: {len(df)}")
print(f"IPs with total_events == 1 : {(df['total_events']==1).sum()} "
      f"({(df['total_events']==1).mean()*100:.1f}%)")
print(f"IPs with total_events  > 1 : {(df['total_events']>1).sum()}")

# -- 2. Split: one-shot vs recurring ---------------------------------------
df_one   = df[df['total_events'] == 1].copy()   # analysed separately
df_clust = df[df['total_events'] >  1].copy().reset_index(drop=True)
print(f"\nClustering dataset: {len(df_clust)} rows")

# -- 3. Preprocessing -------------------------------------------------------
# 3a. Log-transform the skewed volume features
df_clust['log_total_events']     = np.log1p(df_clust['total_events'])
df_clust['log_unique_dst_ports'] = np.log1p(df_clust['unique_dst_ports'])

# 3b. Encode srcOrg_grouped (kept for descriptive use in Bab 4; NOT a model feature)
le_org = LabelEncoder()
df_clust['srcOrg_enc'] = le_org.fit_transform(df_clust['srcOrg_grouped'])

# 3c. Encode srcCountryName
le_cty = LabelEncoder()
df_clust['srcCountry_enc'] = le_cty.fit_transform(df_clust['srcCountryName'])

# 3d. Feature columns fed to K-Means
# NOTE: srcOrg_enc was REMOVED from the model after the ablation study (see
# manuscript Section 5.2): dropping it left cluster membership identical
# (adjusted Rand index = 1.0) and slightly RAISED the silhouette score. It is
# still computed above for descriptive use in Bab 4, but is not a model input.
# The model now uses 12 features. srcCountry_enc is kept (modest contribution).
FEAT_COLS = [
    'log_total_events',
    'log_unique_dst_ports',
    'unique_dst_hosts',
    'pct_sip', 'pct_smbd', 'pct_ssh', 'pct_telnet', 'pct_others',
    'hour_entropy',
    'active_hours',
    'dominant_hour',
    'srcCountry_enc',
]

X = df_clust[FEAT_COLS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nFeature matrix shape: {X_scaled.shape}")
print(f"Features: {FEAT_COLS}")

# -- 4. Elbow + silhouette --------------------------------------------------
K_RANGE = range(2, 13)
inertias, sil_scores = [], []

print(f"\nk sweep (n_init={SWEEP_N_INIT}, max_iter={SWEEP_MAX_ITER}, random_state=42)")
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42,
                n_init=SWEEP_N_INIT, max_iter=SWEEP_MAX_ITER)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(
        X_scaled, labels,
        sample_size=min(5000, len(X_scaled)), random_state=42))
    print(f"  k={k:2d} | inertia={km.inertia_:,.0f} | silhouette={sil_scores[-1]:.4f}")

# -- 5. Plot ----------------------------------------------------------------
# Light-mode, print-oriented styling to match the Stage-4 figures and the
# journal template. Annotations for the highest-silhouette k and the adopted k
# are placed on opposite sides so they cannot overlap when the two k values are
# adjacent (they are: k=7 highest vs k=6 adopted, ~0.005 apart).
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(13.5, 5))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.26)

K_LIST      = list(K_RANGE)
elbow_color = '#2a78d6'   # same blue as the Stage-4 charts
sil_color   = '#1baf7a'   # same green as the Stage-4 charts
anno_color  = '#b06000'   # dark amber, readable on white
pick_color  = '#c1272d'   # dark red for the adopted k
grid_color  = '#e1e0d9'
text_color  = '#52514e'

# - Elbow -
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('white')
ax1.plot(K_LIST, inertias, 'o-', color=elbow_color, lw=2, ms=6, zorder=3)
ax1.fill_between(K_LIST, inertias, alpha=0.07, color=elbow_color)
ax1.set_xlabel('Number of clusters (k)', fontsize=11)
ax1.set_ylabel('Inertia (WCSS)', fontsize=11)
ax1.set_title('Elbow method', fontsize=13, fontweight='500', pad=10)
ax1.tick_params(labelsize=10)
ax1.grid(True, color=grid_color, lw=0.8)
ax1.set_axisbelow(True)

ax1.axvline(ELBOW_K, color=anno_color, ls='--', lw=1.3, alpha=0.8, zorder=2)
ax1.scatter([ELBOW_K], [inertias[ELBOW_K - 2]], color=anno_color, s=110,
            zorder=5, edgecolors='white', linewidths=1.2)
ax1.annotate(f'k={ELBOW_K} (inflection)',
             xy=(ELBOW_K, inertias[ELBOW_K - 2]),
             xytext=(ELBOW_K + 1.4, inertias[ELBOW_K - 2] * 1.16),
             ha='left', va='bottom',
             color=anno_color, fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=anno_color, lw=1.2))

# - Silhouette -
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('white')
ax2.plot(K_LIST, sil_scores, 's-', color=sil_color, lw=2, ms=6, zorder=3)
ax2.fill_between(K_LIST, sil_scores, alpha=0.07, color=sil_color)

best_k_sil = K_LIST[int(np.argmax(sil_scores))]
best_sil   = max(sil_scores)
chosen_sil = sil_scores[K_LIST.index(CHOSEN_K)]

# Headroom so both callouts sit above the curve, never on top of it.
ax2.set_ylim(0, best_sil * 1.62)

# Highest silhouette -> callout to the UPPER RIGHT.
ax2.scatter([best_k_sil], [best_sil], color=anno_color, s=110, zorder=5,
            edgecolors='white', linewidths=1.2)
ax2.annotate(f'k={best_k_sil} highest (sil={best_sil:.3f})',
             xy=(best_k_sil, best_sil),
             xytext=(best_k_sil + 1.0, best_sil * 1.44),
             ha='left', va='center',
             color=anno_color, fontsize=9.5, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=anno_color, lw=1.2,
                             connectionstyle='arc3,rad=-0.15'))

# Adopted k -> callout to the UPPER LEFT, well clear of the one above.
ax2.scatter([CHOSEN_K], [chosen_sil], color=pick_color, s=130, zorder=6,
            edgecolors='white', linewidths=1.2)
ax2.annotate(f'k={CHOSEN_K} adopted (sil={chosen_sil:.3f})',
             xy=(CHOSEN_K, chosen_sil),
             xytext=(K_LIST[0] - 0.2, best_sil * 1.20),
             ha='left', va='center',
             color=pick_color, fontsize=9.5, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=pick_color, lw=1.2,
                             connectionstyle='arc3,rad=0.18'))

ax2.set_xlabel('Number of clusters (k)', fontsize=11)
ax2.set_ylabel('Silhouette score', fontsize=11)
ax2.set_title('Silhouette score', fontsize=13, fontweight='500', pad=10)
ax2.tick_params(labelsize=10)
ax2.grid(True, color=grid_color, lw=0.8)
ax2.set_axisbelow(True)

fig.suptitle('K-Means cluster selection — MURHCAD honeypot dataset\n'
             f'(n={len(df_clust)} IPs | total_events > 1)',
             fontsize=13.5, fontweight='500', y=1.03)

plt.savefig('elbow_silhouette.png', dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("\nFigure saved: elbow_silhouette.png")

# -- 6. One-shot attacker summary ------------------------------------------
print(f"\n-- One-shot attacker summary ({len(df_one)} IPs) --")
print(df_one['srcCountryName'].value_counts().head(10).to_string())
print("\nMean protocol share among one-shot IPs:")
print(df_one[['pct_sip', 'pct_smbd', 'pct_ssh',
              'pct_telnet', 'pct_others']].mean().round(3).to_string())

# Persist preprocessing objects for the clustering stage
import pickle
with open('prep_objects.pkl', 'wb') as f:
    pickle.dump({
        'df_clust': df_clust,
        'df_one': df_one,
        'X_scaled': X_scaled,
        'FEAT_COLS': FEAT_COLS,
        'scaler': scaler,
        'le_org': le_org,
        'le_cty': le_cty,
        'K_LIST': K_LIST,
        'inertias': inertias,
        'sil_scores': sil_scores,
    }, f)
print("\nPreprocessing objects saved: prep_objects.pkl")
print(f"Highest silhouette at k={best_k_sil} (score={best_sil:.4f}); "
      f"k={CHOSEN_K} adopted (score={chosen_sil:.4f})")