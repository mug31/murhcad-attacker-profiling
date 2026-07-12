import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Load prep objects ─────────────────────────────────────────────────────
with open('prep_objects.pkl', 'rb') as f:
    prep = pickle.load(f)

df_clust  = prep['df_clust']
X_scaled  = prep['X_scaled']
FEAT_COLS = prep['FEAT_COLS']
le_org    = prep['le_org']
le_cty    = prep['le_cty']

# ── K-Means k=6 ───────────────────────────────────────────────────────────
K = 6
km = KMeans(n_clusters=K, random_state=42, n_init=20, max_iter=500)
df_clust['cluster'] = km.fit_predict(X_scaled)

print("=" * 60)
print(f"K-Means  k={K}  |  n={len(df_clust)} IP  |  inertia={km.inertia_:,.0f}")
print("=" * 60)

# ── 1. Distribusi IP per cluster ──────────────────────────────────────────
print("\n── 1. Distribusi IP per Cluster ──")
dist = df_clust['cluster'].value_counts().sort_index()
for c, n in dist.items():
    print(f"  Cluster {c}: {n:4d} IP  ({n/len(df_clust)*100:.1f}%)")

# ── 2. Profil rata-rata per cluster ───────────────────────────────────────
print("\n── 2. Profil Rata-rata per Cluster ──")
PROFILE_COLS = [
    'total_events','pct_sip','pct_smbd','pct_ssh',
    'pct_telnet','dominant_hour','hour_entropy','active_hours'
]
profile = df_clust.groupby('cluster')[PROFILE_COLS].mean().round(3)
print(profile.to_string())

# ── 3. Top 3 negara & srcOrg per cluster ─────────────────────────────────
print("\n── 3. Top 3 Negara & Org per Cluster ──")
for c in range(K):
    sub = df_clust[df_clust['cluster'] == c]
    top_cty = sub['srcCountryName'].value_counts().head(3)
    top_org = sub['srcOrg_grouped'].value_counts().head(3)
    print(f"\n  Cluster {c} (n={len(sub)}):")
    print(f"    Negara : {', '.join([f'{n}({v})' for n,v in top_cty.items()])}")
    print(f"    Org    : {', '.join([f'{n}({v})' for n,v in top_org.items()])}")

# ── 4. Tipologi label ─────────────────────────────────────────────────────
# Berdasarkan profil dominan — ditetapkan setelah melihat output profil
def assign_label(row):
    c = int(row['cluster'])
    p = profile.loc[c]
    if p['pct_sip'] > 0.5:
        return 'SIP Scanner'
    elif p['pct_smbd'] > 0.4:
        return 'SMB Brute-forcer'
    elif p['pct_ssh'] > 0.5:
        return 'SSH Brute-forcer'
    elif p['pct_telnet'] > 0.5:
        return 'Telnet Botnet'
    elif p['active_hours'] > 10:
        return 'Persistent Multi-Protocol'
    else:
        return 'Multi-Protocol Scanner'

# Label manual berdasarkan inspeksi profil
cluster_labels = {}
proto_profile = df_clust.groupby('cluster')[
    ['pct_sip','pct_smbd','pct_ssh','pct_telnet','pct_others']
].mean()

for c in range(K):
    p       = profile.loc[c]
    pp      = proto_profile.loc[c]
    dominant_proto = pp.idxmax()
    proto_val      = pp.max()
    acts           = p['active_hours']

    if proto_val < 0.35:
        label = 'Multi-Protocol Scanner'
    elif dominant_proto == 'pct_sip':
        label = 'SIP Scanner'
    elif dominant_proto == 'pct_smbd':
        label = 'SMB Brute-forcer'
    elif dominant_proto == 'pct_ssh':
        label = 'SSH Persistent Crawler' if acts > 8 else 'SSH Brute-forcer'
    elif dominant_proto == 'pct_telnet':
        label = 'Telnet Botnet'
    else:
        label = 'Mixed/Others Scanner'

    cluster_labels[c] = label

print("\n── 4. Usulan Tipologi per Cluster ──")
for c, lbl in cluster_labels.items():
    p  = profile.loc[c]
    pp = proto_profile.loc[c]
    dom = pp.idxmax()
    print(f"  Cluster {c} → [{lbl}]  (dominant: {dom}={pp[dom]:.2f}, "
          f"events_avg={p['total_events']:.0f}, active_hrs={p['active_hours']:.1f})")

df_clust['tipologi'] = df_clust['cluster'].map(cluster_labels)

# ── 5. PCA 2D Scatter ─────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

PALETTE = ['#4fc3f7','#81c784','#ff8a65','#ffd54f','#ce93d8','#80cbc4']
CLUSTER_COLORS = {c: PALETTE[c] for c in range(K)}

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#1a1d2e')

for c in range(K):
    mask = df_clust['cluster'] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=CLUSTER_COLORS[c], label=f"C{c}: {cluster_labels[c]}",
               alpha=0.65, s=30, edgecolors='none', zorder=3)

# centroid di PCA space
centroids_pca = pca.transform(km.cluster_centers_)
for c in range(K):
    ax.scatter(centroids_pca[c, 0], centroids_pca[c, 1],
               c=CLUSTER_COLORS[c], s=220, marker='*',
               edgecolors='white', linewidths=0.8, zorder=5)
    ax.annotate(f'C{c}', (centroids_pca[c, 0], centroids_pca[c, 1]),
                fontsize=9, fontweight='bold', color='white',
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}% variance)', color='#c9cdd4', fontsize=11)
ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}% variance)', color='#c9cdd4', fontsize=11)
ax.set_title(
    f'K-Means Cluster (k=6) — PCA 2D Projection\n'
    f'MURHCAD Honeypot  |  n={len(df_clust)} IP  (total_events > 1)',
    color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='#c9cdd4')
for spine in ax.spines.values():
    spine.set_color('#2a2d3a')
ax.grid(True, color='#2a2d3a', ls='--', alpha=0.6)

legend = ax.legend(loc='upper right', framealpha=0.25, facecolor='#1a1d2e',
                   edgecolor='#2a2d3a', labelcolor='white', fontsize=9,
                   title='Cluster', title_fontsize=10)
legend.get_title().set_color('white')

fig.tight_layout()
plt.savefig('pca_clusters.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("\nChart disimpan: pca_clusters.png")

# ── Simpan hasil ──────────────────────────────────────────────────────────
out_cols = [
    'srcIp','total_events','unique_dst_ports','unique_dst_hosts',
    'pct_sip','pct_smbd','pct_ssh','pct_telnet','pct_others',
    'dominant_hour','hour_entropy','active_hours',
    'srcCountryName','srcOrg','srcOrg_grouped',
    'cluster','tipologi'
]
df_clust[out_cols].to_csv('clustered_ips.csv', index=False)
print("Hasil disimpan : clustered_ips.csv")
print(f"\nPCA variance explained: PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%  total={sum(var_exp)*100:.1f}%")