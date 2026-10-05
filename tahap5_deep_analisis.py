"""
Stage 5 — deep-dive analysis and per-cluster temporal figures.

CHANGES FROM PREVIOUS VERSION
  1. LABELS_DRAFT removed. Labels now come from cluster_labels.py (Table 4.1).
     The old draft dict is what put 'Mixed/Others Scanner' on Figure 4.5 and
     'SIP Scanner' on the C5 panel.
  2. A cluster identity check runs before labels are applied.
  3. All figure text and console output is in English.
  4. Figures saved at 300 dpi.
  5. NEW: generates 07_jam_aktif_per_tipologi.png (mean active hours per
     typology). That figure is embedded in the manuscript but had no script in
     the pipeline, which is why its labels drifted out of sync with Table 4.1.
  6. The raw-event section now degrades gracefully if HoneyNetEvents_Clean.csv
     is absent, so the figures can be regenerated on their own.

Clustering parameters are UNCHANGED.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import warnings
warnings.filterwarnings('ignore')

from cluster_labels import (CLUSTER_LABELS, CLUSTER_LABELS_SHORT,
                            CLUSTER_LABELS_WRAPPED, verify_cluster_identity,
                            canonicalise_cluster_ids)

DPI = 300
RAW_EVENTS_PATH = 'HoneyNetEvents_Clean.csv'

# -- Load data --------------------------------------------------------------
with open('prep_objects.pkl', 'rb') as f:
    prep = pickle.load(f)

df_clust = prep['df_clust']
X_scaled = prep['X_scaled']

# Re-attach cluster labels (same seed -> same result)
km = KMeans(n_clusters=6, random_state=42, n_init=20, max_iter=500)
raw_ids = km.fit_predict(X_scaled)
df_clust['cluster'], _map = canonicalise_cluster_ids(
    df_clust.assign(cluster=raw_ids), X_scaled, km, cluster_col='cluster')

print("=" * 65)
print("IN-DEPTH ANALYSIS — K-MEANS k=6, MURHCAD HONEYPOT")
print("=" * 65)

verify_cluster_identity(df_clust, cluster_col='cluster', strict=False)

# ===================================================================
# PART 1: Detailed protocol breakdown, C0 vs C3   (needs raw events)
# ===================================================================
if os.path.exists(RAW_EVENTS_PATH):
    df_raw = pd.read_csv(RAW_EVENTS_PATH)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'],
                                         format='mixed', utc=True)
    df_raw_c = df_raw.merge(df_clust[['srcIp', 'cluster']], on='srcIp', how='inner')

    print("\n" + "-" * 65)
    print("PART 1: Detailed protocol breakdown — C0 vs C3")
    print("-" * 65)

    for c in [0, 3]:
        sub = df_raw_c[df_raw_c['cluster'] == c]
        total = len(sub)
        proto_dist = sub['protocol'].value_counts().reset_index()
        proto_dist.columns = ['protocol', 'count']
        proto_dist['pct'] = (proto_dist['count'] / total * 100).round(2)
        print(f"\nCluster {c} — {CLUSTER_LABELS[c]}")
        print(f"  {df_clust[df_clust['cluster']==c].shape[0]} IPs, {total} events")
        print(proto_dist.head(12).to_string(index=False))
else:
    print(f"\n[skip] PART 1 requires {RAW_EVENTS_PATH}; file not found.")

# ===================================================================
# PART 2: Top 5 srcOrg per cluster
# ===================================================================
print("\n" + "-" * 65)
print("PART 2: Top 5 srcOrg_grouped per cluster")
print("-" * 65)

for c in range(6):
    sub = df_clust[df_clust['cluster'] == c]
    n_ip = len(sub)
    print(f"\nCluster {c} — {CLUSTER_LABELS[c]} (n={n_ip}):")
    for org, cnt in sub['srcOrg_grouped'].value_counts().head(5).items():
        print(f"  {cnt:4d} IPs  ({cnt/n_ip*100:5.1f}%)  {org}")

# ===================================================================
# PART 3: Representative IPs per cluster
# ===================================================================
print("\n" + "-" * 65)
print("PART 3: Representative IPs per cluster (closest to centroid)")
print("-" * 65)

FEAT_COLS = prep['FEAT_COLS']
SHOW_COLS = ['srcIp', 'total_events', 'pct_sip', 'pct_smbd', 'pct_ssh',
             'pct_telnet', 'pct_others', 'dominant_hour', 'hour_entropy',
             'active_hours', 'srcCountryName', 'srcOrg_grouped']

for c in range(6):
    idx = np.where(df_clust['cluster'] == c)[0]
    dists = np.linalg.norm(X_scaled[idx] - km.cluster_centers_[c], axis=1)
    closest_idx = idx[np.argsort(dists)[:5]]

    print(f"\nCluster {c} — {CLUSTER_LABELS[c]} — 5 IPs closest to centroid:")
    rep = df_clust.iloc[closest_idx][SHOW_COLS].copy()
    for col in ['pct_sip', 'pct_smbd', 'pct_ssh', 'pct_telnet',
                'pct_others', 'hour_entropy']:
        rep[col] = rep[col].round(3)
    print(rep.to_string(index=False))

# ===================================================================
# PART 4: Dominant-hour histogram per cluster (6 panels)
#         -> dominant_hour_clusters.png   (Figure 4.5)
# ===================================================================
# Light-mode palette matched to tahap5_kmeans.py and the Stage-4 charts.
PALETTE = ['#2a78d6', '#1baf7a', '#d1495b', '#eda100', '#8a5fbf', '#0f8b8d']

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor('white')
axes = axes.flatten()

for c in range(6):
    ax  = axes[c]
    sub = df_clust[df_clust['cluster'] == c]
    ax.set_facecolor('white')

    counts, _ = np.histogram(sub['dominant_hour'], bins=range(25))
    ax.bar(range(24), counts, color=PALETTE[c], alpha=0.9, width=0.82)

    ax.set_title(f"C{c}: {CLUSTER_LABELS_SHORT[c]}\nn={len(sub)} IPs",
                 fontsize=10.5, fontweight='500', pad=8)
    ax.set_xlabel('Hour UTC (dominant_hour)', fontsize=9.5)
    ax.set_ylabel('Number of IPs', fontsize=9.5)
    ax.tick_params(labelsize=9)
    ax.set_xticks(range(0, 24, 3))
    ax.grid(True, axis='y', color='#e1e0d9', lw=0.8)
    ax.set_axisbelow(True)

    # Headroom so the callout never sits on top of the tallest bar.
    ax.set_ylim(0, counts.max() * 1.28)

    peak_hr  = int(np.argmax(counts))
    peak_val = counts[peak_hr]
    ax.annotate(f'mode {peak_hr}:00',
                xy=(peak_hr, peak_val),
                xytext=(peak_hr + (2.5 if peak_hr < 16 else -2.5),
                        counts.max() * 1.16),
                ha='left' if peak_hr < 16 else 'right', va='center',
                color='#52514e', fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#52514e', lw=1))

fig.suptitle('Distribution of dominant hour per cluster — MURHCAD honeypot',
             fontsize=13.5, fontweight='500', y=1.0)
plt.tight_layout()
plt.savefig('dominant_hour_clusters.png', dpi=DPI,
            bbox_inches='tight', facecolor='white')
plt.close()
print("\n\nFigure saved: dominant_hour_clusters.png")

# ===================================================================
# PART 5: Mean active hours per typology
#         -> 07_jam_aktif_per_tipologi.png   (Figure 4.4)
# NEW in this version: this figure previously had no generating script.
# Filename kept as-is because the manuscript embeds it under that name.
# ===================================================================
BAR_COLORS = {0: '#f0a020', 1: '#2e9e35', 2: '#4b3f9e',
              3: '#26a69a', 4: '#808080', 5: '#1f77b4'}

active = df_clust.groupby('cluster')['active_hours'].mean().sort_values()

fig, ax = plt.subplots(figsize=(9.5, 5.4))
ypos = range(len(active))
ax.barh(list(ypos), active.values,
        color=[BAR_COLORS[c] for c in active.index], height=0.62)
ax.set_yticks(list(ypos))
ax.set_yticklabels([CLUSTER_LABELS_WRAPPED[c].replace('—', '-')
                    for c in active.index], fontsize=10.5)
ax.set_xlim(0, max(10, active.max() * 1.25))
ax.set_xlabel('Mean active hours per IP', fontsize=12)
ax.set_title('Mean active hours per typology (cluster)', fontsize=14, pad=14)

for i, v in enumerate(active.values):
    ax.text(v + 0.18, i, f"{v:.1f} h", va='center',
            fontsize=11.5, color='#333333')

ax.grid(axis='x', color='#dddddd', linewidth=0.8)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.spines['left'].set_color('#666666')
ax.spines['bottom'].set_color('#666666')
ax.tick_params(axis='x', labelsize=11)

plt.tight_layout()
plt.savefig('07_jam_aktif_per_tipologi.png', dpi=DPI,
            facecolor='white', bbox_inches='tight')
plt.close()
print("Figure saved: 07_jam_aktif_per_tipologi.png")