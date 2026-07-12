import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import pickle, warnings
warnings.filterwarnings('ignore')

# ── Load data ─────────────────────────────────────────────────────────────
with open('prep_objects.pkl', 'rb') as f:
    prep = pickle.load(f)

df_clust  = prep['df_clust']
X_scaled  = prep['X_scaled']

# Re-attach cluster labels (same seed → same result)
km = KMeans(n_clusters=6, random_state=42, n_init=20, max_iter=500)
df_clust['cluster'] = km.fit_predict(X_scaled)

# Load raw event data untuk breakdown protocol detail
df_raw = pd.read_csv('HoneyNetEvents_Clean.csv')
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], format='mixed', utc=True)

MAJOR = ['sip','smbd','ssh','telnet']
df_raw['protocol_group'] = df_raw['protocol'].apply(
    lambda x: x if x in MAJOR else x)   # semua protocol, tidak digrup

# Merge cluster info ke raw events
ip_cluster = df_clust[['srcIp','cluster']].copy()
df_raw_c = df_raw.merge(ip_cluster, on='srcIp', how='inner')

print("=" * 65)
print("ANALISIS MENDALAM K-MEANS k=6 — MURHCAD HONEYPOT")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════
# BAGIAN 1: Breakdown protocol detail C0 vs C3
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("BAGIAN 1: Breakdown Protocol Detail — C0 vs C3")
print("─"*65)

for c in [0, 3]:
    sub = df_raw_c[df_raw_c['cluster'] == c]
    total = len(sub)
    proto_dist = (sub['protocol'].value_counts()
                                 .reset_index()
                                 .rename(columns={'protocol':'protocol','count':'count'}))
    proto_dist['pct'] = (proto_dist['count'] / total * 100).round(2)
    print(f"\nCluster {c} — {df_clust[df_clust['cluster']==c].shape[0]} IP, "
          f"{total} total events")
    print(proto_dist.head(12).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# BAGIAN 2: Top 5 srcOrg per cluster
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("BAGIAN 2: Top 5 srcOrg_grouped per Cluster")
print("─"*65)

for c in range(6):
    sub = df_clust[df_clust['cluster'] == c]
    top = sub['srcOrg_grouped'].value_counts().head(5)
    n_ip = len(sub)
    print(f"\nCluster {c} (n={n_ip}):")
    for org, cnt in top.items():
        print(f"  {cnt:4d} IP  ({cnt/n_ip*100:5.1f}%)  {org}")

# ═══════════════════════════════════════════════════════════════════
# BAGIAN 3: IP representatif per cluster
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("BAGIAN 3: IP Representatif per Cluster (closest to centroid)")
print("─"*65)

FEAT_COLS = prep['FEAT_COLS']
X_df = pd.DataFrame(X_scaled, columns=FEAT_COLS)

SHOW_COLS = ['srcIp','total_events','pct_sip','pct_smbd','pct_ssh',
             'pct_telnet','pct_others','dominant_hour','hour_entropy',
             'active_hours','srcCountryName','srcOrg_grouped']

for c in range(6):
    mask = df_clust['cluster'] == c
    idx  = np.where(mask)[0]
    center = km.cluster_centers_[c]
    dists  = np.linalg.norm(X_scaled[idx] - center, axis=1)
    closest_idx = idx[np.argsort(dists)[:5]]

    print(f"\nCluster {c} — 5 IP terdekat ke centroid:")
    rep = df_clust.iloc[closest_idx][SHOW_COLS].copy()
    # Format float
    for col in ['pct_sip','pct_smbd','pct_ssh','pct_telnet','pct_others',
                'hour_entropy']:
        rep[col] = rep[col].round(3)
    print(rep.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# BAGIAN 4: Histogram dominant_hour per cluster (6-panel)
# ═══════════════════════════════════════════════════════════════════
PALETTE = ['#4fc3f7','#81c784','#ff8a65','#ffd54f','#ce93d8','#80cbc4']
LABELS_DRAFT = {
    0: 'Mixed/Others Scanner',
    1: 'SMB Brute-forcer',
    2: 'SSH Brute-forcer',
    3: 'Persistent Multi-Protocol',
    4: 'Telnet Botnet',
    5: 'SIP Scanner',
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor('#0f1117')
axes = axes.flatten()

for c in range(6):
    ax   = axes[c]
    sub  = df_clust[df_clust['cluster'] == c]
    ax.set_facecolor('#1a1d2e')

    counts, _ = np.histogram(sub['dominant_hour'], bins=range(25))
    ax.bar(range(24), counts, color=PALETTE[c], alpha=0.85, width=0.85)

    ax.set_title(f"C{c}: {LABELS_DRAFT[c]}\nn={len(sub)} IP",
                 color='white', fontsize=9.5, fontweight='bold')
    ax.set_xlabel('Jam UTC (dominant_hour)', color='#c9cdd4', fontsize=8)
    ax.set_ylabel('Jumlah IP', color='#c9cdd4', fontsize=8)
    ax.tick_params(colors='#c9cdd4', labelsize=7.5)
    ax.set_xticks(range(0, 24, 3))
    for spine in ax.spines.values():
        spine.set_color('#2a2d3a')
    ax.grid(True, axis='y', color='#2a2d3a', ls='--', alpha=0.6)

    # annotate peak hour
    peak_hr = int(np.argmax(counts))
    peak_val = counts[peak_hr]
    ax.annotate(f'peak\n{peak_hr}:00',
                xy=(peak_hr, peak_val),
                xytext=(peak_hr + (2 if peak_hr < 20 else -4), peak_val * 0.85),
                color='white', fontsize=7.5,
                arrowprops=dict(arrowstyle='->', color='white', lw=0.8))

fig.suptitle('Distribusi Dominant Hour per Cluster — MURHCAD Honeypot',
             color='white', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('dominant_hour_clusters.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("\n\nChart disimpan: dominant_hour_clusters.png")