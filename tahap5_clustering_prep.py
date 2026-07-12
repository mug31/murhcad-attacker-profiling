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

# ── 1. Load fitur per-IP ──────────────────────────────────────────────────
df = pd.read_csv("features_per_ip.csv")
print(f"Total IP: {len(df)}")
print(f"IP dengan total_events=1 : {(df['total_events']==1).sum()} ({(df['total_events']==1).mean()*100:.1f}%)")
print(f"IP dengan total_events>1 : {(df['total_events']>1).sum()}")

# ── 2. Split: one-shot vs recurring ──────────────────────────────────────
df_one   = df[df['total_events'] == 1].copy()   # analisis terpisah
df_clust = df[df['total_events'] >  1].copy().reset_index(drop=True)
print(f"\nDataset clustering: {len(df_clust)} baris")

# ── 3. Preprocessing ─────────────────────────────────────────────────────
# 3a. Log-transform fitur volume
df_clust['log_total_events']    = np.log1p(df_clust['total_events'])
df_clust['log_unique_dst_ports'] = np.log1p(df_clust['unique_dst_ports'])

# 3b. Encode srcOrg_grouped (top-20 + others → integer)
le_org = LabelEncoder()
df_clust['srcOrg_enc'] = le_org.fit_transform(df_clust['srcOrg_grouped'])

# 3c. Encode srcCountryName
le_cty = LabelEncoder()
df_clust['srcCountry_enc'] = le_cty.fit_transform(df_clust['srcCountryName'])

# 3d. Pilih kolom fitur untuk clustering
FEAT_COLS = [
    'log_total_events',
    'log_unique_dst_ports',
    'unique_dst_hosts',
    'pct_sip', 'pct_smbd', 'pct_ssh', 'pct_telnet', 'pct_others',
    'hour_entropy',
    'active_hours',
    'dominant_hour',
    'srcOrg_enc',
    'srcCountry_enc',
]

X = df_clust[FEAT_COLS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nFeature matrix shape: {X_scaled.shape}")
print(f"Fitur: {FEAT_COLS}")

# ── 4. Elbow + Silhouette ─────────────────────────────────────────────────
K_RANGE = range(2, 13)
inertias, sil_scores = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)), random_state=42))
    print(f"  k={k:2d} | inertia={km.inertia_:,.0f} | silhouette={sil_scores[-1]:.4f}")

# ── 5. Plot ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
fig.patch.set_facecolor('#0f1117')

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

K_LIST = list(K_RANGE)
elbow_color   = '#4fc3f7'
sil_color     = '#81c784'
anno_color    = '#ffd54f'
grid_color    = '#2a2d3a'

# — Elbow —
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#1a1d2e')
ax1.plot(K_LIST, inertias, 'o-', color=elbow_color, lw=2, ms=7, zorder=3)
ax1.fill_between(K_LIST, inertias, alpha=0.12, color=elbow_color)
ax1.set_xlabel('Jumlah Cluster (k)', color='#c9cdd4', fontsize=11)
ax1.set_ylabel('Inertia (WCSS)', color='#c9cdd4', fontsize=11)
ax1.set_title('Elbow Method', color='white', fontsize=13, fontweight='bold')
ax1.tick_params(colors='#c9cdd4')
ax1.spines[:].set_color(grid_color)
ax1.set_facecolor('#1a1d2e')
ax1.grid(True, color=grid_color, ls='--', alpha=0.7)

# annotate elbow candidate
best_k_elbow = 5
ax1.axvline(best_k_elbow, color=anno_color, ls='--', lw=1.5, alpha=0.85)
ax1.annotate(f'k={best_k_elbow}', xy=(best_k_elbow, inertias[best_k_elbow-2]),
             xytext=(best_k_elbow+0.5, inertias[best_k_elbow-2]*1.05),
             color=anno_color, fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=anno_color, lw=1.2))

# — Silhouette —
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#1a1d2e')
ax2.plot(K_LIST, sil_scores, 's-', color=sil_color, lw=2, ms=7, zorder=3)
ax2.fill_between(K_LIST, sil_scores, alpha=0.12, color=sil_color)

best_k_sil = K_LIST[np.argmax(sil_scores)]
best_sil   = max(sil_scores)
ax2.axvline(best_k_sil, color=anno_color, ls='--', lw=1.5, alpha=0.85)
ax2.annotate(f'k={best_k_sil}\n(sil={best_sil:.3f})',
             xy=(best_k_sil, best_sil),
             xytext=(best_k_sil+0.5, best_sil-0.005),
             color=anno_color, fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=anno_color, lw=1.2))

ax2.set_xlabel('Jumlah Cluster (k)', color='#c9cdd4', fontsize=11)
ax2.set_ylabel('Silhouette Score', color='#c9cdd4', fontsize=11)
ax2.set_title('Silhouette Score', color='white', fontsize=13, fontweight='bold')
ax2.tick_params(colors='#c9cdd4')
ax2.spines[:].set_color(grid_color)
ax2.grid(True, color=grid_color, ls='--', alpha=0.7)

fig.suptitle('K-Means Cluster Selection — MURHCAD Honeypot Dataset\n'
             f'(n={len(df_clust)} IP | total_events > 1)',
             color='white', fontsize=14, fontweight='bold', y=1.02)

plt.savefig('elbow_silhouette.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("\nChart disimpan: elbow_silhouette.png")

# ── 6. Ringkasan one-shot attacker ───────────────────────────────────────
print(f"\n── One-Shot Attacker Summary ({len(df_one)} IP) ──")
print(df_one['srcCountryName'].value_counts().head(10).to_string())
print(f"\nTop protocol one-shot:")
print(df_one[['pct_sip','pct_smbd','pct_ssh','pct_telnet','pct_others']].mean().round(3).to_string())

# Simpan objek preprocessing untuk dipakai di tahap clustering
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
print("\nPreprocessing objects disimpan: prep_objects.pkl")
print("\nBest k — Silhouette:", best_k_sil, f"(score={best_sil:.4f})")