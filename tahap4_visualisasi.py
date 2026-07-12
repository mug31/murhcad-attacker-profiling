"""
Tahap 4 — Analisis Pola Temporal & Protokol
Dataset: MURHCAD (HoneyNetEvents_Clean.csv)
Output: 6 file PNG di folder ./output_charts/

Cara pakai:
    python tahap4_visualisasi.py

Pastikan HoneyNetEvents_Clean.csv ada di direktori yang sama,
atau ubah FILE_PATH di bawah.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
FILE_PATH   = "HoneyNetEvents_Clean.csv"
OUTPUT_DIR  = "output_charts"
DPI         = 150

# Palet warna konsisten (Tidepool)
COLORS = {
    "sip"    : "#2a78d6",
    "smbd"   : "#1baf7a",
    "telnet" : "#eda100",
    "ssh"    : "#4a3aa7",
    "others" : "#888780",
    "vm-centralindia"     : "#2a78d6",
    "vm-centralus"        : "#1baf7a",
    "vm-southafricanorth" : "#eda100",
    "vm-spaincentral"     : "#4a3aa7",
}

MAJOR_PROTOCOLS = ["sip", "smbd", "telnet", "ssh"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(FILE_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
df["protocol_group"] = df["protocol"].apply(
    lambda x: x if x in MAJOR_PROTOCOLS else "others"
)
print(f"  {len(df):,} event dimuat.\n")

# Gaya global
plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.color"      : "#e1e0d9",
    "grid.linewidth"  : 0.6,
    "figure.dpi"      : DPI,
})

# ── CHART 1 — Distribusi Protokol (horizontal bar) ───────────────────────────
print("Chart 1: Distribusi protokol...")

proto_counts = df["protocol_group"].value_counts().reindex(
    ["sip", "smbd", "telnet", "ssh", "others"]
)
proto_pct = (proto_counts / len(df) * 100).round(1)
colors_bar = [COLORS[p] for p in proto_counts.index]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(proto_counts.index, proto_counts.values,
               color=colors_bar, edgecolor="none", height=0.6)

for bar, pct, count in zip(bars, proto_pct.values, proto_counts.values):
    ax.text(bar.get_width() + 400, bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({pct}%)",
            va="center", ha="left", fontsize=10, color="#52514e")

ax.set_xlabel("Jumlah event", fontsize=11)
ax.set_title("Distribusi protokol serangan", fontsize=13, fontweight="500", pad=12)
ax.set_xlim(0, proto_counts.max() * 1.22)
ax.grid(axis="y", visible=False)
ax.invert_yaxis()
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "01_distribusi_protokol.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Tersimpan: {out}")

# ── CHART 2 — Volume Serangan per Jam (line chart) ───────────────────────────
print("Chart 2: Volume per jam (total)...")

hourly_total = df.groupby("hour").size().reset_index(name="count")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly_total["hour"], hourly_total["count"],
        color="#2a78d6", linewidth=2.2, marker="o", markersize=4,
        markerfacecolor="#2a78d6", zorder=3)
ax.fill_between(hourly_total["hour"], hourly_total["count"],
                alpha=0.08, color="#2a78d6")

peak_hour = hourly_total.loc[hourly_total["count"].idxmax(), "hour"]
peak_val  = hourly_total["count"].max()
ax.annotate(f"Puncak: {peak_val:,}\n(jam {peak_hour}:00 UTC)",
            xy=(peak_hour, peak_val),
            xytext=(peak_hour + 1.5, peak_val - 1500),
            fontsize=9, color="#185FA5",
            arrowprops=dict(arrowstyle="->", color="#185FA5", lw=1))

ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=9)
ax.set_xlabel("Jam (UTC)", fontsize=11)
ax.set_ylabel("Jumlah event", fontsize=11)
ax.set_title("Volume serangan per jam (semua protokol, UTC)", fontsize=13, fontweight="500", pad=12)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "02_volume_per_jam_total.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Tersimpan: {out}")

# ── CHART 3 — Volume per Jam per Protokol (multi-line) ───────────────────────
print("Chart 3: Volume per jam per protokol...")

hourly_proto = df.groupby(["hour", "protocol_group"]).size().reset_index(name="count")

fig, ax = plt.subplots(figsize=(10, 5))
for proto in MAJOR_PROTOCOLS:
    d = hourly_proto[hourly_proto["protocol_group"] == proto]
    ax.plot(d["hour"], d["count"],
            label=proto.upper(), color=COLORS[proto],
            linewidth=2, marker="o", markersize=3)

ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=9)
ax.set_xlabel("Jam (UTC)", fontsize=11)
ax.set_ylabel("Jumlah event", fontsize=11)
ax.set_title("Volume serangan per jam — dipisah per protokol (UTC)", fontsize=13, fontweight="500", pad=12)
ax.legend(frameon=False, fontsize=10, loc="upper left")
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "03_volume_per_jam_per_protokol.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Tersimpan: {out}")

# ── CHART 4 — Heatmap Protokol × Jam ─────────────────────────────────────────
print("Chart 4: Heatmap protokol × jam...")

pivot_proto = df.groupby(["protocol_group", "hour"]).size().unstack(fill_value=0)
pivot_proto = pivot_proto.reindex(["sip", "smbd", "telnet", "ssh", "others"])

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(
    pivot_proto, ax=ax,
    cmap="Blues", linewidths=0.3, linecolor="#f0efec",
    cbar_kws={"label": "Jumlah event", "shrink": 0.8},
    fmt="d", annot=True, annot_kws={"size": 7.5}
)
ax.set_xlabel("Jam (UTC)", fontsize=11)
ax.set_ylabel("Protokol", fontsize=11)
ax.set_title("Heatmap intensitas serangan — protokol × jam (UTC)", fontsize=13, fontweight="500", pad=12)
ax.set_xticklabels([f"{int(t.get_text()):02d}h" for t in ax.get_xticklabels()], fontsize=8.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "04_heatmap_protokol_jam.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Tersimpan: {out}")

# ── CHART 5 — Heatmap VM Region × Jam ────────────────────────────────────────
print("Chart 5: Heatmap region × jam...")

pivot_region = df.groupby(["dstHostname", "hour"]).size().unstack(fill_value=0)
region_order = ["vm-centralindia", "vm-centralus", "vm-southafricanorth", "vm-spaincentral"]
pivot_region = pivot_region.reindex(region_order)

fig, ax = plt.subplots(figsize=(14, 3.5))
sns.heatmap(
    pivot_region, ax=ax,
    cmap="Blues", linewidths=0.3, linecolor="#f0efec",
    cbar_kws={"label": "Jumlah event", "shrink": 0.8},
    fmt="d", annot=True, annot_kws={"size": 7.5}
)
ax.set_xlabel("Jam (UTC)", fontsize=11)
ax.set_ylabel("VM Region", fontsize=11)
ax.set_title("Heatmap intensitas serangan — VM region × jam (UTC)", fontsize=13, fontweight="500", pad=12)
ax.set_xticklabels([f"{int(t.get_text()):02d}h" for t in ax.get_xticklabels()], fontsize=8.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "05_heatmap_region_jam.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Tersimpan: {out}")

# ── CHART 6 — Distribusi Serangan per VM Region (bar) ────────────────────────
print("Chart 6: Distribusi per VM region...")

region_counts = df["dstHostname"].value_counts().reindex(region_order)
region_pct    = (region_counts / len(df) * 100).round(1)
colors_region = [COLORS[r] for r in region_order]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(region_counts.index, region_counts.values,
               color=colors_region, edgecolor="none", height=0.6)

for bar, pct, count in zip(bars, region_pct.values, region_counts.values):
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({pct}%)",
            va="center", ha="left", fontsize=10, color="#52514e")

ax.set_xlabel("Jumlah event", fontsize=11)
ax.set_title("Distribusi serangan per VM region", fontsize=13, fontweight="500", pad=12)
ax.set_xlim(0, region_counts.max() * 1.28)
ax.grid(axis="y", visible=False)
ax.invert_yaxis()
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "06_distribusi_region.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Tersimpan: {out}")

# ── SELESAI ───────────────────────────────────────────────────────────────────
print(f"\n✅ Semua chart tersimpan di folder: ./{OUTPUT_DIR}/")
print("   01_distribusi_protokol.png")
print("   02_volume_per_jam_total.png")
print("   03_volume_per_jam_per_protokol.png")
print("   04_heatmap_protokol_jam.png")
print("   05_heatmap_region_jam.png")
print("   06_distribusi_region.png")
