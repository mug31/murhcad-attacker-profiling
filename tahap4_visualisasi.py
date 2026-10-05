"""
Stage 4 — Temporal and protocol pattern analysis
Dataset: MURHCAD (HoneyNetEvents_Clean.csv)
Output: 6 PNG files in ./output_charts/

Usage:
    python tahap4_visualisasi.py

CHANGES FROM PREVIOUS VERSION
  1. All figure text (titles, axis labels, annotations) is in English. Review
     comment "Saran Umum II.4" flagged the previous mix of Indonesian axis
     labels with English legend entries.
  2. DPI raised from 150 to 300.
Chart logic, palette, and data handling are otherwise unchanged.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# -- CONFIG -----------------------------------------------------------------
FILE_PATH  = "HoneyNetEvents_Clean.csv"
OUTPUT_DIR = "output_charts"
DPI        = 300          # was 150

# Consistent colour palette (Tidepool)
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

# -- LOAD DATA --------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(FILE_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
df["protocol_group"] = df["protocol"].apply(
    lambda x: x if x in MAJOR_PROTOCOLS else "others"
)
print(f"  {len(df):,} events loaded.\n")

# Global style
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.color"       : "#e1e0d9",
    "grid.linewidth"   : 0.6,
    "figure.dpi"       : DPI,
})

# -- CHART 1 — Protocol distribution (horizontal bar) -----------------------
print("Chart 1: Protocol distribution...")

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

ax.set_xlabel("Number of events", fontsize=11)
ax.set_title("Attack distribution by protocol", fontsize=13, fontweight="medium", pad=12)
ax.set_xlim(0, proto_counts.max() * 1.22)
ax.grid(axis="y", visible=False)
ax.invert_yaxis()
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "01_distribusi_protokol.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# -- CHART 2 — Attack volume per hour (line chart) --------------------------
print("Chart 2: Hourly volume (all protocols)...")

hourly_total = df.groupby("hour").size().reset_index(name="count")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly_total["hour"], hourly_total["count"],
        color="#2a78d6", linewidth=2.2, marker="o", markersize=4,
        markerfacecolor="#2a78d6", zorder=3)
ax.fill_between(hourly_total["hour"], hourly_total["count"],
                alpha=0.08, color="#2a78d6")

peak_hour = hourly_total.loc[hourly_total["count"].idxmax(), "hour"]
peak_val  = hourly_total["count"].max()
ax.annotate(f"Peak: {peak_val:,}\n({peak_hour}:00 UTC)",
            xy=(peak_hour, peak_val),
            xytext=(peak_hour + 1.5, peak_val - 1500),
            fontsize=9, color="#185FA5",
            arrowprops=dict(arrowstyle="->", color="#185FA5", lw=1))

ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=9)
ax.set_xlabel("Hour (UTC)", fontsize=11)
ax.set_ylabel("Number of events", fontsize=11)
ax.set_title("Hourly attack volume (all protocols, UTC)",
             fontsize=13, fontweight="medium", pad=12)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "02_volume_per_jam_total.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# -- CHART 3 — Hourly volume by protocol (multi-line) -----------------------
print("Chart 3: Hourly volume by protocol...")

hourly_proto = df.groupby(["hour", "protocol_group"]).size().reset_index(name="count")

fig, ax = plt.subplots(figsize=(10, 5))
for proto in MAJOR_PROTOCOLS:
    d = hourly_proto[hourly_proto["protocol_group"] == proto]
    ax.plot(d["hour"], d["count"],
            label=proto.upper(), color=COLORS[proto],
            linewidth=2, marker="o", markersize=3)

ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=9)
ax.set_xlabel("Hour (UTC)", fontsize=11)
ax.set_ylabel("Number of events", fontsize=11)
ax.set_title("Hourly attack volume by protocol (UTC)",
             fontsize=13, fontweight="medium", pad=12)
ax.legend(frameon=False, fontsize=10, loc="upper left")
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "03_volume_per_jam_per_protokol.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# -- CHART 4 — Heatmap protocol x hour --------------------------------------
print("Chart 4: Heatmap protocol x hour...")

pivot_proto = df.groupby(["protocol_group", "hour"]).size().unstack(fill_value=0)
pivot_proto = pivot_proto.reindex(["sip", "smbd", "telnet", "ssh", "others"])

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(
    pivot_proto, ax=ax,
    cmap="Blues", linewidths=0.3, linecolor="#f0efec",
    cbar_kws={"label": "Number of events", "shrink": 0.8},
    fmt="d", annot=True, annot_kws={"size": 7.5}
)
ax.set_xlabel("Hour (UTC)", fontsize=11)
ax.set_ylabel("Protocol", fontsize=11)
ax.set_title("Attack intensity heatmap — protocol x hour (UTC)",
             fontsize=13, fontweight="medium", pad=12)
ax.set_xticklabels([f"{int(t.get_text()):02d}h" for t in ax.get_xticklabels()], fontsize=8.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "04_heatmap_protokol_jam.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# -- CHART 5 — Heatmap VM region x hour -------------------------------------
print("Chart 5: Heatmap region x hour...")

pivot_region = df.groupby(["dstHostname", "hour"]).size().unstack(fill_value=0)
region_order = ["vm-centralindia", "vm-centralus", "vm-southafricanorth", "vm-spaincentral"]
pivot_region = pivot_region.reindex(region_order)

fig, ax = plt.subplots(figsize=(14, 3.5))
sns.heatmap(
    pivot_region, ax=ax,
    cmap="Blues", linewidths=0.3, linecolor="#f0efec",
    cbar_kws={"label": "Number of events", "shrink": 0.8},
    fmt="d", annot=True, annot_kws={"size": 7.5}
)
ax.set_xlabel("Hour (UTC)", fontsize=11)
ax.set_ylabel("VM region", fontsize=11)
ax.set_title("Attack intensity heatmap — VM region x hour (UTC)",
             fontsize=13, fontweight="medium", pad=12)
ax.set_xticklabels([f"{int(t.get_text()):02d}h" for t in ax.get_xticklabels()], fontsize=8.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "05_heatmap_region_jam.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# -- CHART 6 — Attack distribution per VM region (bar) ----------------------
print("Chart 6: Distribution per VM region...")

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

ax.set_xlabel("Number of events", fontsize=11)
ax.set_title("Attack distribution by VM region", fontsize=13, fontweight="medium", pad=12)
ax.set_xlim(0, region_counts.max() * 1.28)
ax.grid(axis="y", visible=False)
ax.invert_yaxis()
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "06_distribusi_region.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# -- DONE -------------------------------------------------------------------
print(f"\nAll charts saved to ./{OUTPUT_DIR}/")
for name in ["01_distribusi_protokol.png",
             "02_volume_per_jam_total.png",
             "03_volume_per_jam_per_protokol.png",
             "04_heatmap_protokol_jam.png",
             "05_heatmap_region_jam.png",
             "06_distribusi_region.png"]:
    print(f"   {name}")