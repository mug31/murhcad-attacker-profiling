"""
Stage 5 — per-IP feature engineering.

CHANGES: console output translated to English. Feature logic, column order,
and the TOP_N_ORG=20 threshold are unchanged.
"""
import pandas as pd
import numpy as np

# --- Load & prep, per handover doc ---
df = pd.read_csv("HoneyNetEvents_Clean.csv")
MAJOR_PROTOCOLS = ["sip", "smbd", "ssh", "telnet"]
df["protocol_group"] = df["protocol"].apply(
    lambda x: x if x in MAJOR_PROTOCOLS else "others"
)

COLS_PAKAI = [
    "timestamp", "hour", "srcIp", "srcOrg", "srcCountryName",
    "srcLat", "srcLon", "dstPort", "dstHostname",
    "protocol_group", "attackType"
]
df = df[COLS_PAKAI]
print("Shape after column filter:", df.shape)
print("Unique IPs:", df["srcIp"].nunique())

# =====================================================
# FEATURE ENGINEERING PER-IP
# =====================================================

grouped = df.groupby("srcIp")

# --- 1. Volume features ---
vol = grouped.agg(
    total_events=("srcIp", "size"),
    unique_dst_ports=("dstPort", "nunique"),
    unique_dst_hosts=("dstHostname", "nunique"),
).reset_index()

# --- 2. Protocol features (share of events) ---
proto_counts = (
    df.groupby(["srcIp", "protocol_group"]).size().unstack(fill_value=0)
)
proto_pct = proto_counts.div(proto_counts.sum(axis=1), axis=0)
proto_pct = proto_pct.rename(columns=lambda c: f"pct_{c}")
# ensure every major-protocol column exists even if zero everywhere
for p in MAJOR_PROTOCOLS + ["others"]:
    col = f"pct_{p}"
    if col not in proto_pct.columns:
        proto_pct[col] = 0.0
proto_pct = proto_pct[[f"pct_{p}" for p in MAJOR_PROTOCOLS + ["others"]]].reset_index()

# --- 3. Temporal features ---
def hour_entropy(hours_series):
    counts = hours_series.value_counts()
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

temporal = grouped.agg(
    dominant_hour=("hour", lambda x: x.value_counts().idxmax()),
    active_hours=("hour", "nunique"),
).reset_index()

entropy_df = grouped["hour"].apply(hour_entropy).reset_index(name="hour_entropy")
temporal = temporal.merge(entropy_df, on="srcIp")

# --- 4. Geographic / organisation features ---
geo = grouped.agg(
    srcCountryName=("srcCountryName", lambda x: x.mode().iloc[0]),
    srcOrg=("srcOrg", lambda x: x.mode().iloc[0]),
    srcLat=("srcLat", "first"),
    srcLon=("srcLon", "first"),
).reset_index()

# Top-N srcOrg + "others" (keeps categorical encoding compact)
TOP_N_ORG = 20
top_orgs = df["srcOrg"].value_counts().nlargest(TOP_N_ORG).index
geo["srcOrg_grouped"] = geo["srcOrg"].apply(lambda x: x if x in top_orgs else "others")

# --- Merge all feature blocks ---
features = vol.merge(proto_pct, on="srcIp") \
              .merge(temporal, on="srcIp") \
              .merge(geo, on="srcIp")

# Fix final column order
final_cols = [
    "srcIp",
    "total_events", "unique_dst_ports", "unique_dst_hosts",
    "pct_sip", "pct_smbd", "pct_ssh", "pct_telnet", "pct_others",
    "dominant_hour", "hour_entropy", "active_hours",
    "srcCountryName", "srcOrg", "srcOrg_grouped", "srcLat", "srcLon",
]
features = features[final_cols]

print("\nPer-IP feature shape:", features.shape)
print("\nPreview:")
print(features.head(10).to_string())

print("\nSummary statistics, numeric features:")
print(features.describe().to_string())

# Save
features.to_csv("features_per_ip.csv", index=False)
print("\nSaved to features_per_ip.csv")