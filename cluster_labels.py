"""
cluster_labels.py — single source of truth for cluster typology labels.

The labels below are copied verbatim from Table 4.1 of the manuscript, which is
the authoritative source. Previously each script derived its own labels (either
via a heuristic in tahap5_kmeans.py or a hard-coded draft dict in
tahap5_deep_analisis.py), which is why Figure 4.4 and Figure 4.5 disagreed with
Table 4.1. Every script now imports from here instead.

If you change a label, change it HERE ONLY — then re-run the plotting scripts
and update Table 4.1 to match.
"""

# Canonical labels — must match Table 4.1 exactly.
CLUSTER_LABELS = {
    0: "Hit-and-Run Database & RPC Scanner",
    1: "SMB Brute-forcer",
    2: "SSH Brute-forcer",
    3: "Persistent Telnet/SSH Brute-forcer",
    4: "Low-volume Telnet IoT Botnet",
    5: "Aggressive SIP Botnet",
}

# Shorter variants for cramped subplot titles / legends.
CLUSTER_LABELS_SHORT = {
    0: "Hit-and-Run DB/RPC Scanner",
    1: "SMB Brute-forcer",
    2: "SSH Brute-forcer",
    3: "Persistent Telnet/SSH",
    4: "Low-volume Telnet IoT Botnet",
    5: "Aggressive SIP Botnet",
}

# Two-line variants for horizontal bar charts with long y-tick labels.
CLUSTER_LABELS_WRAPPED = {
    0: "C0 — Hit-and-Run Database\n& RPC Scanner",
    1: "C1 — SMB Brute-forcer",
    2: "C2 — SSH Brute-forcer",
    3: "C3 — Persistent Telnet/SSH\nBrute-forcer",
    4: "C4 — Low-volume Telnet\nIoT Botnet",
    5: "C5 — Aggressive SIP Botnet",
}

# Expected cluster sizes as published in Table 4.1.
#
# WHY THIS EXISTS: K-Means cluster IDs are not semantically meaningful. They are
# stable for a fixed random_state within one scikit-learn build, but a library
# upgrade or a change upstream in feature engineering can permute them. If that
# happens, label 0 would silently be attached to a different group and every
# figure would be wrong in a way that is hard to spot. Verified against
# scikit-learn 1.8.0: assignment reproduces exactly (ARI = 1.0).
EXPECTED_SIZES = {0: 245, 1: 175, 2: 206, 3: 68, 4: 180, 5: 27}

# Feature expected to dominate each cluster's MEAN PER-IP protocol profile,
# with a minimum share. Values verified against the current pipeline output.
#
# Note on C0 and C3: both are dominated by 'pct_others' on a per-IP basis
# (0.89 and 0.51 respectively), even though Table 4.1 describes C3 as Telnet-led
# (84%) on an event-weighted basis. That divergence is real and is already
# disclosed in the footnote to Table 4.1 — it is not an error here.
EXPECTED_DOMINANT = {
    0: ("pct_others", 0.50),
    1: ("pct_smbd",   0.50),
    2: ("pct_ssh",    0.50),
    3: ("pct_others", 0.40),
    4: ("pct_telnet", 0.50),
    5: ("pct_sip",    0.50),
}

PROTO_COLS = ["pct_sip", "pct_smbd", "pct_ssh", "pct_telnet", "pct_others"]


def verify_cluster_identity(df_clust, cluster_col="cluster", strict=False):
    """Check that cluster IDs still mean what CLUSTER_LABELS says they mean.

    Prints a report and returns True if everything matches. Set strict=True to
    raise instead of warn — recommended for any run whose output goes into the
    manuscript.
    """
    ok = True
    print("\n-- Cluster identity check (vs Table 4.1) --")

    sizes = df_clust[cluster_col].value_counts().sort_index()
    for c, expected in EXPECTED_SIZES.items():
        actual = int(sizes.get(c, 0))
        flag = "OK " if actual == expected else "MISMATCH"
        if actual != expected:
            ok = False
        print(f"  C{c}  n={actual:4d}  (expected {expected:4d})  {flag}")

    profile = df_clust.groupby(cluster_col)[PROTO_COLS].mean()
    for c, (feat, floor) in EXPECTED_DOMINANT.items():
        if c not in profile.index:
            ok = False
            continue
        row = profile.loc[c]
        top, val = row.idxmax(), row.max()
        good = (top == feat) and (val >= floor)
        if not good:
            ok = False
        print(f"  C{c}  {'OK ' if good else 'MISMATCH'}  "
              f"top={top} ({val:.2f}); expected {feat} >= {floor:.2f}")

    if ok:
        print("  --> All checks passed. Labels are safe to apply.\n")
    else:
        msg = ("Cluster IDs no longer match Table 4.1. Do NOT trust the labels "
               "on the generated figures. Re-inspect the cluster profiles and "
               "either remap CLUSTER_LABELS or update Table 4.1.")
        print(f"  --> WARNING: {msg}\n")
        if strict:
            raise RuntimeError(msg)
    return ok


def canonicalise_cluster_ids(df_clust, X_scaled, km, cluster_col="cluster"):
    """Relabel raw K-Means cluster IDs to the canonical numbering of Table 4.1.

    K-Means IDs are arbitrary: dropping srcOrg from the feature set left the
    PARTITION identical (adjusted Rand index = 1.0 vs the 13-feature run) but
    permuted the integer IDs. Table 4.1, and all of Bab 4, refer to fixed IDs
    C0..C5 defined by protocol/behaviour. This maps each raw ID to the canonical
    ID whose EXPECTED_DOMINANT profile it matches, so labels stay correct
    regardless of how K-Means happened to number the clusters.

    Returns a new integer Series aligned to df_clust, and remaps
    km.cluster_centers_ in place is NOT done (centroids keep raw order); callers
    that need centroids by canonical ID should index via the returned mapping.
    """
    import numpy as np
    profile = df_clust.groupby(cluster_col)[PROTO_COLS].mean()
    # Build canonical signature: (dominant feature, is_mixed) per canonical ID.
    raw_to_canon = {}
    used = set()
    # First pass: unambiguous single-protocol clusters (smbd/ssh/telnet/sip).
    for raw in profile.index:
        row = profile.loc[raw]
        top, val = row.idxmax(), row.max()
        for canon, (feat, floor) in EXPECTED_DOMINANT.items():
            if canon in used or feat == "pct_others":
                continue
            if top == feat and val >= floor:
                raw_to_canon[raw] = canon; used.add(canon); break
    # Second pass: the two 'mixed/others' clusters (C0 vs C3) by size/active_hours.
    remaining_raw = [r for r in profile.index if r not in raw_to_canon]
    remaining_canon = [c for c in EXPECTED_DOMINANT if c not in used]
    # C0 is the larger, low-persistence scanner; C3 is small, high active_hours.
    sizes = df_clust[cluster_col].value_counts()
    remaining_raw.sort(key=lambda r: sizes[r], reverse=True)   # big first -> C0
    remaining_canon.sort()                                     # [0, 3]
    for r, c in zip(remaining_raw, remaining_canon):
        raw_to_canon[r] = c
    return df_clust[cluster_col].map(raw_to_canon), raw_to_canon