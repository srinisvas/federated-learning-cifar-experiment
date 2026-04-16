"""
Deep diagnostic of per_update_features.csv — goes beyond the
pre-flight verifier to surface signal the simple population-mean
comparison misses.

Analyses:
  A. Per-round Z-scored separation (corrects for round-level drift)
  B. Full feature ranking by raw separation strength
  C. Temporal consistency: do malicious clients show persistent patterns
     across rounds that honest clients don't?
  D. Krum score analysis
  E. Attack effectiveness check (needs per_round_centralized.csv)
  F. NEW FEATURES HIGHLIGHT — reports on the 15 squeeze-targeting features
     (cos_to_mean_leave_one_out, head_sign_agreement_with_ref,
      backbone_kurtosis_max, head_backbone_conservation_product,
      and 5 stage_pair projected cosines)
  G. SQUEEZE PAIR CORRELATION — tests the inverse-correlation hypothesis
     directly: do attackers land in the "forbidden" regions of squeeze
     plots while honest clients cluster in the natural region?

Usage:
  python deep_diagnostic.py [path/to/per_update_features.csv]
"""

import sys
import math
import warnings
import os
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================================
# Squeeze definitions — the five theoretical squeezes as feature pairs.
# =========================================================================

SQUEEZE_PAIRS = [
    {
        "name": "Squeeze 1: norm <-> class entropy",
        "feat_x": "total_update_l2_norm",
        "feat_y": "per_class_delta_entropy",
        "hypothesis": "honest=moderate both; attacker forced to low-norm-low-entropy or high-norm-high-entropy",
    },
    {
        "name": "Squeeze 2: head signal <-> backbone kurtosis",
        "feat_x": "stage_head_norm_ratio_to_total",
        "feat_y": "backbone_kurtosis_max",
        "hypothesis": "attacker suppressing head ratio must accept elevated backbone kurtosis",
    },
    {
        "name": "Squeeze 3: centroid alignment <-> temporal consistency",
        "feat_x": "cosine_to_round_mean_update",
        "feat_y": None,  # Temporal; handled specially
        "hypothesis": "attacker too centroid-aligned OR inconsistent across rounds",
    },
    {
        "name": "Squeeze 4: layer4 kurtosis <-> layer4 l2",
        "feat_x": "stage_layer4_kurtosis",
        "feat_y": "stage_layer4_l2_norm",
        "hypothesis": "attacker smoothing kurtosis must raise l2, and vice versa",
    },
    {
        "name": "Squeeze 5: head spectral entropy <-> head linf/l2",
        "feat_x": "stage_head_spectral_entropy",
        "feat_y": "stage_head_linf_to_l2_ratio",
        "hypothesis": "attacker masking rank-1 structure creates outlier rows",
    },
]

# The 15 new squeeze-targeting features added in the extension
NEW_FEATURES = [
    "cosine_to_mean_leave_one_out",
    "stage_stem_cos_to_mean_loo",
    "stage_layer1_cos_to_mean_loo",
    "stage_layer2_cos_to_mean_loo",
    "stage_layer3_cos_to_mean_loo",
    "stage_layer4_cos_to_mean_loo",
    "stage_head_cos_to_mean_loo",
    "backbone_kurtosis_max",
    "head_backbone_conservation_product",
    "head_sign_agreement_with_ref",
    "stage_pair_stem_to_layer1_projected_cos",
    "stage_pair_layer1_to_layer2_projected_cos",
    "stage_pair_layer2_to_layer3_projected_cos",
    "stage_pair_layer3_to_layer4_projected_cos",
    "stage_pair_layer4_to_head_projected_cos",
]


def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {path}: {len(df)} rows, {df['round'].nunique()} rounds, "
          f"{df['partition_id'].nunique()} unique partitions")
    n_mal = df["malicious_flag"].sum()
    n_hon = (1 - df["malicious_flag"]).sum()
    print(f"  honest updates: {int(n_hon)}, malicious updates: {int(n_mal)} "
          f"({100*n_mal/len(df):.1f}%)")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all numeric columns that are actual features (not metadata)."""
    meta_cols = {
        "timestamp", "simulation_id", "round", "cid", "partition_id",
        "malicious_flag", "attack_mode", "attack_type", "local_data_size",
        "local_epochs", "local_lr", "scale_factor", "selected_by_aggregator",
        "krum_score", "krum_rank", "dirichlet_alpha", "target_label",
        "aggregation_method", "num_clients",
    }
    feat_cols = [c for c in df.columns
                 if c not in meta_cols and df[c].dtype in ('float64', 'float32', 'int64')]
    return feat_cols


def zscore_within_rounds(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Return a copy of df where feat_cols are Z-scored within each round."""
    df_z = df[["round", "malicious_flag", "partition_id"] + feat_cols].copy()
    for col in feat_cols:
        round_stats = df_z.groupby("round")[col].agg(["mean", "std"])
        for rnd, row in round_stats.iterrows():
            mask = df_z["round"] == rnd
            std = row["std"] if row["std"] > 1e-12 else 1.0
            df_z.loc[mask, col] = (df_z.loc[mask, col] - row["mean"]) / std
    return df_z


def cohen_d_and_mwu(h: pd.Series, m: pd.Series):
    """Cohen's d + Mann-Whitney U p-value + rank-biserial correlation."""
    from scipy import stats as sp_stats
    h = h.dropna()
    m = m.dropna()
    if h.empty or m.empty:
        return None, None, None
    pooled_std = np.sqrt((h.var(ddof=0) + m.var(ddof=0)) / 2 + 1e-12)
    d = abs(h.mean() - m.mean()) / (pooled_std + 1e-12)
    try:
        u_stat, p_val = sp_stats.mannwhitneyu(h, m, alternative="two-sided")
        rank_biserial = 1 - 2 * u_stat / (len(h) * len(m))
    except Exception:
        p_val = 1.0
        rank_biserial = 0.0
    return d, abs(rank_biserial), p_val


# =========================================================================
# A. Per-round Z-scored separation
# =========================================================================

def analysis_zscored_separation(df: pd.DataFrame, feat_cols: list):
    print("\n" + "=" * 70)
    print("A. PER-ROUND Z-SCORED FEATURE SEPARATION")
    print("=" * 70)
    print("  (Z-scoring within each round neutralizes round-level model drift)")

    df_z = zscore_within_rounds(df, feat_cols)
    honest_z = df_z[df_z["malicious_flag"] == 0]
    mal_z = df_z[df_z["malicious_flag"] == 1]

    results = []
    for col in feat_cols:
        h = honest_z[col]
        m = mal_z[col]
        d, rb, p = cohen_d_and_mwu(h, m)
        if d is None:
            continue
        results.append({
            "feature": col,
            "honest_z_mean": h.dropna().mean(),
            "mal_z_mean": m.dropna().mean(),
            "cohen_d": d,
            "rank_biserial": rb,
            "p_value": p,
            "is_new": col in NEW_FEATURES,
        })

    results_df = pd.DataFrame(results).sort_values("cohen_d", ascending=False)

    print(f"\n  Top 25 features by Cohen's d (Z-scored within rounds):\n")
    print(f"  {'feature':<50s} {'h_zmean':>8s} {'m_zmean':>8s} {'d':>7s} {'|r_b|':>7s} {'p':>10s} new?")
    print(f"  {'-'*50:<50s} {'-'*8:>8s} {'-'*8:>8s} {'-'*7:>7s} {'-'*7:>7s} {'-'*10:>10s} ----")
    for _, row in results_df.head(25).iterrows():
        sig = " **" if row["p_value"] < 0.01 else (" *" if row["p_value"] < 0.05 else "")
        new_marker = " NEW" if row["is_new"] else ""
        print(f"  {row['feature']:<50s} {row['honest_z_mean']:>+8.3f} {row['mal_z_mean']:>+8.3f} "
              f"{row['cohen_d']:>7.3f} {row['rank_biserial']:>7.3f} {row['p_value']:>10.4g}{sig}{new_marker}")

    sig_count = (results_df["p_value"] < 0.05).sum()
    strong_count = (results_df["cohen_d"] > 0.5).sum()
    new_strong = ((results_df["cohen_d"] > 0.5) & (results_df["is_new"])).sum()
    print(f"\n  Summary: {sig_count} features p<0.05, {strong_count} with d>0.5 "
          f"({new_strong} of the {len(NEW_FEATURES)} new features have d>0.5)")
    return results_df


# =========================================================================
# B. Raw separation
# =========================================================================

def analysis_raw_separation(df: pd.DataFrame, feat_cols: list):
    print("\n" + "=" * 70)
    print("B. RAW FEATURE SEPARATION (ALL FEATURES, honest vs malicious)")
    print("=" * 70)

    honest = df[df["malicious_flag"] == 0]
    mal = df[df["malicious_flag"] == 1]

    results = []
    for col in feat_cols:
        h = honest[col].dropna()
        m = mal[col].dropna()
        if h.empty or m.empty:
            continue
        pooled_std = np.sqrt((h.var(ddof=0) + m.var(ddof=0)) / 2 + 1e-12)
        d = abs(h.mean() - m.mean()) / (pooled_std + 1e-12)
        direction = "mal>" if m.mean() > h.mean() else "mal<"
        results.append({
            "feature": col, "h_mean": h.mean(), "m_mean": m.mean(),
            "cohen_d": d, "direction": direction, "is_new": col in NEW_FEATURES,
        })

    results_df = pd.DataFrame(results).sort_values("cohen_d", ascending=False)
    print(f"\n  Top 20 features by raw Cohen's d:\n")
    print(f"  {'feature':<50s} {'h_mean':>10s} {'m_mean':>10s} {'d':>7s} {'dir':>5s} new?")
    print(f"  {'-'*50:<50s} {'-'*10:>10s} {'-'*10:>10s} {'-'*7:>7s} {'-'*5:>5s} ----")
    for _, row in results_df.head(20).iterrows():
        new_marker = " NEW" if row["is_new"] else ""
        print(f"  {row['feature']:<50s} {row['h_mean']:>10.4f} {row['m_mean']:>10.4f} "
              f"{row['cohen_d']:>7.3f} {row['direction']:>5s}{new_marker}")
    return results_df


# =========================================================================
# C. Temporal consistency — now computed on ALL features
# =========================================================================

def analysis_temporal_consistency(df: pd.DataFrame, feat_cols: list):
    print("\n" + "=" * 70)
    print("C. TEMPORAL CONSISTENCY (per-partition across rounds, ALL features)")
    print("=" * 70)

    pid_counts = df.groupby("partition_id")["round"].count()
    multi_round_pids = pid_counts[pid_counts >= 2].index
    df_multi = df[df["partition_id"].isin(multi_round_pids)].copy()

    if df_multi.empty:
        print("  Not enough multi-round partitions for temporal analysis.")
        return None

    pid_mal = df_multi.groupby("partition_id")["malicious_flag"].max()
    print(f"  Multi-round partitions: {len(multi_round_pids)} "
          f"(honest: {(pid_mal == 0).sum()}, malicious: {(pid_mal == 1).sum()})")

    df_z = zscore_within_rounds(df_multi, feat_cols)

    # Per partition: compute temporal stats on ALL features (was feat_cols[:20])
    temporal_stats = []
    for pid, group in df_z.groupby("partition_id"):
        is_mal = int(pid_mal.loc[pid])
        row = {"partition_id": pid, "malicious": is_mal, "n_appearances": len(group)}
        group_sorted = group.sort_values("round")
        for col in feat_cols:
            vals = group_sorted[col].dropna().values
            if len(vals) < 2:
                continue
            row[f"{col}__mean_z"] = np.mean(vals)
            row[f"{col}__std_z"] = np.std(vals, ddof=0)
            if len(vals) >= 3:
                x = np.arange(len(vals), dtype=float)
                row[f"{col}__slope"] = np.polyfit(x, vals, 1)[0]
        temporal_stats.append(row)

    tdf = pd.DataFrame(temporal_stats)
    honest_t = tdf[tdf["malicious"] == 0]
    mal_t = tdf[tdf["malicious"] == 1]
    if mal_t.empty:
        print("  No multi-round malicious partitions found.")
        return None

    temporal_cols = [c for c in tdf.columns
                     if "__" in c and tdf[c].dtype in ('float64', 'float32')]
    results = []
    for col in temporal_cols:
        d, _, _ = cohen_d_and_mwu(honest_t[col], mal_t[col])
        if d is None:
            continue
        h_mean = honest_t[col].dropna().mean()
        m_mean = mal_t[col].dropna().mean()
        underlying = col.rsplit("__", 1)[0]
        results.append({
            "temporal_feature": col, "h_mean": h_mean, "m_mean": m_mean, "d": d,
            "is_new": underlying in NEW_FEATURES,
        })

    if not results:
        print("  No temporal features computed.")
        return None

    results_df = pd.DataFrame(results).sort_values("d", ascending=False)
    print(f"\n  Top 20 temporal features by Cohen's d:\n")
    print(f"  {'temporal_feature':<60s} {'h_mean':>8s} {'m_mean':>8s} {'d':>7s} new?")
    print(f"  {'-'*60:<60s} {'-'*8:>8s} {'-'*8:>8s} {'-'*7:>7s} ----")
    for _, row in results_df.head(20).iterrows():
        new_marker = " NEW" if row["is_new"] else ""
        print(f"  {row['temporal_feature']:<60s} {row['h_mean']:>+8.3f} "
              f"{row['m_mean']:>+8.3f} {row['d']:>7.3f}{new_marker}")

    strong = (results_df["d"] > 0.5).sum()
    new_strong = ((results_df["d"] > 0.5) & (results_df["is_new"])).sum()
    print(f"\n  Temporal features with d > 0.5: {strong} "
          f"({new_strong} involve new features)")
    return results_df


# =========================================================================
# D. Krum
# =========================================================================

def analysis_krum_correlation(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("D. KRUM SCORE ANALYSIS")
    print("=" * 70)
    honest = df[df["malicious_flag"] == 0]
    mal = df[df["malicious_flag"] == 1]
    print(f"\n  Krum score: honest mean={honest['krum_score'].mean():.4f}, "
          f"mal mean={mal['krum_score'].mean():.4f}")
    winners = df[df["selected_by_aggregator"] == 1]
    mal_wins = winners["malicious_flag"].sum()
    print(f"  Krum winner is malicious: {int(mal_wins)}/{len(winners)} "
          f"({100*mal_wins/max(1,len(winners)):.1f}%)")
    print(f"  Malicious rank distribution: mean={mal['krum_rank'].mean():.1f}, "
          f"median={mal['krum_rank'].median():.0f}")
    mal_top3 = (mal["krum_rank"] <= 3).sum()
    print(f"  Malicious in top-3: {int(mal_top3)}/{len(mal)} "
          f"({100*mal_top3/max(1,len(mal)):.1f}%)")


# =========================================================================
# E. ASR check
# =========================================================================

def analysis_asr_check(csv_path: str):
    print("\n" + "=" * 70)
    print("E. ATTACK EFFECTIVENESS CHECK")
    print("=" * 70)
    central_path = os.path.join(os.path.dirname(csv_path) or ".",
                                "per_round_centralized.csv")
    if not os.path.exists(central_path):
        print(f"  per_round_centralized.csv not found at {central_path}")
        return
    cdf = pd.read_csv(central_path)
    last = cdf.tail(5)
    print(f"\n  Last 5 rounds:")
    print(f"  {'round':>6s} {'MTA':>8s} {'ASR':>8s}")
    for _, row in last.iterrows():
        print(f"  {int(row.get('round', 0)):>6d} "
              f"{row.get('centralized_mta', 0):>8.4f} "
              f"{row.get('centralized_asr', 0):>8.4f}")
    if "centralized_asr" in cdf.columns:
        final_asr = cdf["centralized_asr"].iloc[-1]
        if final_asr < 0.1:
            print(f"\n  >> ASR = {final_asr:.4f} (< 10%). Attack signal not yet embedded.")
        elif final_asr > 0.7:
            print(f"\n  >> ASR = {final_asr:.4f} (> 70%). Attack has fully embedded.")
            print("  >> Features that still separate are catching the attack DESPITE successful camouflage.")


# =========================================================================
# F. New features highlight
# =========================================================================

def analysis_new_features_highlight(df: pd.DataFrame, feat_cols: list,
                                     z_results):
    print("\n" + "=" * 70)
    print("F. NEW FEATURES HIGHLIGHT (15 squeeze-targeting features)")
    print("=" * 70)

    new_present = [f for f in NEW_FEATURES if f in feat_cols]
    missing = [f for f in NEW_FEATURES if f not in feat_cols]
    if missing:
        print(f"\n  MISSING new features (old CSV?): {missing}")
        print(f"  Re-run with updated metrics_extractor.py to get these.")
    print(f"\n  New features present: {len(new_present)}/{len(NEW_FEATURES)}")

    if z_results is None or len(z_results) == 0:
        return

    print(f"\n  Per-round Z-scored separation for NEW features:\n")
    print(f"  {'feature':<50s} {'h_zmean':>8s} {'m_zmean':>8s} {'d':>7s} {'rank':>6s}")
    print(f"  {'-'*50:<50s} {'-'*8:>8s} {'-'*8:>8s} {'-'*7:>7s} {'-'*6:>6s}")

    z_ranked = z_results.reset_index(drop=True).copy()
    z_ranked["rank"] = z_ranked.index + 1
    new_only = z_ranked[z_ranked["is_new"]].sort_values("cohen_d", ascending=False)
    for _, row in new_only.iterrows():
        print(f"  {row['feature']:<50s} {row['honest_z_mean']:>+8.3f} "
              f"{row['mal_z_mean']:>+8.3f} {row['cohen_d']:>7.3f} "
              f"{int(row['rank']):>4d}/{len(z_ranked)}")


# =========================================================================
# G. Squeeze pair correlation test
# =========================================================================

def analysis_squeeze_correlation(df: pd.DataFrame, feat_cols: list):
    print("\n" + "=" * 70)
    print("G. SQUEEZE PAIR CORRELATION TEST")
    print("=" * 70)
    print("""
  For each squeeze pair, we test: does the honest population follow a natural
  correlation (the "allowed" trajectory), and do malicious updates systematically
  lie OFF that trajectory? The 2D Cohen's d measures joint separation; when
  it's much larger than max(1D d for either feature), the squeeze is genuine —
  the pair catches what neither feature catches alone.
""")

    honest = df[df["malicious_flag"] == 0]
    mal = df[df["malicious_flag"] == 1]

    for sq in SQUEEZE_PAIRS:
        print(f"\n  {sq['name']}")
        print(f"    Hypothesis: {sq['hypothesis']}")
        if sq["feat_y"] is None:
            print(f"    [temporal squeeze; see section C for std_z patterns]")
            continue
        fx, fy = sq["feat_x"], sq["feat_y"]
        if fx not in feat_cols or fy not in feat_cols:
            print(f"    SKIP: feature(s) missing ({fx} or {fy})")
            continue

        h_xy = honest[[fx, fy]].dropna()
        m_xy = mal[[fx, fy]].dropna()
        if len(h_xy) < 10 or len(m_xy) < 5:
            print(f"    SKIP: insufficient data (n_honest={len(h_xy)}, n_mal={len(m_xy)})")
            continue

        d_x, _, _ = cohen_d_and_mwu(h_xy[fx], m_xy[fx])
        d_y, _, _ = cohen_d_and_mwu(h_xy[fy], m_xy[fy])
        rho_h = h_xy[fx].corr(h_xy[fy])
        rho_m = m_xy[fx].corr(m_xy[fy])

        mean_h = h_xy.mean().values
        mean_m = m_xy.mean().values
        cov_h = h_xy.cov().values
        cov_m = m_xy.cov().values
        n_h, n_m = len(h_xy), len(m_xy)
        pooled_cov = ((n_h - 1) * cov_h + (n_m - 1) * cov_m) / (n_h + n_m - 2)
        try:
            inv_cov = np.linalg.pinv(pooled_cov + 1e-12 * np.eye(2))
            diff = mean_h - mean_m
            d_2d = float(np.sqrt(diff @ inv_cov @ diff))
        except Exception:
            d_2d = float("nan")

        best_1d = max(d_x or 0, d_y or 0)
        gain = d_2d / (best_1d + 1e-12) if not math.isnan(d_2d) else 0.0

        print(f"    {fx}: d={d_x:.3f}")
        print(f"    {fy}: d={d_y:.3f}")
        print(f"    honest rho({fx}, {fy}) = {rho_h:+.3f}")
        print(f"    malicious rho({fx}, {fy}) = {rho_m:+.3f}")
        print(f"    2D Cohen's d (Mahalanobis): {d_2d:.3f}")
        verdict = ("SQUEEZE VALIDATED" if gain > 1.2 else
                   ("weak squeeze signal" if gain > 0.9 else "NO squeeze signal"))
        print(f"    2D/best-1D ratio: {gain:.2f} -> {verdict}")


def main(csv_path: str):
    df = load_and_validate(csv_path)
    feat_cols = get_feature_cols(df)
    print(f"  Feature columns: {len(feat_cols)}")

    analysis_krum_correlation(df)
    analysis_asr_check(csv_path)
    z_results = analysis_zscored_separation(df, feat_cols)
    raw_results = analysis_raw_separation(df, feat_cols)
    temp_results = analysis_temporal_consistency(df, feat_cols)
    analysis_new_features_highlight(df, feat_cols, z_results)
    analysis_squeeze_correlation(df, feat_cols)

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
  Priority reading order:
    1. Section E: is the attack working? (ASR > 70% = fully embedded)
    2. Section A: which per-round features separate? (d > 0.5 = signal)
    3. Section F: do the new squeeze-targeting features help?
    4. Section G: do the theoretical squeezes hold empirically?
       (2D Cohen's d > 1.2 * max(1D d) = squeeze is catching what
        single features miss)
    5. Section C: does the temporal signal exist? (Principle 1 validation)

  What to look for in Section G:
    - Squeeze VALIDATED on 2-3+ pairs => strong multi-feature joint constraint
      argument for the paper
    - Squeeze validated on Squeeze 2 (head-backbone conservation) specifically
      => Principle 2 (dual-objective leakage) is directly observable
    - No squeeze validated => the attack is successfully gaming joint
      distributions; temporal model is your only leverage
    """)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "per_update_features.csv"
    main(path)