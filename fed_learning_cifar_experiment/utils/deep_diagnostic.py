"""
Deep diagnostic of per_update_features.csv — goes beyond the
pre-flight verifier to surface signal the simple population-mean
comparison misses.

Three analyses:
  A. Per-round Z-scored separation (corrects for round-level drift)
  B. Full 63-feature ranking by separation strength
  C. Temporal consistency: do malicious clients show persistent patterns
     across rounds that honest clients don't?

Usage:
  python deep_diagnostic.py [path/to/per_update_features.csv]
"""

import sys
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


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
    feat_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in ('float64', 'float32', 'int64')]
    return feat_cols


# =========================================================================
# A. Per-round Z-scored separation
# =========================================================================

def analysis_zscored_separation(df: pd.DataFrame, feat_cols: list):
    """
    For each round, Z-score each feature relative to the round's mean/std.
    Then compare Z-scored distributions of honest vs malicious across ALL
    rounds. This corrects for round-level drift in model dynamics.
    """
    print("\n" + "=" * 70)
    print("A. PER-ROUND Z-SCORED FEATURE SEPARATION")
    print("=" * 70)
    print("  (Z-scoring within each round neutralizes round-level model drift)")

    df_z = df[["round", "malicious_flag"] + feat_cols].copy()

    # Z-score within each round
    for col in feat_cols:
        round_stats = df_z.groupby("round")[col].agg(["mean", "std"])
        for rnd, row in round_stats.iterrows():
            mask = df_z["round"] == rnd
            std = row["std"] if row["std"] > 1e-12 else 1.0
            df_z.loc[mask, col] = (df_z.loc[mask, col] - row["mean"]) / std

    honest_z = df_z[df_z["malicious_flag"] == 0]
    mal_z = df_z[df_z["malicious_flag"] == 1]

    results = []
    for col in feat_cols:
        h = honest_z[col].dropna()
        m = mal_z[col].dropna()
        if h.empty or m.empty:
            continue
        # Cohen's d on Z-scored values
        pooled_std = np.sqrt((h.var(ddof=0) + m.var(ddof=0)) / 2 + 1e-12)
        d = abs(h.mean() - m.mean()) / (pooled_std + 1e-12)
        # Also compute the rank-biserial correlation (robust to outliers)
        from scipy import stats as sp_stats
        try:
            u_stat, p_val = sp_stats.mannwhitneyu(h, m, alternative="two-sided")
            rank_biserial = 1 - 2 * u_stat / (len(h) * len(m))
        except Exception:
            p_val = 1.0
            rank_biserial = 0.0
        results.append({
            "feature": col,
            "honest_z_mean": h.mean(),
            "mal_z_mean": m.mean(),
            "cohen_d": d,
            "rank_biserial": abs(rank_biserial),
            "p_value": p_val,
        })

    results_df = pd.DataFrame(results).sort_values("cohen_d", ascending=False)

    print(f"\n  Top 20 features by Cohen's d (Z-scored within rounds):\n")
    print(f"  {'feature':<45s} {'h_zmean':>8s} {'m_zmean':>8s} {'d':>7s} {'|r_b|':>7s} {'p':>10s}")
    print(f"  {'-'*45:<45s} {'-'*8:>8s} {'-'*8:>8s} {'-'*7:>7s} {'-'*7:>7s} {'-'*10:>10s}")
    for _, row in results_df.head(20).iterrows():
        sig = " **" if row["p_value"] < 0.01 else (" *" if row["p_value"] < 0.05 else "")
        print(f"  {row['feature']:<45s} {row['honest_z_mean']:>+8.3f} {row['mal_z_mean']:>+8.3f} "
              f"{row['cohen_d']:>7.3f} {row['rank_biserial']:>7.3f} {row['p_value']:>10.4g}{sig}")

    sig_count = (results_df["p_value"] < 0.05).sum()
    strong_count = (results_df["cohen_d"] > 0.5).sum()
    print(f"\n  Summary: {sig_count} features with p < 0.05, "
          f"{strong_count} with Cohen's d > 0.5")

    if strong_count == 0:
        print("  >> No strong per-round signal. The attack is effectively "
              "camouflaged within each round.")
        print("  >> This validates the temporal analysis hypothesis: "
              "signal accumulates across rounds.")
    return results_df


# =========================================================================
# B. Raw separation (all 63 features, not just 9 candidates)
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
        # Effect direction: positive means malicious is HIGHER
        direction = "mal>" if m.mean() > h.mean() else "mal<"
        results.append({
            "feature": col,
            "h_mean": h.mean(),
            "m_mean": m.mean(),
            "cohen_d": d,
            "direction": direction,
        })

    results_df = pd.DataFrame(results).sort_values("cohen_d", ascending=False)
    print(f"\n  Top 20 features by raw Cohen's d:\n")
    print(f"  {'feature':<45s} {'h_mean':>10s} {'m_mean':>10s} {'d':>7s} {'dir':>5s}")
    print(f"  {'-'*45:<45s} {'-'*10:>10s} {'-'*10:>10s} {'-'*7:>7s} {'-'*5:>5s}")
    for _, row in results_df.head(20).iterrows():
        print(f"  {row['feature']:<45s} {row['h_mean']:>10.4f} {row['m_mean']:>10.4f} "
              f"{row['cohen_d']:>7.3f} {row['direction']:>5s}")
    return results_df


# =========================================================================
# C. Temporal consistency analysis
# =========================================================================

def analysis_temporal_consistency(df: pd.DataFrame, feat_cols: list):
    """
    Even if per-round features overlap between honest and malicious, malicious
    clients might show CONSISTENT PATTERNS across their appearances:
      - Lower variance across rounds (attack is mechanically repeatable)
      - Consistent direction of deviation from round mean
      - Trend (slope) in certain features over time

    This analysis groups by partition_id, computes temporal statistics, and
    compares honest vs malicious partition-level summaries.
    """
    print("\n" + "=" * 70)
    print("C. TEMPORAL CONSISTENCY (per-partition across rounds)")
    print("=" * 70)

    # Only analyze partitions that appear in 2+ rounds
    pid_counts = df.groupby("partition_id")["round"].count()
    multi_round_pids = pid_counts[pid_counts >= 2].index
    df_multi = df[df["partition_id"].isin(multi_round_pids)].copy()

    if df_multi.empty:
        print("  Not enough multi-round partitions for temporal analysis.")
        return None

    # For each partition, is it ever malicious? (partition-level label)
    pid_mal = df_multi.groupby("partition_id")["malicious_flag"].max()
    print(f"  Multi-round partitions: {len(multi_round_pids)} "
          f"(honest: {(pid_mal == 0).sum()}, malicious: {(pid_mal == 1).sum()})")

    # Z-score within rounds first
    df_z = df_multi.copy()
    for col in feat_cols:
        round_stats = df_z.groupby("round")[col].agg(["mean", "std"])
        for rnd, row in round_stats.iterrows():
            mask = df_z["round"] == rnd
            std = row["std"] if row["std"] > 1e-12 else 1.0
            df_z.loc[mask, col] = (df_z.loc[mask, col] - row["mean"]) / std

    # Per partition: compute temporal statistics on Z-scored features
    temporal_stats = []
    for pid, group in df_z.groupby("partition_id"):
        is_mal = int(pid_mal.loc[pid])
        n_appearances = len(group)
        row = {"partition_id": pid, "malicious": is_mal, "n_appearances": n_appearances}

        group_sorted = group.sort_values("round")
        for col in feat_cols[:20]:  # top-20 features to keep output manageable
            vals = group_sorted[col].dropna().values
            if len(vals) < 2:
                continue
            # Mean Z-score (consistent offset from round mean?)
            row[f"{col}__mean_z"] = np.mean(vals)
            # Std of Z-scores (consistency of deviation pattern)
            row[f"{col}__std_z"] = np.std(vals, ddof=0)
            # Trend slope (linear regression over appearances)
            if len(vals) >= 3:
                x = np.arange(len(vals), dtype=float)
                slope = np.polyfit(x, vals, 1)[0]
                row[f"{col}__slope"] = slope
        temporal_stats.append(row)

    tdf = pd.DataFrame(temporal_stats)

    # Compare honest vs malicious on temporal summary stats
    honest_t = tdf[tdf["malicious"] == 0]
    mal_t = tdf[tdf["malicious"] == 1]

    if mal_t.empty:
        print("  No multi-round malicious partitions found.")
        return None

    print(f"\n  Temporal features with strongest honest vs malicious separation:\n")

    temporal_cols = [c for c in tdf.columns if "__" in c and tdf[c].dtype in ('float64', 'float32')]
    results = []
    for col in temporal_cols:
        h = honest_t[col].dropna()
        m = mal_t[col].dropna()
        if h.empty or m.empty:
            continue
        pooled_std = np.sqrt((h.var(ddof=0) + m.var(ddof=0)) / 2 + 1e-12)
        d = abs(h.mean() - m.mean()) / (pooled_std + 1e-12)
        results.append({"temporal_feature": col, "h_mean": h.mean(), "m_mean": m.mean(), "d": d})

    if not results:
        print("  No temporal features computed (insufficient data).")
        return None

    results_df = pd.DataFrame(results).sort_values("d", ascending=False)
    print(f"  {'temporal_feature':<55s} {'h_mean':>8s} {'m_mean':>8s} {'d':>7s}")
    print(f"  {'-'*55:<55s} {'-'*8:>8s} {'-'*8:>8s} {'-'*7:>7s}")
    for _, row in results_df.head(15).iterrows():
        print(f"  {row['temporal_feature']:<55s} {row['h_mean']:>+8.3f} {row['m_mean']:>+8.3f} {row['d']:>7.3f}")

    strong = (results_df["d"] > 0.5).sum()
    print(f"\n  Temporal features with d > 0.5: {strong}")
    if strong > 0:
        print("  >> Temporal consistency signal exists! This supports the "
              "cross-round modeling hypothesis.")
    else:
        print("  >> No strong temporal signal yet. May need more rounds "
              "or the attack may be adapting its temporal signature.")
    return results_df


# =========================================================================
# D. Krum-score correlation: are malicious updates scoring differently?
# =========================================================================

def analysis_krum_correlation(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("D. KRUM SCORE ANALYSIS")
    print("=" * 70)

    honest = df[df["malicious_flag"] == 0]
    mal = df[df["malicious_flag"] == 1]

    print(f"\n  Krum score:  honest mean={honest['krum_score'].mean():.6f} "
          f"std={honest['krum_score'].std():.6f}")
    print(f"  Krum score:  mal    mean={mal['krum_score'].mean():.6f} "
          f"std={mal['krum_score'].std():.6f}")

    # How often is the winner malicious?
    winners = df[df["selected_by_aggregator"] == 1]
    mal_wins = winners["malicious_flag"].sum()
    total_wins = len(winners)
    print(f"\n  Krum winner is malicious: {int(mal_wins)}/{total_wins} rounds "
          f"({100*mal_wins/max(1,total_wins):.1f}%)")

    # Rank distribution
    print(f"\n  Krum rank distribution:")
    print(f"    honest: mean rank = {honest['krum_rank'].mean():.1f}, "
          f"median = {honest['krum_rank'].median():.0f}")
    print(f"    mal:    mean rank = {mal['krum_rank'].mean():.1f}, "
          f"median = {mal['krum_rank'].median():.0f}")

    # How often is malicious in top-3?
    mal_top3 = mal[mal["krum_rank"] <= 3]
    print(f"    malicious in top-3: {len(mal_top3)}/{len(mal)} "
          f"({100*len(mal_top3)/max(1,len(mal)):.1f}%)")


# =========================================================================
# E. ASR diagnostic: check if the attack is even working
# =========================================================================

def analysis_asr_check(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("E. ATTACK EFFECTIVENESS CHECK")
    print("=" * 70)

    # Check per_round_centralized.csv alongside if it exists
    import os
    central_path = os.path.join(os.path.dirname(sys.argv[1] if len(sys.argv) > 1 else "."),
                                "per_round_centralized.csv")
    if os.path.exists(central_path):
        cdf = pd.read_csv(central_path)
        last_rounds = cdf.tail(5)
        print(f"\n  Last 5 rounds from per_round_centralized.csv:")
        print(f"  {'round':>6s} {'MTA':>8s} {'ASR':>8s}")
        for _, row in last_rounds.iterrows():
            print(f"  {int(row.get('round', 0)):>6d} "
                  f"{row.get('centralized_mta', 0):>8.4f} "
                  f"{row.get('centralized_asr', 0):>8.4f}")
        final_asr = cdf["centralized_asr"].iloc[-1] if "centralized_asr" in cdf.columns else None
        if final_asr is not None and final_asr < 0.1:
            print(f"\n  >> ASR = {final_asr:.4f} (< 10%). The attack may not have "
                  "injected enough backdoor signal yet.")
            print("  >> Feature overlap with honest clients is EXPECTED when ASR is low.")
            print("  >> Run longer (50-100 rounds) before concluding features are uninformative.")
    else:
        print(f"  per_round_centralized.csv not found at {central_path}")
        print("  Cannot verify ASR. Check manually whether the attack is succeeding.")


def main(csv_path: str):
    df = load_and_validate(csv_path)
    feat_cols = get_feature_cols(df)
    print(f"  Feature columns: {len(feat_cols)}")

    analysis_krum_correlation(df)
    analysis_asr_check(df)
    z_df = analysis_zscored_separation(df, feat_cols)
    raw_df = analysis_raw_separation(df, feat_cols)
    temp_df = analysis_temporal_consistency(df, feat_cols)

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
  If ASR is low (< 10%):
    The attack hasn't injected enough signal yet. Feature overlap with honest
    is expected. Run longer (50-100 rounds). This is NOT a defense failure.

  If ASR is high but per-round features don't separate:
    The attack is successfully camouflaged within each round. This validates
    the temporal modeling hypothesis. Check section C for cross-round signal.

  If temporal features DO separate (section C, d > 0.5):
    The attack leaks signal across rounds even though per-round features
    overlap. This is exactly the gap your BiGRU temporal model is designed
    to exploit. Proceed with the temporal modeling architecture.

  If nothing separates at all (ASR high, no temporal signal):
    The attack is extremely sophisticated. Consider:
    - Adding more features (gradient-flow features, loss landscape curvature)
    - Checking if alpha=0.9 is too homogeneous (try alpha=0.3 for wider
      honest variance, which might expose attacker outliers by contrast)
    - Running with more rounds (100+) to accumulate temporal signal
    """)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "per_update_features.csv"
    main(path)