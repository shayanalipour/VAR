# RQ3: How consistent is expert performance across different community contexts?

import os
import logging
from datetime import datetime

import warnings

warnings.filterwarnings("ignore")

import yaml
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def calculate_var_scores(df, lambda_smooth, beta_decay):
    df = df[[c for c in df.columns if c != "weight"]].copy()

    if beta_decay > 0:
        t_max, t_min = df["timestamp"].max(), df["timestamp"].min()
        T = (t_max - t_min) if t_max > t_min else 1.0
        df["d_time"] = np.exp(-beta_decay * (t_max - df["timestamp"]) / T)
    else:
        df["d_time"] = 1.0

    sub_counts = df.groupby(["community", "label"]).size().reset_index(name="count")
    sub_totals = df.groupby("community").size().reset_index(name="total")
    weights = sub_counts.merge(sub_totals, on="community")
    weights["weight"] = np.log(weights["total"] / weights["count"])

    df = df.merge(
        weights[["community", "label", "weight"]], on=["community", "label"], how="left"
    )
    df["is_correct"] = (df["vote"] == df["label"]).astype(int)
    df["score_contribution"] = df["is_correct"] * df["weight"] * df["d_time"]

    out = (
        df.groupby(["community", "username"])
        .agg(
            effective_n=("d_time", "sum"),
            weighted_score_sum=("score_contribution", "sum"),
        )
        .reset_index()
    )
    out["VAR_score"] = out["weighted_score_sum"] / (out["effective_n"] + lambda_smooth)
    return out[["community", "username", "VAR_score"]]


def normalize_within_community(df):
    """Z-score VAR within each community so scales are comparable."""
    df = df.copy()
    df["VAR_z"] = df.groupby("community")["VAR_score"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    return df


def build_correlation_matrix(df, min_overlap):
    """Pivot to user x community matrix and compute pairwise Pearson r."""
    matrix = df.pivot_table(index="username", columns="community", values="VAR_z")
    corr = matrix.corr(method="pearson", min_periods=min_overlap)
    return corr, matrix


def hierarchical_order(corr_matrix):
    """Reorder communities by average-linkage clustering on 1-|r|."""
    dist = 1.0 - corr_matrix.abs().fillna(0).values
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    try:
        Z = linkage(squareform(dist, checks=False), method="average")
        return leaves_list(Z)
    except Exception:
        return np.arange(len(corr_matrix))


def build_pairwise_table(corr_matrix, pivot_matrix, min_overlap):

    cols = corr_matrix.columns.tolist()
    rows = []
    arr = corr_matrix.values

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = arr[i, j]
            if np.isnan(r):
                continue

            # users present in both communities
            col_a = pivot_matrix[cols[i]]
            col_b = pivot_matrix[cols[j]]
            mask = col_a.notna() & col_b.notna()
            n = int(mask.sum())

            if n >= max(2, min_overlap):
                _, p_value = stats.pearsonr(col_a[mask].values, col_b[mask].values)
            else:
                p_value = float("nan")

            rows.append(
                {
                    "community_a": cols[i],
                    "community_b": cols[j],
                    "pearson_r": round(r, 4),
                    "p_value": (
                        round(p_value, 4) if not np.isnan(p_value) else float("nan")
                    ),
                    "n": n,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("pearson_r", ascending=False)
        .reset_index(drop=True)
    )


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    log_dir = config["project"]["log_dir"]
    data_dir = config["project"]["data_dir"]
    figures_dir = config["project"]["figures_dir"]
    cfg = config["rq3"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"step3a_rq3_cross_community_{run_timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    train_csv = os.path.join(data_dir, "rq2_train.csv")
    params_yaml = os.path.join(data_dir, "rq2_best_params.yaml")
    corr_csv = os.path.join(data_dir, "rq3_correlation_matrix.csv")
    pairs_csv = os.path.join(data_dir, "rq3_pairwise_table.csv")

    logger.info("loading train set … %s", train_csv)
    df = pd.read_csv(train_csv, dtype={"item_id": "string", "username": "string"})
    df["timestamp"] = (
        pd.to_datetime(df["vote_timestamp"], errors="coerce").astype("int64") // 10**9
    )
    df = df.dropna(subset=["timestamp", "vote", "label"])
    logger.info(
        "loaded %d rows | %d users | %d communities",
        len(df),
        df["username"].nunique(),
        df["community"].nunique(),
    )

    # load test set to identify evaluable communities
    test_csv = os.path.join(data_dir, "rq2_test.csv")
    logger.info("loading test set for community alignment … %s", test_csv)
    test = pd.read_csv(test_csv, dtype={"item_id": "string", "username": "string"})

    has_violation_catch = (
        test[(test["vote"] == -1) & (test["label"] == -1)]
        .groupby("community")
        .size()
        .reset_index(name="n_catches")
    )
    evaluable_communities = set(has_violation_catch["community"])
    logger.info(
        "evaluable communities (test set, ≥1 violation catch): %d",
        len(evaluable_communities),
    )

    # filter train scores to matching communities only
    df = df[df["community"].isin(evaluable_communities)].copy()
    logger.info(
        "after alignment: %d rows | %d communities",
        len(df),
        df["community"].nunique(),
    )

    with open(params_yaml) as f:
        best_params = yaml.safe_load(f)
    logger.info(
        "VAR params: lambda=%.3f  beta=%.3f",
        best_params["lambda_smooth"],
        best_params["beta_decay"],
    )

    logger.info("computing VAR scores …")
    scores = calculate_var_scores(df, **best_params)
    logger.info("user-community pairs: %d", len(scores))

    scores = normalize_within_community(scores)

    min_overlap = cfg["min_overlap_for_corr"]
    corr_matrix, pivot_matrix = build_correlation_matrix(
        scores, min_overlap=min_overlap
    )
    logger.info("correlation matrix: %d x %d communities", *corr_matrix.shape)

    corr_matrix.round(4).to_csv(corr_csv)
    logger.info("saved correlation matrix → %s", corr_csv)

    pairs_df = build_pairwise_table(corr_matrix, pivot_matrix, min_overlap)
    pairs_df.to_csv(pairs_csv, index=False)
    logger.info("saved pairwise table → %s (%d pairs)", pairs_csv, len(pairs_df))

    # global correlation, just in case
    global_x, global_y = [], []
    cols = pivot_matrix.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a = pivot_matrix[cols[i]]
            col_b = pivot_matrix[cols[j]]
            mask = col_a.notna() & col_b.notna()

            if mask.sum() >= max(2, min_overlap):
                global_x.extend(col_a[mask].values)
                global_y.extend(col_b[mask].values)

    if len(global_x) > 1:
        global_r, global_p = stats.pearsonr(global_x, global_y)
    else:
        global_r, global_p = float("nan"), float("nan")

    # summary stats
    r_vals = pairs_df["pearson_r"].values
    t_stat, p_val = stats.ttest_1samp(r_vals, 0)

    logger.info("pairwise correlation summary:")
    logger.info("  valid pairs:       %d", len(r_vals))
    logger.info("  mean r:            %.4f", np.mean(r_vals))
    logger.info("  median r:          %.4f", np.median(r_vals))
    logger.info("  std r:             %.4f", np.std(r_vals))
    logger.info("  range:             [%.4f, %.4f]", r_vals.min(), r_vals.max())
    logger.info("  %% pairs r > 0:    %.1f%%", (r_vals > 0).mean() * 100)
    logger.info("  t-test vs 0:       t=%.3f  p=%.4e", t_stat, p_val)
    logger.info("  median n (overlap):%.1f", pairs_df["n"].median())
    logger.info("  median pair p-val: %.4f", pairs_df["p_value"].median())
    logger.info("  ---------------------------------")
    logger.info("  global pooled r:   %.4f", global_r)
    logger.info("  global pooled p:   %.4e", global_p)
    logger.info(
        "  global pooled N:   %d (total overlapping user-scores)", len(global_x)
    )

    logger.info("done.")


if __name__ == "__main__":
    main()
