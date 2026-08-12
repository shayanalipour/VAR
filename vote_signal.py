# RQ1: Are community votes informative or random?
# Computes upvote ratio per post and compares distributions between
# approved vs removed posts using Mann-Whitney U and Cohen's d.

import os
import logging
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
from scipy import stats


def compute_upvote_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate vote-level rows to one row per post with upvote_ratio."""
    agg = (
        df.groupby("item_id")
        .agg(
            community=("community", "first"),
            label=("label", "first"),
            total_votes=("vote", "count"),
            upvotes=("vote", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )
    agg["upvote_ratio"] = agg["upvotes"] / agg["total_votes"]
    return agg


def build_categories(df: pd.DataFrame) -> pd.DataFrame:
    df["mod_action"] = df["label"].map({1: "Approval", -1: "Removal"})
    return df


def run_analysis(df: pd.DataFrame, logger: logging.Logger):
    summary = (
        df.groupby("mod_action")["upvote_ratio"]
        .agg(N="count", Mean="mean", Std="std", Median="median")
        .reindex(["Approval", "Removal"])
        .round(3)
    )
    logger.info("upvote ratio summary by moderation outcome:\n%s", summary.to_string())

    approved = df.loc[df["mod_action"] == "Approval", "upvote_ratio"]
    removed = df.loc[df["mod_action"] == "Removal", "upvote_ratio"]

    n1, n2 = len(approved), len(removed)
    _, p_val = stats.mannwhitneyu(approved, removed, alternative="two-sided")
    pooled_std = np.sqrt(
        ((n1 - 1) * approved.std() ** 2 + (n2 - 1) * removed.std() ** 2) / (n1 + n2 - 2)
    )
    cohens_d = (
        (approved.mean() - removed.mean()) / pooled_std if pooled_std > 0 else np.nan
    )

    logger.info(
        "approved mean=%.3f (N=%d)  removed mean=%.3f (N=%d)  "
        "Mann-Whitney p=%.2e  Cohen's d=%.3f",
        approved.mean(),
        n1,
        removed.mean(),
        n2,
        p_val,
        cohens_d,
    )


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    log_dir = config["project"]["log_dir"]
    data_dir = config["project"]["data_dir"]
    os.makedirs(log_dir, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"step1a_rq1_vote_signal_{run_timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    votes_csv = os.path.join(data_dir, "modlog_votes.csv")
    logger.info("loading %s", votes_csv)
    df = pd.read_csv(votes_csv, dtype={"item_id": "string", "username": "string"})
    logger.info("loaded %d vote rows across %d posts", len(df), df["item_id"].nunique())

    post_df = compute_upvote_ratio(df)
    post_df = build_categories(post_df)
    logger.info("post-level dataset: %d rows", len(post_df))
    logger.info("outcome counts:\n%s", post_df["mod_action"].value_counts().to_string())

    run_analysis(post_df, logger)


if __name__ == "__main__":
    main()
