# RQ2 step 1: iterative cleanup and chronological train/test split


import os
import logging
from datetime import datetime

import yaml
import pandas as pd


def iterative_cleanup(df, min_user_votes, min_sub_users, logger):
    """Drop low-activity users and small communities until stable."""
    logger.info("cleanup start: %d rows", len(df))
    while True:
        n_before = len(df)
        ucnt = df["username"].value_counts()
        df = df[df["username"].isin(ucnt[ucnt >= min_user_votes].index)]
        scnt = df.groupby("community")["username"].nunique()
        df = df[df["community"].isin(scnt[scnt >= min_sub_users].index)]
        if len(df) == n_before:
            break
    logger.info(
        "cleanup end: %d rows | %d users | %d communities",
        len(df),
        df["username"].nunique(),
        df["community"].nunique(),
    )
    return df.reset_index(drop=True)


def temporal_split(df, split_frac, min_train_votes, min_eval_users, logger):
    """Chronological split with dense user overlap between train and test."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_time = df.iloc[min(int(len(df) * split_frac), len(df) - 1)]["timestamp"]

    train_raw = df[df["timestamp"] <= split_time]
    test_raw = df[df["timestamp"] > split_time]

    # keep only users with enough train votes who also appear in test
    ucnt = train_raw["username"].value_counts()
    valid_users = set(ucnt[ucnt >= min_train_votes].index) & set(test_raw["username"])

    train = train_raw[train_raw["username"].isin(valid_users)].copy()
    test = test_raw[test_raw["username"].isin(valid_users)].copy()

    # keep only communities with enough eval users in train
    scnt = train.groupby("community")["username"].nunique()
    valid_subs = scnt[scnt >= min_eval_users].index
    train = train[train["community"].isin(valid_subs)]
    test = test[test["community"].isin(valid_subs)]

    logger.info(
        "train: %d rows | test: %d rows | %d users | %d communities",
        len(train),
        len(test),
        train["username"].nunique(),
        train["community"].nunique(),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    log_dir = config["project"]["log_dir"]
    data_dir = config["project"]["data_dir"]
    cfg = config["rq2"]
    os.makedirs(log_dir, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"step2a_prepare_{run_timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    in_csv = os.path.join(data_dir, "modlog_votes.csv")
    train_csv = os.path.join(data_dir, "rq2_train.csv")
    test_csv = os.path.join(data_dir, "rq2_test.csv")

    logger.info("loading %s", in_csv)
    df = pd.read_csv(in_csv, dtype={"item_id": "string", "username": "string"})
    df["timestamp"] = (
        pd.to_datetime(df["vote_timestamp"], errors="coerce").astype("int64") // 10**9
    )
    df = df.dropna(subset=["timestamp", "vote", "label"]).sort_values("timestamp")
    logger.info("loaded %d rows", len(df))

    df = iterative_cleanup(
        df,
        min_user_votes=cfg["min_user_votes"],
        min_sub_users=cfg["min_sub_users"],
        logger=logger,
    )

    train, test = temporal_split(
        df,
        split_frac=cfg["train_split_frac"],
        min_train_votes=cfg["min_train_votes_per_user"],
        min_eval_users=cfg["min_eval_users_per_sub"],
        logger=logger,
    )

    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)
    logger.info("saved train: %s", train_csv)
    logger.info("saved test: %s", test_csv)


if __name__ == "__main__":
    main()
