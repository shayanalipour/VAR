# RQ2 step 2: compute VAR (with Optuna tuning) and all baseline credibility scores


import os
import logging
from datetime import datetime

import warnings

warnings.filterwarnings("ignore")

import yaml
import numpy as np
import pandas as pd
import optuna
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from sklearn.metrics import ndcg_score
import networkx as nx

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# VAR
# =============================================================================
def calculate_var_scores(df, lambda_smooth=100.0, beta_decay=0.0):
    """
    VAR_u = sum_i [ correct_i * class_weight_i * d_time_i ]
            / ( sum_i d_time_i + lambda )

    class_weight(s, l) = log(|I_s| / |I_{s,l}|)  — rewards minority-class accuracy
    d_time_i = exp(-beta * (t_max - t_i) / T)     — recency weighting
    lambda                                          — Bayesian smoothing toward 0
    """
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
            total_votes=("item_id", "count"),
            effective_n=("d_time", "sum"),
            weighted_score_sum=("score_contribution", "sum"),
            raw_correct=("is_correct", "sum"),
        )
        .reset_index()
    )
    out["VAR_score"] = out["weighted_score_sum"] / (out["effective_n"] + lambda_smooth)
    out["accuracy"] = out["raw_correct"] / out["total_votes"]
    return out


# =============================================================================
# BASELINES
# =============================================================================
def calculate_hits_scores(df, max_iter=100, tol=1e-8):
    df = df.copy()
    df["is_correct"] = (df["vote"] == df["label"]).astype(int)
    rows = []
    for community, group in df.groupby("community"):
        users, items = group["username"].unique(), group["item_id"].unique()
        if len(users) < 2 or len(items) < 2:
            continue
        u_idx = {u: i for i, u in enumerate(users)}
        i_idx = {it: i for i, it in enumerate(items)}
        L = csr_matrix(
            (
                group["is_correct"].values.astype(float),
                (
                    group["username"].map(u_idx).values,
                    group["item_id"].map(i_idx).values,
                ),
            ),
            shape=(len(users), len(items)),
        )
        auth = np.ones(len(users))
        for _ in range(max_iter):
            hub = L.T.dot(auth)
            if (n := np.linalg.norm(hub)) > 0:
                hub /= n
            auth_new = L.dot(hub)
            if (n := np.linalg.norm(auth_new)) > 0:
                auth_new /= n
            if np.linalg.norm(auth_new - auth) < tol:
                break
            auth = auth_new
        for user, idx in u_idx.items():
            rows.append(
                {"community": community, "username": user, "HITS_score": auth[idx]}
            )
    return pd.DataFrame(rows)


def calculate_balanced_accuracy_scores(df):
    """Balanced accuracy (TPR + TNR) / 2 — Dawid-Skene."""
    df = df.copy()
    df["is_TP"] = ((df["vote"] == 1) & (df["label"] == 1)).astype(int)
    df["is_TN"] = ((df["vote"] == -1) & (df["label"] == -1)).astype(int)
    df["is_P"] = (df["label"] == 1).astype(int)
    df["is_N"] = (df["label"] == -1).astype(int)
    s = (
        df.groupby(["community", "username"])
        .agg(
            TP=("is_TP", "sum"),
            TN=("is_TN", "sum"),
            P=("is_P", "sum"),
            N=("is_N", "sum"),
        )
        .reset_index()
    )
    s["TPR"] = np.where(s["P"] > 0, s["TP"] / s["P"], 0)
    s["TNR"] = np.where(s["N"] > 0, s["TN"] / s["N"], 0)
    s["BalAcc_score"] = (s["TPR"] + s["TNR"]) / 2.0
    return s[["community", "username", "BalAcc_score"]]


def calculate_glad_scores(df):
    """1PL IRT (Rasch): user ability estimated via L-BFGS-B."""
    df = df.copy()
    df["is_correct"] = (df["vote"] == df["label"]).astype(int)
    rows = []
    for community, group in df.groupby("community"):
        users, items = group["username"].unique(), group["item_id"].unique()
        n_u, n_i = len(users), len(items)
        u_idx = {u: i for i, u in enumerate(users)}
        i_idx = {it: i + n_u for i, it in enumerate(items)}
        u_ind = group["username"].map(u_idx).values
        i_ind = group["item_id"].map(i_idx).values
        y = group["is_correct"].values

        def nll(params):
            logits = params[u_ind] - params[i_ind]
            return np.sum(np.logaddexp(0, -logits) + (1 - y) * logits) + 0.01 * np.sum(
                params**2
            )

        try:
            res = minimize(
                nll, np.zeros(n_u + n_i), method="L-BFGS-B", options={"maxiter": 50}
            )
            alphas = res.x[:n_u]
        except Exception:
            alphas = np.zeros(n_u)

        for user, idx in u_idx.items():
            rows.append(
                {"community": community, "username": user, "GLAD_score": alphas[idx]}
            )
    return pd.DataFrame(rows)


def calculate_irt2pl_scores(df):
    """2PL IRT: user ability + item discrimination estimated via L-BFGS-B."""
    df = df.copy()
    df["is_correct"] = (df["vote"] == df["label"]).astype(int)
    rows = []
    for community, group in df.groupby("community"):
        users, items = group["username"].unique(), group["item_id"].unique()
        n_u, n_i = len(users), len(items)
        u_idx = {u: i for i, u in enumerate(users)}
        i_idx = {it: i for i, it in enumerate(items)}
        u_ind = group["username"].map(u_idx).values
        i_ind = group["item_id"].map(i_idx).values
        y = group["is_correct"].values

        def nll(params):
            theta, b, a = params[:n_u], params[n_u : n_u + n_i], params[n_u + n_i :]
            logits = a[i_ind] * (theta[u_ind] - b[i_ind])
            return np.sum(np.logaddexp(0, -logits) + (1 - y) * logits) + 0.1 * (
                np.sum(theta**2) + np.sum(b**2) + np.sum((a - 1) ** 2)
            )

        init = np.zeros(n_u + 2 * n_i)
        init[n_u + n_i :] = 1.0
        bounds = [(None, None)] * n_u + [(None, None)] * n_i + [(0.01, 5.0)] * n_i
        try:
            res = minimize(
                nll, init, bounds=bounds, method="L-BFGS-B", options={"maxiter": 50}
            )
            thetas = res.x[:n_u]
        except Exception:
            thetas = np.zeros(n_u)

        for user, idx in u_idx.items():
            rows.append(
                {"community": community, "username": user, "IRT2PL_score": thetas[idx]}
            )
    return pd.DataFrame(rows)


def calculate_pagerank_scores(df):
    df = df.copy()
    df["is_correct"] = (df["vote"] == df["label"]).astype(int)
    df["edge_weight"] = np.where(df["is_correct"] == 1, 1.0, 0.01)
    rows = []
    for community, group in df.groupby("community"):
        G = nx.Graph()
        for _, row in group.iterrows():
            G.add_edge(
                f"U_{row['username']}", f"I_{row['item_id']}", weight=row["edge_weight"]
            )
        try:
            pr = nx.pagerank(G, alpha=0.85, weight="weight")
        except Exception:
            pr = {}
        for user in group["username"].unique():
            rows.append(
                {
                    "community": community,
                    "username": user,
                    "PageRank_score": pr.get(f"U_{user}", 0.0),
                }
            )
    return pd.DataFrame(rows)


def calculate_volume_scores(df):
    return df.groupby(["community", "username"]).size().reset_index(name="Volume_score")


def calculate_minority_precision_scores(df):
    """Minority Precision: P(label=-1 | vote=-1) — precision of downvotes on actual removals."""
    df = df[df["vote"] == -1].copy()
    df["is_correct"] = (df["label"] == -1).astype(int)
    s = (
        df.groupby(["community", "username"])["is_correct"]
        .agg(["sum", "count"])
        .reset_index()
    )
    s["MinorityPrec_score"] = s["sum"] / s["count"]
    return s[["community", "username", "MinorityPrec_score"]]


def calculate_random_scores(df, seed):
    rng = np.random.default_rng(seed)
    users = df[["community", "username"]].drop_duplicates().copy()
    users["Random_score"] = rng.random(len(users))
    return users


# =============================================================================
# VAR TUNING UTILITIES
# =============================================================================
def build_test_relevance(df):
    df = df.copy()
    df["is_violation_catch"] = ((df["vote"] == -1) & (df["label"] == -1)).astype(int)
    return (
        df.groupby(["community", "username"])["is_violation_catch"]
        .sum()
        .reset_index()
        .rename(columns={"is_violation_catch": "test_relevance"})
    )


def mean_ndcg_at_k(eval_df, score_col, k):
    vals = []
    for _, sub_data in eval_df.groupby("community"):
        if len(sub_data) < 2:
            continue
        y_true = (sub_data["test_relevance"] > 0).astype(int).values
        if y_true.sum() == 0:
            continue
        try:
            vals.append(ndcg_score([y_true], [sub_data[score_col].values], k=k))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else 0.0


def build_inner_split(train_df, split_frac, min_train_votes=3, min_eval_users=10):
    train_df = train_df.sort_values("timestamp")
    split_time = train_df.iloc[min(int(len(train_df) * split_frac), len(train_df) - 1)][
        "timestamp"
    ]
    inner_tr = train_df[train_df["timestamp"] <= split_time]
    inner_val = train_df[train_df["timestamp"] > split_time]
    ucnt = inner_tr["username"].value_counts()
    valid_users = set(ucnt[ucnt >= min_train_votes].index) & set(inner_val["username"])
    inner_tr = inner_tr[inner_tr["username"].isin(valid_users)]
    inner_val = inner_val[inner_val["username"].isin(valid_users)]
    scnt = inner_tr.groupby("community")["username"].nunique()
    valid_subs = scnt[scnt >= min_eval_users].index
    return (
        inner_tr[inner_tr["community"].isin(valid_subs)].copy(),
        inner_val[inner_val["community"].isin(valid_subs)].copy(),
    )


def tune_var(
    inner_tr, inner_val, n_trials, tuning_k, lambda_bounds, beta_bounds, seed, logger
):
    val_perf = build_test_relevance(inner_val)

    def objective(trial):
        lam = trial.suggest_float("lambda_smooth", *lambda_bounds)
        beta = trial.suggest_float("beta_decay", *beta_bounds)
        scores = calculate_var_scores(inner_tr, lambda_smooth=lam, beta_decay=beta)
        ev = val_perf.merge(
            scores[["community", "username", "VAR_score"]],
            on=["community", "username"],
            how="left",
        )
        ev["VAR_score"] = ev["VAR_score"].fillna(0)
        return mean_ndcg_at_k(ev, "VAR_score", k=tuning_k)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    logger.info(
        "VAR tuning done: lambda=%.3f  beta=%.3f  val NDCG@%d=%.4f",
        best["lambda_smooth"],
        best["beta_decay"],
        tuning_k,
        study.best_value,
    )
    return best, study


def run_sensitivity_grid(inner_tr, inner_val, lambdas, betas, k, logger):
    """Evaluate VAR over (lambda, beta) grid on inner val set."""
    val_perf = build_test_relevance(inner_val)
    rows = []
    total = len(lambdas) * len(betas)
    logger.info("running sensitivity grid (%d combinations) …", total)
    for lam in lambdas:
        for beta in betas:
            scores = calculate_var_scores(inner_tr, lambda_smooth=lam, beta_decay=beta)
            ev = val_perf.merge(
                scores[["community", "username", "VAR_score"]],
                on=["community", "username"],
                how="left",
            )
            ev["VAR_score"] = ev["VAR_score"].fillna(0)
            for metric in ("ndcg", "precision", "recall"):
                v = (
                    mean_ndcg_at_k(ev, "VAR_score", k)
                    if metric == "ndcg"
                    else _mean_metric(ev, "VAR_score", k, metric)
                )
                rows.append(
                    {
                        "lambda_smooth": lam,
                        "beta_decay": beta,
                        "k": k,
                        "metric": metric,
                        "value": v,
                    }
                )
    return pd.DataFrame(rows)


def _mean_metric(eval_df, score_col, k, metric):
    vals = []
    for _, sub in eval_df.groupby("community"):
        if len(sub) < 2:
            continue
        y_true = (sub["test_relevance"] > 0).astype(int).values
        if y_true.sum() == 0:
            continue
        top_k = sub.nlargest(k, score_col)
        if metric == "precision":
            vals.append((top_k["test_relevance"] > 0).sum() / k)
        elif metric == "recall":
            vals.append((top_k["test_relevance"] > 0).sum() / y_true.sum())
    return float(np.mean(vals)) if vals else 0.0


# =============================================================================
# MAIN
# =============================================================================
def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    log_dir = config["project"]["log_dir"]
    data_dir = config["project"]["data_dir"]
    cfg = config["rq2"]
    seed = cfg["random_seed"]
    os.makedirs(log_dir, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"step2b_score_{run_timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    train_csv = os.path.join(data_dir, "rq2_train.csv")
    scores_csv = os.path.join(data_dir, "rq2_scores.csv")
    params_out = os.path.join(data_dir, "rq2_best_params.yaml")

    logger.info("loading train set … %s", train_csv)
    train = pd.read_csv(train_csv, dtype={"item_id": "string", "username": "string"})
    train["timestamp"] = (
        pd.to_datetime(train["vote_timestamp"], errors="coerce").astype("int64")
        // 10**9
    )
    train = train.dropna(subset=["timestamp"])
    logger.info("loaded %d rows", len(train))

    logger.info("building inner split for VAR tuning …")
    inner_tr, inner_val = build_inner_split(
        train, split_frac=cfg["inner_val_split_frac"]
    )
    logger.info("inner train: %d | inner val: %d", len(inner_tr), len(inner_val))

    logger.info("tuning VAR with Optuna (%d trials) ...", cfg["optuna_n_trials"])
    best_params, study = tune_var(
        inner_tr,
        inner_val,
        n_trials=cfg["optuna_n_trials"],
        tuning_k=cfg["optuna_tuning_k"],
        lambda_bounds=cfg["lambda_bounds"],
        beta_bounds=cfg["beta_bounds"],
        seed=seed,
        logger=logger,
    )

    with open(params_out, "w") as f:
        yaml.dump(best_params, f)
    logger.info("best params saved -> %s", params_out)

    trials_csv = os.path.join(data_dir, "rq2_optuna_trials.csv")
    trials_df = study.trials_dataframe()[
        ["number", "value", "params_lambda_smooth", "params_beta_decay", "state"]
    ]
    trials_df.rename(columns={"value": f"ndcg@{cfg['optuna_tuning_k']}"}, inplace=True)
    trials_df.to_csv(trials_csv, index=False)
    logger.info("optuna trials saved -> %s", trials_csv)

    sens_df = run_sensitivity_grid(
        inner_tr,
        inner_val,
        lambdas=cfg["lambda_grid"],
        betas=cfg["beta_grid"],
        k=cfg["help_hurt_k"],
        logger=logger,
    )
    sens_csv = os.path.join(data_dir, "rq2_sensitivity.csv")
    sens_df.to_csv(sens_csv, index=False)
    logger.info("sensitivity grid saved -> %s", sens_csv)

    models = [
        (
            "VAR",
            lambda: calculate_var_scores(train, **best_params)[
                ["community", "username", "VAR_score", "accuracy"]
            ],
        ),
        ("HITS", lambda: calculate_hits_scores(train)),
        ("BalAcc", lambda: calculate_balanced_accuracy_scores(train)),
        ("GLAD", lambda: calculate_glad_scores(train)),
        ("IRT2PL", lambda: calculate_irt2pl_scores(train)),
        ("PageRank", lambda: calculate_pagerank_scores(train)),
        ("Volume", lambda: calculate_volume_scores(train)),
        ("MinorityPrec", lambda: calculate_minority_precision_scores(train)),
        ("Random", lambda: calculate_random_scores(train, seed)),
    ]

    scores_df = train[["community", "username"]].drop_duplicates()
    for name, fn in models:
        logger.info("scoring %s …", name)
        s = fn()
        scores_df = scores_df.merge(s, on=["community", "username"], how="left")

    scores_df.to_csv(scores_csv, index=False)
    logger.info("saved scores → %s (%d rows)", scores_csv, len(scores_df))


if __name__ == "__main__":
    main()
