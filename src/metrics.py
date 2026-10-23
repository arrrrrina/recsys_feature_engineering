from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from models.als import ALSCandidateModel

def evaluate_recall_at_k(
    model: ALSCandidateModel,
    targets: pd.DataFrame,
    k: int = 200,
    batch_size: int = 2_000,
    show_progress: bool = True,
) -> dict[str, float]:
    """Посчитать средний пользовательский Recall@K."""

    if targets.empty:
        raise ValueError("targets must not be empty")

    actual_by_user = (
        targets.groupby("user_id")["item_id"]
        .agg(set)
        .to_dict()
    )

    target_users = list(actual_by_user)

    known_users = set(model.user_ids.tolist())
    known_items = set(model.item_ids.tolist())

    recall_all_sum = 0.0
    recall_all_count = 0

    recall_warm_sum = 0.0
    recall_warm_count = 0

    warm_user_count = 0
    total_positive_count = 0
    retrievable_positive_count = 0

    batch_starts = range(
        0,
        len(target_users),
        batch_size,
    )

    if show_progress:
        batch_starts = tqdm(
            batch_starts,
            total=(
                len(target_users) + batch_size - 1
            ) // batch_size,
            desc=f"Recall@{k}",
        )

    for start in batch_starts:
        user_batch = target_users[
            start : start + batch_size
        ]

        candidates = model.recommend(
            user_ids=user_batch,
            n=k,
            filter_seen=True,
            batch_size=batch_size,
        )

        if candidates.empty:
            recommended_by_user = {}
        else:
            recommended_by_user = (
                candidates.groupby("user_id")["item_id"]
                .agg(set)
                .to_dict()
            )

        for user_id in user_batch:
            actual_items = actual_by_user[user_id]

            recommended_items = recommended_by_user.get(
                user_id,
                set(),
            )

            hits = len(
                actual_items & recommended_items
            )
            
            # Recall по всем пользователям.
       
            recall_all_sum += (
                hits / len(actual_items)
            )
            recall_all_count += 1
            total_positive_count += len(actual_items)

            # Recall только по warm-start части.
            if user_id in known_users:
                warm_user_count += 1

                warm_actual_items = (
                    actual_items & known_items
                )

                retrievable_positive_count += len(
                    warm_actual_items
                )

                if warm_actual_items:
                    warm_hits = len(
                        warm_actual_items
                        & recommended_items
                    )

                    recall_warm_sum += (
                        warm_hits
                        / len(warm_actual_items)
                    )
                    recall_warm_count += 1

    recall_all = (
        recall_all_sum / recall_all_count
    )

    recall_warm = (
        recall_warm_sum / recall_warm_count
        if recall_warm_count > 0
        else np.nan
    )

    return {
        f"recall@{k}_all": recall_all,
        f"recall@{k}_warm": recall_warm,
        "warm_user_share": (
            warm_user_count / len(target_users)
        ),
        "retrievable_positive_share": (
            retrievable_positive_count
            / total_positive_count
        ),
        "target_users": len(target_users),
        "warm_target_users": warm_user_count,
        "evaluated_warm_users": recall_warm_count,
        "target_positives": total_positive_count,
        "retrievable_positives": (
            retrievable_positive_count
        ),
    }


def evaluate_ndcg_at_k(
    model: ALSCandidateModel,
    targets: pd.DataFrame,
    k: int = 200,
    batch_size: int = 2_000,
    show_progress: bool = True,
) -> dict[str, float]:
    """Calculate mean user NDCG@K for binary implicit-feedback targets.

    ``ndcg@K_all`` includes cold users and cold items, while ``ndcg@K_warm``
    is calculated only for users and positive items known to the fitted model.
    """

    if targets.empty:
        raise ValueError("targets must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    required_columns = {"user_id", "item_id"}
    missing_columns = required_columns.difference(targets.columns)
    if missing_columns:
        raise ValueError(
            f"targets is missing columns: {sorted(missing_columns)}"
        )

    actual_by_user = (
        targets[["user_id", "item_id"]]
        .drop_duplicates()
        .groupby("user_id")["item_id"]
        .agg(set)
        .to_dict()
    )
    target_users = list(actual_by_user)

    known_users = set(model.user_ids.tolist())
    known_items = set(model.item_ids.tolist())

    ndcg_all_sum = 0.0
    ndcg_all_count = 0
    ndcg_warm_sum = 0.0
    ndcg_warm_count = 0
    warm_user_count = 0
    total_positive_count = 0
    retrievable_positive_count = 0

    batch_starts = range(0, len(target_users), batch_size)
    if show_progress:
        batch_starts = tqdm(
            batch_starts,
            total=(len(target_users) + batch_size - 1) // batch_size,
            desc=f"NDCG@{k}",
        )

    for start in batch_starts:
        user_batch = target_users[start : start + batch_size]
        candidates = model.recommend(
            user_ids=user_batch,
            n=k,
            filter_seen=True,
            batch_size=batch_size,
        )

        if candidates.empty:
            recommended_by_user = {}
        else:
            recommended_by_user = (
                candidates.sort_values(["user_id", "als_rank"])
                .groupby("user_id", sort=False)["item_id"]
                .agg(list)
                .to_dict()
            )

        for user_id in user_batch:
            actual_items = actual_by_user[user_id]
            recommended_items = recommended_by_user.get(user_id, [])

            ndcg_all_sum += _binary_ndcg_at_k(
                recommended_items, actual_items, k
            )
            ndcg_all_count += 1
            total_positive_count += len(actual_items)

            if user_id in known_users:
                warm_user_count += 1
                warm_actual_items = actual_items & known_items
                retrievable_positive_count += len(warm_actual_items)

                if warm_actual_items:
                    ndcg_warm_sum += _binary_ndcg_at_k(
                        recommended_items, warm_actual_items, k
                    )
                    ndcg_warm_count += 1

    return {
        f"ndcg@{k}_all": ndcg_all_sum / ndcg_all_count,
        f"ndcg@{k}_warm": (
            ndcg_warm_sum / ndcg_warm_count
            if ndcg_warm_count > 0
            else np.nan
        ),
        "warm_user_share": warm_user_count / len(target_users),
        "retrievable_positive_share": (
            retrievable_positive_count / total_positive_count
        ),
        "target_users": len(target_users),
        "warm_target_users": warm_user_count,
        "evaluated_warm_users": ndcg_warm_count,
        "target_positives": total_positive_count,
        "retrievable_positives": retrievable_positive_count,
    }


def _binary_ndcg_at_k(
    recommended_items: list[object],
    relevant_items: set[object],
    k: int,
) -> float:
    """Calculate binary NDCG@K for one user."""
    if not relevant_items:
        return 0.0

    ranked_items = recommended_items[:k]
    seen_items: set[object] = set()
    relevance_values: list[float] = []
    for item_id in ranked_items:
        is_relevant = item_id in relevant_items and item_id not in seen_items
        relevance_values.append(float(is_relevant))
        seen_items.add(item_id)

    relevance = np.asarray(relevance_values, dtype=np.float64)
    discounts = np.log2(np.arange(2, len(ranked_items) + 2))
    dcg = float(np.sum(relevance / discounts))

    ideal_length = min(len(relevant_items), k)
    ideal_discounts = np.log2(np.arange(2, ideal_length + 2))
    idcg = float(np.sum(1.0 / ideal_discounts))
    return dcg / idcg


def ndcg_at_k_from_scores(
    scored_candidates: pd.DataFrame,
    targets: pd.DataFrame,
    score_col: str = "score",
    k: int = 10,
) -> float:
    """Calculate mean user NDCG@K from an arbitrary candidate score.

    The metric is averaged over every user present in ``targets``. A target
    user missing from ``scored_candidates`` receives zero. Relevant items that
    are absent from the candidate set remain in the ideal DCG denominator, so
    the result is an end-to-end ranking metric rather than a conditional metric
    calculated only for retrievable positives.
    """

    if scored_candidates.empty:
        raise ValueError("scored_candidates must not be empty")
    if targets.empty:
        raise ValueError("targets must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")

    candidate_columns = {"user_id", "item_id", score_col}
    missing_candidate_columns = candidate_columns.difference(
        scored_candidates.columns
    )
    if missing_candidate_columns:
        raise ValueError(
            "scored_candidates is missing columns: "
            f"{sorted(missing_candidate_columns)}"
        )

    target_columns = {"user_id", "item_id"}
    missing_target_columns = target_columns.difference(targets.columns)
    if missing_target_columns:
        raise ValueError(
            f"targets is missing columns: {sorted(missing_target_columns)}"
        )

    if scored_candidates[score_col].isna().any():
        raise ValueError(f"{score_col!r} must not contain missing values")
    if scored_candidates.duplicated(["user_id", "item_id"]).any():
        raise ValueError("scored_candidates contains duplicate user-item pairs")

    actual_by_user = (
        targets[["user_id", "item_id"]]
        .drop_duplicates()
        .groupby("user_id")["item_id"]
        .agg(set)
        .to_dict()
    )
    ranked_by_user = (
        scored_candidates.sort_values(
            ["user_id", score_col],
            ascending=[True, False],
            kind="mergesort",
        )
        .groupby("user_id", sort=False)["item_id"]
        .agg(list)
        .to_dict()
    )

    user_ndcg = [
        _binary_ndcg_at_k(
            ranked_by_user.get(user_id, []),
            relevant_items,
            k,
        )
        for user_id, relevant_items in actual_by_user.items()
    ]
    return float(np.mean(user_ndcg))

def oracle_ndcg_at_k(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    k: int = 10,
) -> float:
    """Максимально возможный NDCG@K на заданных кандидатах."""

    actual_by_user = (
        targets[["user_id", "item_id"]]
        .drop_duplicates()
        .groupby("user_id")["item_id"]
        .agg(set)
        .to_dict()
    )

    candidates_by_user = (
        candidates[["user_id", "item_id"]]
        .drop_duplicates()
        .groupby("user_id")["item_id"]
        .agg(set)
        .to_dict()
    )

    discounts = 1 / np.log2(
        np.arange(2, k + 2)
    )

    user_oracle_ndcg = []

    for user_id, relevant_items in actual_by_user.items():
        candidate_items = candidates_by_user.get(
            user_id,
            set(),
        )

        found_positives = len(
            relevant_items & candidate_items
        )

        ideal_found_count = min(
            found_positives,
            k,
        )

        ideal_total_count = min(
            len(relevant_items),
            k,
        )

        if ideal_total_count == 0:
            continue

        best_possible_dcg = discounts[
            :ideal_found_count
        ].sum()

        ideal_dcg = discounts[
            :ideal_total_count
        ].sum()

        user_oracle_ndcg.append(
            best_possible_dcg / ideal_dcg
        )

    return float(np.mean(user_oracle_ndcg))
