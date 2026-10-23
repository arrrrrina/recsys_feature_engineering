from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from models.graph_model import GraphEmbeddings


def build_graph_pair_features(
    candidates: pd.DataFrame,
    embeddings: GraphEmbeddings,
) -> tuple[pd.DataFrame, list[str]]:
    """Build rotation-invariant graph features for user-item candidates.

    Raw embedding coordinates are intentionally not returned. Independently
    trained graph models can represent the same geometry in rotated embedding
    spaces, while dot product, cosine similarity, Euclidean distance and norms
    remain comparable between temporal snapshots.

    Candidate-generation fields ``als_score`` and ``als_rank`` are preserved
    when present. User and item identifiers are returned as CatBoost
    categorical features.
    """

    required_columns = {"user_id", "item_id"}
    missing_columns = required_columns.difference(candidates.columns)
    if missing_columns:
        raise ValueError(
            f"candidates is missing columns: {sorted(missing_columns)}"
        )
    if candidates.empty:
        raise ValueError("candidates must not be empty")
    if candidates.duplicated(["user_id", "item_id"]).any():
        raise ValueError("candidates contains duplicate user-item pairs")

    user_mapping = {
        user_id: index for index, user_id in enumerate(embeddings.user_ids)
    }
    item_mapping = {
        item_id: index for index, item_id in enumerate(embeddings.item_ids)
    }

    user_indices = candidates["user_id"].map(user_mapping)
    item_indices = candidates["item_id"].map(item_mapping)
    unknown_mask = user_indices.isna() | item_indices.isna()
    if unknown_mask.any():
        examples = candidates.loc[
            unknown_mask, ["user_id", "item_id"]
        ].head(5)
        raise KeyError(
            f"{int(unknown_mask.sum())} candidate pairs contain nodes "
            f"unknown to the graph; examples:\n{examples.to_string(index=False)}"
        )

    user_vectors = embeddings.user_embeddings[
        user_indices.to_numpy(dtype=np.int64)
    ].astype(np.float32, copy=False)
    item_vectors = embeddings.item_embeddings[
        item_indices.to_numpy(dtype=np.int64)
    ].astype(np.float32, copy=False)

    user_norm = np.linalg.norm(user_vectors, axis=1)
    item_norm = np.linalg.norm(item_vectors, axis=1)
    dot_product = np.sum(user_vectors * item_vectors, axis=1)
    norm_product = user_norm * item_norm
    cosine_similarity = np.divide(
        dot_product,
        norm_product,
        out=np.zeros_like(dot_product, dtype=np.float32),
        where=norm_product > 0,
    )
    l2_distance = np.linalg.norm(user_vectors - item_vectors, axis=1)

    output_columns = ["user_id", "item_id"]
    output_columns.extend(
        column
        for column in ("als_score", "als_rank")
        if column in candidates.columns
    )
    features = candidates[output_columns].copy().reset_index(drop=True)
    features["graph_dot_product"] = dot_product.astype(np.float32)
    features["graph_cosine_similarity"] = cosine_similarity.astype(np.float32)
    features["graph_l2_distance"] = l2_distance.astype(np.float32)
    features["graph_user_norm"] = user_norm.astype(np.float32)
    features["graph_item_norm"] = item_norm.astype(np.float32)

    categorical_columns = ["user_id", "item_id"]
    for column in categorical_columns:
        features[column] = features[column].astype("string")

    return features, categorical_columns
