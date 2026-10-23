from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


NodeId = Any


@dataclass
class GraphDataBundle:
    """A heterogeneous graph together with external-ID mappings.

    ``data`` contains integer node indices used by PyTorch Geometric, while
    ``node_ids`` and ``id_to_index`` make it possible to convert model output
    back to the original identifiers from the source tables.
    """

    data: HeteroData
    node_ids: dict[str, np.ndarray]
    id_to_index: dict[str, dict[NodeId, int]]
    feature_names: dict[str, list[str]]
    cutoff: pd.Timestamp

    def indices_for(self, node_type: str, external_ids: list[NodeId]) -> np.ndarray:
        """Convert external node identifiers to graph indices.

        Raises an explicit error for unknown nodes instead of silently dropping
        them, which helps catch mismatched temporal graph snapshots.
        """

        if node_type not in self.id_to_index:
            raise KeyError(f"Unknown node type: {node_type!r}")

        mapping = self.id_to_index[node_type]
        missing = [node_id for node_id in external_ids if node_id not in mapping]
        if missing:
            preview = missing[:5]
            raise KeyError(
                f"{len(missing)} unknown {node_type} ids; examples: {preview}"
            )
        return np.asarray([mapping[node_id] for node_id in external_ids])


def build_heterogeneous_graph(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    cutoff: str | date | datetime | pd.Timestamp,
    *,
    date_col: str = "last_watch_dt",
    drop_invalid_dates: bool = False,
) -> GraphDataBundle:
    """Build a leakage-safe user-item-metadata graph before ``cutoff``.

    The graph contains these node and directed edge types:

    - ``user --watched--> item`` and its reverse;
    - ``item --has_genre--> genre`` and its reverse;
    - ``item --made_in--> country`` and its reverse;
    - ``item --has_type--> content_type`` and its reverse.

    Every unique user-item pair is treated as one binary interaction. Rows on
    or after ``cutoff`` are excluded before any graph statistic is calculated.
    """

    _validate_source_tables(interactions, users, items, date_col)

    cutoff_ts = pd.Timestamp(cutoff)
    if pd.isna(cutoff_ts):
        raise ValueError("cutoff must be a valid date")

    event_dates = pd.to_datetime(interactions[date_col], errors="coerce")
    invalid_count = int(event_dates.isna().sum())
    if invalid_count and not drop_invalid_dates:
        raise ValueError(
            f"{date_col!r} contains {invalid_count} missing or invalid dates"
        )

    history = (
        interactions.loc[event_dates.notna() & (event_dates < cutoff_ts)]
        .copy()
        .reset_index(drop=True)
    )
    if history.empty:
        raise ValueError("No interactions exist before cutoff")

    interaction_pairs = (
        history[["user_id", "item_id"]]
        .drop_duplicates()
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )

    unknown_items = set(interaction_pairs["item_id"]).difference(items["item_id"])
    if unknown_items:
        raise ValueError(
            f"items metadata is missing {len(unknown_items)} interacted items"
        )

    user_ids = _sorted_unique(interaction_pairs["user_id"])
    item_ids = _sorted_unique(interaction_pairs["item_id"])
    item_metadata = (
        items.set_index("item_id", drop=False)
        .loc[item_ids]
        .reset_index(drop=True)
    )

    genre_edges, genre_ids = _category_edges(item_metadata, "genres", item_ids)
    country_edges, country_ids = _category_edges(
        item_metadata, "countries", item_ids
    )
    content_type_edges, content_type_ids = _single_category_edges(
        item_metadata, "content_type", item_ids
    )

    node_ids = {
        "user": user_ids,
        "item": item_ids,
        "genre": genre_ids,
        "country": country_ids,
        "content_type": content_type_ids,
    }
    id_to_index = {
        node_type: {node_id: index for index, node_id in enumerate(ids)}
        for node_type, ids in node_ids.items()
    }

    watched_edges = _interaction_edge_index(
        interaction_pairs,
        id_to_index["user"],
        id_to_index["item"],
    )

    user_x, user_feature_names = _build_user_features(
        users=users,
        user_ids=user_ids,
        interaction_pairs=interaction_pairs,
    )
    item_x, item_feature_names = _build_item_features(
        item_metadata=item_metadata,
        interaction_pairs=interaction_pairs,
        cutoff=cutoff_ts,
    )

    data = HeteroData()
    data["user"].num_nodes = len(user_ids)
    data["user"].x = user_x
    data["item"].num_nodes = len(item_ids)
    data["item"].x = item_x

    for node_type in ("genre", "country", "content_type"):
        data[node_type].num_nodes = len(node_ids[node_type])

    _add_bidirectional_edges(
        data,
        ("user", "watched", "item"),
        watched_edges,
        ("item", "watched_by", "user"),
    )
    _add_bidirectional_edges(
        data,
        ("item", "has_genre", "genre"),
        genre_edges,
        ("genre", "genre_of", "item"),
    )
    _add_bidirectional_edges(
        data,
        ("item", "made_in", "country"),
        country_edges,
        ("country", "country_of", "item"),
    )
    _add_bidirectional_edges(
        data,
        ("item", "has_type", "content_type"),
        content_type_edges,
        ("content_type", "type_of", "item"),
    )

    data.validate(raise_on_error=True)
    return GraphDataBundle(
        data=data,
        node_ids=node_ids,
        id_to_index=id_to_index,
        feature_names={
            "user": user_feature_names,
            "item": item_feature_names,
        },
        cutoff=cutoff_ts,
    )


def _build_user_features(
    users: pd.DataFrame,
    user_ids: np.ndarray,
    interaction_pairs: pd.DataFrame,
) -> tuple[torch.Tensor, list[str]]:
    metadata = users.set_index("user_id").reindex(user_ids)
    feature_arrays: list[np.ndarray] = []
    feature_names: list[str] = []

    for column in ("age", "income", "sex"):
        if column not in users.columns:
            continue
        vocabulary = sorted(users[column].dropna().astype(str).unique().tolist())
        values = metadata[column].astype("string").fillna("__missing__")
        vocabulary = [*vocabulary, "__missing__"]
        encoded = np.column_stack(
            [(values == category).to_numpy(dtype=np.float32) for category in vocabulary]
        )
        feature_arrays.append(encoded)
        feature_names.extend([f"{column}={category}" for category in vocabulary])

    if "kids_flg" in metadata.columns:
        kids = pd.to_numeric(metadata["kids_flg"], errors="coerce")
        feature_arrays.extend(
            [
                kids.fillna(0).to_numpy(dtype=np.float32)[:, None],
                kids.isna().to_numpy(dtype=np.float32)[:, None],
            ]
        )
        feature_names.extend(["kids_flg", "kids_flg_missing"])

    degree = (
        interaction_pairs.groupby("user_id")["item_id"]
        .nunique()
        .reindex(user_ids, fill_value=0)
    )
    feature_arrays.append(np.log1p(degree.to_numpy(dtype=np.float32))[:, None])
    feature_names.append("log_user_degree")

    matrix = np.concatenate(feature_arrays, axis=1)
    return torch.from_numpy(matrix), feature_names


def _build_item_features(
    item_metadata: pd.DataFrame,
    interaction_pairs: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[torch.Tensor, list[str]]:
    size = len(item_metadata)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    if "release_year" in item_metadata.columns:
        release_year = pd.to_numeric(item_metadata["release_year"], errors="coerce")
        item_age = (cutoff.year - release_year).clip(lower=0, upper=100) / 100
        arrays.extend(
            [
                item_age.fillna(0).to_numpy(dtype=np.float32)[:, None],
                release_year.isna().to_numpy(dtype=np.float32)[:, None],
            ]
        )
        names.extend(["item_age_scaled", "release_year_missing"])

    if "age_rating" in item_metadata.columns:
        age_rating = pd.to_numeric(item_metadata["age_rating"], errors="coerce")
        arrays.extend(
            [
                (age_rating.fillna(0) / 18).to_numpy(dtype=np.float32)[:, None],
                age_rating.isna().to_numpy(dtype=np.float32)[:, None],
            ]
        )
        names.extend(["age_rating_scaled", "age_rating_missing"])

    if "for_kids" in item_metadata.columns:
        for_kids = pd.to_numeric(item_metadata["for_kids"], errors="coerce")
        arrays.extend(
            [
                for_kids.fillna(0).to_numpy(dtype=np.float32)[:, None],
                for_kids.isna().to_numpy(dtype=np.float32)[:, None],
            ]
        )
        names.extend(["for_kids", "for_kids_missing"])

    item_ids = item_metadata["item_id"].to_numpy()
    degree = (
        interaction_pairs.groupby("item_id")["user_id"]
        .nunique()
        .reindex(item_ids, fill_value=0)
    )
    arrays.append(np.log1p(degree.to_numpy(dtype=np.float32))[:, None])
    names.append("log_item_degree")

    if not arrays:
        arrays.append(np.zeros((size, 1), dtype=np.float32))
        names.append("constant")

    matrix = np.concatenate(arrays, axis=1)
    return torch.from_numpy(matrix), names


def _interaction_edge_index(
    pairs: pd.DataFrame,
    user_mapping: dict[NodeId, int],
    item_mapping: dict[NodeId, int],
) -> torch.Tensor:
    users = pairs["user_id"].map(user_mapping).to_numpy(dtype=np.int64)
    items = pairs["item_id"].map(item_mapping).to_numpy(dtype=np.int64)
    return torch.from_numpy(np.vstack([users, items]))


def _category_edges(
    item_metadata: pd.DataFrame,
    column: str,
    item_ids: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    if column not in item_metadata.columns:
        return _empty_edges(), np.asarray([], dtype=object)

    exploded = pd.DataFrame(
        {
            "item_id": item_ids,
            "category": item_metadata[column]
            .fillna("")
            .astype(str)
            .str.split(","),
        }
    ).explode("category")
    exploded["category"] = exploded["category"].str.strip()
    exploded = exploded[exploded["category"] != ""].drop_duplicates()
    return _encode_category_edges(exploded, item_ids)


def _single_category_edges(
    item_metadata: pd.DataFrame,
    column: str,
    item_ids: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    if column not in item_metadata.columns:
        return _empty_edges(), np.asarray([], dtype=object)

    frame = pd.DataFrame(
        {
            "item_id": item_ids,
            "category": item_metadata[column].astype("string").fillna("__missing__"),
        }
    ).drop_duplicates()
    return _encode_category_edges(frame, item_ids)


def _encode_category_edges(
    frame: pd.DataFrame,
    item_ids: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    if frame.empty:
        return _empty_edges(), np.asarray([], dtype=object)

    category_ids = _sorted_unique(frame["category"])
    item_mapping = {item_id: index for index, item_id in enumerate(item_ids)}
    category_mapping = {
        category: index for index, category in enumerate(category_ids)
    }
    source = frame["item_id"].map(item_mapping).to_numpy(dtype=np.int64)
    target = frame["category"].map(category_mapping).to_numpy(dtype=np.int64)
    return torch.from_numpy(np.vstack([source, target])), category_ids


def _add_bidirectional_edges(
    data: HeteroData,
    forward_type: tuple[str, str, str],
    edge_index: torch.Tensor,
    reverse_type: tuple[str, str, str],
) -> None:
    data[forward_type].edge_index = edge_index
    data[reverse_type].edge_index = edge_index.flip(0)


def _empty_edges() -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long)


def _sorted_unique(values: pd.Series) -> np.ndarray:
    unique = values.dropna().unique()
    try:
        return np.asarray(sorted(unique.tolist()))
    except TypeError:
        return np.asarray(sorted(unique.tolist(), key=str), dtype=object)


def _validate_source_tables(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    date_col: str,
) -> None:
    requirements = {
        "interactions": (interactions, {"user_id", "item_id", date_col}),
        "users": (users, {"user_id"}),
        "items": (items, {"item_id"}),
    }
    for name, (frame, required_columns) in requirements.items():
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} must not be empty")

    if interactions[["user_id", "item_id"]].isna().any().any():
        raise ValueError("interaction user_id and item_id must not contain nulls")
    if users.duplicated("user_id").any():
        raise ValueError("users contains duplicate user_id values")
    if items.duplicated("item_id").any():
        raise ValueError("items contains duplicate item_id values")
