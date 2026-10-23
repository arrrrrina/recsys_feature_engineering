from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import numpy as np
import pandas as pd


def build_manual_features(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    cutoff: str | date | datetime | pd.Timestamp,
    windows: Sequence[int] = (7, 30, 90),
) -> tuple[pd.DataFrame, list[str]]:
    """Build leakage-safe manual features for candidate user-item pairs.

    Parameters
    ----------
    candidates:
        Candidate pairs. Must contain ``user_id`` and ``item_id``. Extra
        columns such as ``als_score`` and ``als_rank`` are preserved.
    history:
        Interactions available strictly before ``cutoff``.
    users, items:
        User and item metadata tables from the KION dataset.
    cutoff:
        Prediction timestamp. Recency and window features are calculated
        relative to this value.
    windows:
        Lookback windows in days for activity and popularity counts.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        Feature table and names of categorical columns for CatBoost.

    Notes
    -----
    The function does not create a target. Labels should be attached to the
    candidate pairs separately for each temporal split.
    """
    _validate_inputs(candidates, history, users, items, windows)

    cutoff_ts = pd.Timestamp(cutoff)
    if pd.isna(cutoff_ts):
        raise ValueError("cutoff must be a valid date")

    history_work = history.copy()
    history_work["_event_dt"] = pd.to_datetime(
        history_work["last_watch_dt"], errors="coerce"
    )
    invalid_dates = int(history_work["_event_dt"].isna().sum())
    if invalid_dates:
        raise ValueError(
            f"history contains {invalid_dates} missing or invalid dates"
        )

    future_rows = int((history_work["_event_dt"] >= cutoff_ts).sum())
    if future_rows:
        raise ValueError(
            f"history contains {future_rows} rows on or after cutoff; "
            "this would leak future information"
        )

    features = candidates.copy().reset_index(drop=True)
    item_metadata = _prepare_item_metadata(items, cutoff_ts)

    user_stats = _build_entity_stats(
        history_work,
        entity_col="user_id",
        other_col="item_id",
        prefix="user",
        cutoff=cutoff_ts,
        windows=windows,
    )
    item_stats = _build_entity_stats(
        history_work,
        entity_col="item_id",
        other_col="user_id",
        prefix="item",
        cutoff=cutoff_ts,
        windows=windows,
    )

    user_preferences = _build_user_preferences(history_work, item_metadata)

    user_columns = [
        column
        for column in ("user_id", "age", "income", "sex", "kids_flg")
        if column in users.columns
    ]
    user_metadata = users[user_columns].drop_duplicates("user_id")

    features = features.merge(
        user_metadata, on="user_id", how="left", validate="many_to_one"
    )
    features = features.merge(
        user_stats, on="user_id", how="left", validate="many_to_one"
    )
    features = features.merge(
        user_preferences, on="user_id", how="left", validate="many_to_one"
    )
    features = features.merge(
        item_metadata, on="item_id", how="left", validate="many_to_one"
    )
    features = features.merge(
        item_stats, on="item_id", how="left", validate="many_to_one"
    )

    features["genre_match"] = (
        features["user_top_genre"]
        .eq(features["item_primary_genre"])
        .fillna(False)
        .astype("int8")
    )
    features["country_match"] = (
        features["user_top_country"]
        .eq(features["item_primary_country"])
        .fillna(False)
        .astype("int8")
    )
    features["content_type_match"] = (
        features["user_top_content_type"]
        .eq(features["content_type"])
        .fillna(False)
        .astype("int8")
    )

    count_columns = [
        column
        for column in features.columns
        if column.endswith("_count")
        or column.endswith("_unique_items")
        or column.endswith("_unique_users")
        or column.endswith("_active_days")
        or "interactions_" in column
    ]
    features[count_columns] = features[count_columns].fillna(0)

    categorical_columns = [
        column
        for column in (
            "user_id",
            "item_id",
            "age",
            "income",
            "sex",
            "kids_flg",
            "content_type",
            "for_kids",
            "item_primary_genre",
            "item_primary_country",
            "user_top_genre",
            "user_top_country",
            "user_top_content_type",
        )
        if column in features.columns
    ]

    for column in categorical_columns:
        features[column] = (
            features[column].astype("string").fillna("__missing__")
        )

    numeric_columns = features.select_dtypes(include=["float64"]).columns
    features[numeric_columns] = features[numeric_columns].astype("float32")

    return features, categorical_columns


def _build_entity_stats(
    history: pd.DataFrame,
    entity_col: str,
    other_col: str,
    prefix: str,
    cutoff: pd.Timestamp,
    windows: Sequence[int],
) -> pd.DataFrame:
    grouped = history.groupby(entity_col, sort=False)
    stats = grouped.agg(
        **{
            f"{prefix}_interaction_count": (other_col, "size"),
            f"{prefix}_unique_{'items' if prefix == 'user' else 'users'}": (
                other_col,
                "nunique",
            ),
            f"{prefix}_active_days": ("_event_dt", "nunique"),
            f"{prefix}_last_event_dt": ("_event_dt", "max"),
            f"{prefix}_avg_watched_pct": ("watched_pct", "mean"),
            f"{prefix}_max_watched_pct": ("watched_pct", "max"),
            f"{prefix}_avg_total_dur": ("total_dur", "mean"),
            f"{prefix}_sum_total_dur": ("total_dur", "sum"),
        }
    ).reset_index()

    last_event_col = f"{prefix}_last_event_dt"
    stats[f"{prefix}_days_since_last_event"] = (
        cutoff - stats[last_event_col]
    ).dt.days
    stats = stats.drop(columns=last_event_col)

    for window in windows:
        window_start = cutoff - pd.Timedelta(days=int(window))
        recent = history.loc[history["_event_dt"] >= window_start]
        recent_counts = (
            recent.groupby(entity_col, sort=False)
            .size()
            .rename(f"{prefix}_interactions_{window}d")
            .reset_index()
        )
        stats = stats.merge(
            recent_counts, on=entity_col, how="left", validate="one_to_one"
        )

    return stats


def _prepare_item_metadata(
    items: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in (
            "item_id",
            "content_type",
            "release_year",
            "genres",
            "countries",
            "for_kids",
            "age_rating",
        )
        if column in items.columns
    ]
    metadata = items[metadata_columns].drop_duplicates("item_id").copy()

    genres = metadata.get(
        "genres", pd.Series("", index=metadata.index, dtype="string")
    ).astype("string")
    countries = metadata.get(
        "countries", pd.Series("", index=metadata.index, dtype="string")
    ).astype("string")

    metadata["item_primary_genre"] = (
        genres.fillna("").str.split(",").str[0].str.strip().replace("", pd.NA)
    )
    metadata["item_primary_country"] = (
        countries.fillna("")
        .str.split(",")
        .str[0]
        .str.strip()
        .replace("", pd.NA)
    )
    metadata["item_genre_count"] = (
        genres.fillna("").str.count(",").add(1).where(genres.fillna("") != "", 0)
    )
    metadata["item_country_count"] = (
        countries.fillna("")
        .str.count(",")
        .add(1)
        .where(countries.fillna("") != "", 0)
    )

    if "release_year" in metadata.columns:
        metadata["item_age"] = cutoff.year - metadata["release_year"]

    return metadata.drop(columns=["genres", "countries"], errors="ignore")


def _build_user_preferences(
    history: pd.DataFrame,
    item_metadata: pd.DataFrame,
) -> pd.DataFrame:
    category_columns = [
        "item_primary_genre",
        "item_primary_country",
        "content_type",
    ]
    interaction_metadata = history[["user_id", "item_id"]].merge(
        item_metadata[["item_id", *category_columns]],
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    preferences = history[["user_id"]].drop_duplicates().copy()
    output_names = {
        "item_primary_genre": "user_top_genre",
        "item_primary_country": "user_top_country",
        "content_type": "user_top_content_type",
    }

    for category_col, output_col in output_names.items():
        counts = (
            interaction_metadata.dropna(subset=[category_col])
            .groupby(["user_id", category_col], sort=False)
            .size()
            .rename("_category_count")
            .reset_index()
        )
        top_category = (
            counts.sort_values(
                ["user_id", "_category_count", category_col],
                ascending=[True, False, True],
            )
            .drop_duplicates("user_id")
            [["user_id", category_col]]
            .rename(columns={category_col: output_col})
        )
        preferences = preferences.merge(
            top_category, on="user_id", how="left", validate="one_to_one"
        )

    diversity = (
        interaction_metadata.groupby("user_id", sort=False)
        .agg(
            user_genre_diversity=("item_primary_genre", "nunique"),
            user_country_diversity=("item_primary_country", "nunique"),
            user_content_type_diversity=("content_type", "nunique"),
        )
        .reset_index()
    )
    return preferences.merge(
        diversity, on="user_id", how="left", validate="one_to_one"
    )


def _validate_inputs(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    windows: Sequence[int],
) -> None:
    required = {
        "candidates": (candidates, {"user_id", "item_id"}),
        "history": (
            history,
            {
                "user_id",
                "item_id",
                "last_watch_dt",
                "total_dur",
                "watched_pct",
            },
        ),
        "users": (users, {"user_id"}),
        "items": (items, {"item_id"}),
    }
    for name, (frame, columns) in required.items():
        missing = columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} must not be empty")

    if candidates.duplicated(["user_id", "item_id"]).any():
        raise ValueError("candidates contains duplicate user-item pairs")
    if users.duplicated("user_id").any():
        raise ValueError("users contains duplicate user_id values")
    if items.duplicated("item_id").any():
        raise ValueError("items contains duplicate item_id values")
    if not windows or any(int(window) <= 0 for window in windows):
        raise ValueError("windows must contain positive day counts")
