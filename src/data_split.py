from __future__ import annotations

import warnings
from datetime import date, datetime

import pandas as pd


def temporal_train_validation_test_split(
    interactions: pd.DataFrame,
    train_cutoff: str | date | datetime | pd.Timestamp,
    validation_cutoff: str | date | datetime | pd.Timestamp,
    date_col: str = "last_watch_dt",
    drop_invalid_dates: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions into train, validation and test by time.

    The intervals are defined as follows:

    - train: dates strictly before ``train_cutoff``;
    - validation: dates from ``train_cutoff`` (inclusive) to
      ``validation_cutoff`` (exclusive);
    - test: dates from ``validation_cutoff`` (inclusive).

    Parameters
    ----------
    interactions:
        Source table. It is not modified by the function.
    train_cutoff:
        First date included in validation.
    validation_cutoff:
        First date included in test.
    date_col:
        Name of the interaction timestamp column.
    drop_invalid_dates:
        If ``False``, raise an error when dates are missing or invalid. If
        ``True``, explicitly drop those rows and emit a warning.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Train, validation and test tables with reset indices.

    Raises
    ------
    ValueError
        If the date column is missing, contains invalid values, or the cutoffs
        are not ordered chronologically.
    """
    if date_col not in interactions.columns:
        raise ValueError(f"Missing date column: {date_col}")

    train_boundary = pd.Timestamp(train_cutoff)
    validation_boundary = pd.Timestamp(validation_cutoff)
    if pd.isna(train_boundary) or pd.isna(validation_boundary):
        raise ValueError("Cutoffs must be valid dates")
    if train_boundary >= validation_boundary:
        raise ValueError("train_cutoff must be earlier than validation_cutoff")

    timestamps = pd.to_datetime(interactions[date_col], errors="coerce")
    invalid_dates = int(timestamps.isna().sum())
    if invalid_dates and not drop_invalid_dates:
        raise ValueError(
            f"Column {date_col!r} contains {invalid_dates} missing or invalid dates"
        )
    if invalid_dates:
        warnings.warn(
            f"Dropped {invalid_dates} rows with missing or invalid {date_col!r}",
            UserWarning,
            stacklevel=2,
        )

    train_mask = timestamps < train_boundary
    validation_mask = (timestamps >= train_boundary) & (
        timestamps < validation_boundary
    )
    test_mask = timestamps >= validation_boundary

    train = interactions.loc[train_mask].copy().reset_index(drop=True)
    validation = interactions.loc[validation_mask].copy().reset_index(drop=True)
    test = interactions.loc[test_mask].copy().reset_index(drop=True)

    return train, validation, test
