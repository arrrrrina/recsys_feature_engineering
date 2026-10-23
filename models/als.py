from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


class ALSCandidateModel:
    """Generate item candidates using implicit-feedback ALS.

    The model treats every unique ``user-item`` pair as one positive signal.
    Interaction strength, watch duration and percentage are ignored.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 20.0,
        iterations: int = 20,
        random_state: int = 42,
        num_threads: int = 0,
    ) -> None:
        if factors <= 0:
            raise ValueError("factors must be positive")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state
        self.num_threads = num_threads

        self.model: AlternatingLeastSquares | None = None
        self.user_item_matrix: csr_matrix | None = None
        self.user_ids: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None
        self.user_to_index: dict[object, int] | None = None
        self.item_to_index: dict[object, int] | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        show_progress: bool = True,
    ) -> ALSCandidateModel:
        """Fit ALS on unique positive interactions.

        Parameters
        ----------
        interactions:
            History already restricted to the required temporal cutoff.
        user_col, item_col:
            Columns containing the original user and item identifiers.
        show_progress:
            Whether ``implicit`` should display its training progress bar.
        """
        self._validate_interactions(interactions, user_col, item_col)

        pairs = interactions[[user_col, item_col]].drop_duplicates()
        user_codes, user_ids = pd.factorize(pairs[user_col], sort=True)
        item_codes, item_ids = pd.factorize(pairs[item_col], sort=True)

        values = np.ones(len(pairs), dtype=np.float32)
        matrix = csr_matrix(
            (values, (user_codes, item_codes)),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        )
        matrix.sort_indices()

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=self.random_state,
            num_threads=self.num_threads,
        )
        model.fit(matrix, show_progress=show_progress)

        self.model = model
        self.user_item_matrix = matrix
        self.user_ids = user_ids.to_numpy(copy=True)
        self.item_ids = item_ids.to_numpy(copy=True)
        self.user_to_index = {
            user_id: index for index, user_id in enumerate(self.user_ids)
        }
        self.item_to_index = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }
        return self

    def recommend(
        self,
        user_ids: Iterable[object],
        n: int = 200,
        filter_seen: bool = True,
        batch_size: int = 4096,
    ) -> pd.DataFrame:
        """Return top-N ALS candidates for known users.

        Unknown users are omitted because ALS cannot construct an embedding for
        them. They should receive a separate fallback, such as popular items.

        Returns
        -------
        pandas.DataFrame
            Columns: ``user_id``, ``item_id``, ``als_score``, ``als_rank``.
        """
        self._check_is_fitted()
        if n <= 0:
            raise ValueError("n must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        requested_users = list(dict.fromkeys(user_ids))
        known_users = [
            user_id
            for user_id in requested_users
            if user_id in self.user_to_index
        ]
        if not known_users:
            return self._empty_recommendations()

        batches: list[pd.DataFrame] = []
        for start in range(0, len(known_users), batch_size):
            external_batch = known_users[start : start + batch_size]
            internal_batch = np.asarray(
                [self.user_to_index[user_id] for user_id in external_batch],
                dtype=np.int32,
            )

            item_indices, scores = self.model.recommend(
                internal_batch,
                self.user_item_matrix[internal_batch],
                N=n,
                filter_already_liked_items=filter_seen,
            )

            if item_indices.ndim == 1:
                item_indices = item_indices.reshape(1, -1)
                scores = scores.reshape(1, -1)

            for row, user_id in enumerate(external_batch):
                valid = item_indices[row] >= 0
                selected_indices = item_indices[row][valid]
                selected_scores = scores[row][valid]
                candidate_count = len(selected_indices)
                if candidate_count == 0:
                    continue

                batches.append(
                    pd.DataFrame(
                        {
                            "user_id": np.repeat(user_id, candidate_count),
                            "item_id": self.item_ids[selected_indices],
                            "als_score": selected_scores.astype(np.float32),
                            "als_rank": np.arange(
                                1, candidate_count + 1, dtype=np.int32
                            ),
                        }
                    )
                )

        if not batches:
            return self._empty_recommendations()
        return pd.concat(batches, ignore_index=True)

    @property
    def n_users(self) -> int:
        """Number of users known to the fitted model."""
        self._check_is_fitted()
        return self.user_item_matrix.shape[0]

    @property
    def n_items(self) -> int:
        """Number of items known to the fitted model."""
        self._check_is_fitted()
        return self.user_item_matrix.shape[1]

    @staticmethod
    def _validate_interactions(
        interactions: pd.DataFrame,
        user_col: str,
        item_col: str,
    ) -> None:
        missing_columns = {
            column
            for column in (user_col, item_col)
            if column not in interactions.columns
        }
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns: {missing}")
        if interactions.empty:
            raise ValueError("interactions must not be empty")
        if interactions[[user_col, item_col]].isna().any().any():
            raise ValueError("user and item identifiers must not contain nulls")

    def _check_is_fitted(self) -> None:
        if (
            self.model is None
            or self.user_item_matrix is None
            or self.user_ids is None
            or self.item_ids is None
            or self.user_to_index is None
            or self.item_to_index is None
        ):
            raise RuntimeError("Fit the model before requesting recommendations")

    @staticmethod
    def _empty_recommendations() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user_id": pd.Series(dtype="object"),
                "item_id": pd.Series(dtype="object"),
                "als_score": pd.Series(dtype="float32"),
                "als_rank": pd.Series(dtype="int32"),
            }
        )
