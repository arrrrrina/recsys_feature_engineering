from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from src.graph_data import GraphDataBundle


class HeterogeneousGraphSAGE(nn.Module):
    """Two-sided heterogeneous GraphSAGE encoder for recommendation."""

    def __init__(
        self,
        data: HeteroData,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_types = list(data.node_types)
        self.edge_types = list(data.edge_types)

        self.id_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(data[node_type].num_nodes, hidden_dim)
                for node_type in self.node_types
            }
        )
        self.feature_encoders = nn.ModuleDict(
            {
                node_type: nn.Linear(data[node_type].x.shape[1], hidden_dim)
                for node_type in self.node_types
                if "x" in data[node_type]
            }
        )

        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        for _ in range(num_layers):
            relation_convolutions = {
                edge_type: SAGEConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    normalize=False,
                )
                for edge_type in self.edge_types
            }
            self.convolutions.append(
                HeteroConv(relation_convolutions, aggr="sum")
            )
            self.normalizations.append(
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(hidden_dim)
                        for node_type in self.node_types
                    }
                )
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.id_embeddings.values():
            nn.init.normal_(embedding.weight, std=0.1)
        for encoder in self.feature_encoders.values():
            encoder.reset_parameters()
        for convolution in self.convolutions:
            convolution.reset_parameters()
        for normalizations in self.normalizations:
            for normalization in normalizations.values():
                normalization.reset_parameters()

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        representations: dict[str, torch.Tensor] = {}
        model_device = next(self.parameters()).device
        for node_type in self.node_types:
            node_count = data[node_type].num_nodes
            node_indices = torch.arange(node_count, device=model_device)
            representation = self.id_embeddings[node_type](node_indices)
            if node_type in self.feature_encoders:
                representation = representation + self.feature_encoders[node_type](
                    data[node_type].x
                )
            representations[node_type] = representation

        for layer_index, convolution in enumerate(self.convolutions):
            messages = convolution(representations, data.edge_index_dict)
            updated: dict[str, torch.Tensor] = {}
            for node_type in self.node_types:
                message = messages.get(node_type)
                if message is None:
                    message = representations[node_type]
                else:
                    message = message + representations[node_type]
                message = self.normalizations[layer_index][node_type](message)
                if layer_index < self.num_layers - 1:
                    message = F.relu(message)
                    message = F.dropout(
                        message,
                        p=self.dropout,
                        training=self.training,
                    )
                updated[node_type] = message
            representations = updated

        return representations

    @staticmethod
    def score_pairs(
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Score user-item pairs with a dot product."""

        return (
            user_embeddings[user_indices] * item_embeddings[item_indices]
        ).sum(dim=-1)


@dataclass
class GraphTrainingResult:
    model: HeterogeneousGraphSAGE
    losses: list[float]
    device: str


@dataclass
class GraphEmbeddings:
    """External IDs and learned user/item embedding matrices."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray

    def score_candidates(
        self,
        candidates: pd.DataFrame,
        *,
        score_col: str = "graph_score",
    ) -> pd.DataFrame:
        """Attach dot-product graph scores to candidate user-item pairs."""

        required = {"user_id", "item_id"}
        missing_columns = required.difference(candidates.columns)
        if missing_columns:
            raise ValueError(
                f"candidates is missing columns: {sorted(missing_columns)}"
            )
        if candidates.duplicated(["user_id", "item_id"]).any():
            raise ValueError("candidates contains duplicate user-item pairs")

        user_mapping = {
            user_id: index for index, user_id in enumerate(self.user_ids)
        }
        item_mapping = {
            item_id: index for index, item_id in enumerate(self.item_ids)
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

        user_matrix = self.user_embeddings[
            user_indices.to_numpy(dtype=np.int64)
        ]
        item_matrix = self.item_embeddings[
            item_indices.to_numpy(dtype=np.int64)
        ]

        result = candidates.copy()
        result[score_col] = np.sum(user_matrix * item_matrix, axis=1).astype(
            np.float32
        )
        return result


def train_graph_model(
    graph: GraphDataBundle,
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    negatives_per_positive: int = 5,
    embedding_regularization: float = 1e-6,
    random_state: int = 42,
    device: str | torch.device | None = None,
    verbose: int = 10,
) -> GraphTrainingResult:
    """Train GraphSAGE with full-graph message passing and BPR loss.

    Negative items are sampled uniformly for every positive user-item edge and
    are guaranteed not to be present among that user's observed interactions.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be positive")
    if embedding_regularization < 0:
        raise ValueError("embedding_regularization must be non-negative")
    if verbose < 0:
        raise ValueError("verbose must be non-negative")

    selected_device = _resolve_device(device)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    data = graph.data.to(selected_device)
    model = HeterogeneousGraphSAGE(
        data=data,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(selected_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    positive_edges = data[("user", "watched", "item")].edge_index
    num_items = data["item"].num_nodes
    generator = torch.Generator(device=selected_device)
    generator.manual_seed(random_state)
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        embeddings = model(data)
        sampled_users, positive_items, negative_items = sample_bpr_negatives(
            positive_edges=positive_edges,
            num_items=num_items,
            negatives_per_positive=negatives_per_positive,
            generator=generator,
        )

        positive_scores = model.score_pairs(
            embeddings["user"],
            embeddings["item"],
            sampled_users,
            positive_items,
        )
        negative_scores = model.score_pairs(
            embeddings["user"],
            embeddings["item"],
            sampled_users,
            negative_items,
        )

        ranking_loss = -F.logsigmoid(
            positive_scores - negative_scores
        ).mean()
        regularization = embedding_regularization * (
            embeddings["user"][sampled_users].pow(2).mean()
            + embeddings["item"][positive_items].pow(2).mean()
            + embeddings["item"][negative_items].pow(2).mean()
        )
        loss = ranking_loss + regularization
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if verbose and (epoch == 1 or epoch % verbose == 0 or epoch == epochs):
            print(f"Epoch {epoch:03d}/{epochs}: BPR loss = {loss_value:.6f}")

    return GraphTrainingResult(
        model=model,
        losses=losses,
        device=str(selected_device),
    )


def sample_bpr_negatives(
    positive_edges: torch.Tensor,
    num_items: int,
    negatives_per_positive: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample unseen negative items while preserving positive-edge users."""

    if positive_edges.ndim != 2 or positive_edges.shape[0] != 2:
        raise ValueError("positive_edges must have shape [2, num_edges]")
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be positive")

    device = positive_edges.device
    users = positive_edges[0].repeat_interleave(negatives_per_positive)
    positive_items = positive_edges[1].repeat_interleave(negatives_per_positive)
    negative_items = torch.randint(
        num_items,
        size=users.shape,
        generator=generator,
        device=device,
    )

    occupied_codes = (
        positive_edges[0] * num_items + positive_edges[1]
    ).sort().values
    collisions = _membership_mask(
        users * num_items + negative_items,
        occupied_codes,
    )
    attempts = 0
    while bool(collisions.any()):
        collision_count = int(collisions.sum())
        negative_items[collisions] = torch.randint(
            num_items,
            size=(collision_count,),
            generator=generator,
            device=device,
        )
        collisions = _membership_mask(
            users * num_items + negative_items,
            occupied_codes,
        )
        attempts += 1
        if attempts >= 100:
            raise RuntimeError(
                "Could not sample unseen items; one or more users may have "
                "interacted with nearly every item"
            )

    return users, positive_items, negative_items


@torch.no_grad()
def extract_graph_embeddings(
    training_result: GraphTrainingResult,
    graph: GraphDataBundle,
) -> GraphEmbeddings:
    """Run the trained encoder and return CPU user/item embeddings."""

    device = torch.device(training_result.device)
    model = training_result.model.to(device)
    data = graph.data.to(device)
    model.eval()
    embeddings = model(data)

    return GraphEmbeddings(
        user_ids=graph.node_ids["user"].copy(),
        item_ids=graph.node_ids["item"].copy(),
        user_embeddings=embeddings["user"].cpu().numpy(),
        item_embeddings=embeddings["item"].cpu().numpy(),
    )


def _membership_mask(values: torch.Tensor, sorted_reference: torch.Tensor) -> torch.Tensor:
    positions = torch.searchsorted(sorted_reference, values)
    valid_positions = positions < len(sorted_reference)
    safe_positions = positions.clamp(max=max(len(sorted_reference) - 1, 0))
    return valid_positions & (sorted_reference[safe_positions] == values)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
