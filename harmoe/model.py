"""Dataset-routed residual expert head used by HarMoE."""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .decoder import TransformerDecoder, TransformerDecoderLayer


@dataclass
class HarMoEOutput:
    logits: Tensor
    shared: Tensor
    dataset_residual: Tensor

    @property
    def probabilities(self) -> Tensor:
        return self.logits.softmax(dim=-1)[..., 1]


class LowRankDelta(nn.Module):
    """A zero-initialized low-rank residual, x @ A @ B."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, dim))
        nn.init.normal_(self.A, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x.matmul(self.A).matmul(self.B)

    def squared_l2(self) -> Tensor:
        return self.A.square().sum() + self.B.square().sum()


class HarMoEHead(nn.Module):
    """Cross-modal decoder with deterministic dataset-routed residual experts.

    Args:
        dim: Image/text token dimension.
        num_layers: Number of cascaded decoder layers.
        rank: Rank of each dataset-specific residual.
        expert_start_layer: Zero-based first layer receiving an expert residual.
        dataset_names: Source identifiers accepted by the deterministic router.
        expert_mode: ``kv`` is the released reference configuration. ``qkvo``
            and ``shared_qkvo`` expose the projection-wise ablation variants.

    During pretraining, pass a source name and ``use_dataset_expert=True``. For
    source-agnostic zero-shot inference, set ``use_dataset_expert=False``; no
    dataset identifier is then required.
    """

    MODE_ALIASES = {
        "qkvo": "qkvo",
        "qkvo_coarse": "qkvo",
        "kv": "kv",
        "kv_only": "kv",
        "shared_qkvo": "shared_qkvo",
        "qkvo_lowrank": "shared_qkvo",
    }

    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 4,
        rank: int = 8,
        expert_start_layer: int = 2,
        dataset_names: Sequence[str] = ("mimic", "chexpert", "chestxray14", "padchest"),
        expert_mode: str = "kv",
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if expert_mode not in self.MODE_ALIASES:
            raise ValueError(f"Unknown expert_mode={expert_mode!r}")
        if not 0 <= expert_start_layer < num_layers:
            raise ValueError("expert_start_layer must index one of the decoder layers")
        if len(set(dataset_names)) != len(dataset_names):
            raise ValueError("dataset_names must be unique")

        self.d_model = dim
        self.expert_start_layer = expert_start_layer
        self.expert_mode = self.MODE_ALIASES[expert_mode]
        self.dataset_names = tuple(dataset_names)

        decoder_layer = TransformerDecoderLayer(dim, num_heads, dim, dropout)
        self.decoder_norm = nn.LayerNorm(dim)
        self.decoder = TransformerDecoder(decoder_layer, num_layers, self.decoder_norm)
        self.dropout_feas = nn.Dropout(dropout)
        self.mlp_head = nn.Linear(dim, 2)

        def expert_bank() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    nn.ModuleDict({name: LowRankDelta(dim, rank) for name in self.dataset_names})
                    for _ in range(num_layers)
                ]
            )

        # The reference checkpoint predates the projection-wise ablations and
        # contains only kv_delta. Later ablation checkpoints contain all banks.
        self.kv_delta = expert_bank()
        if self.expert_mode != "kv":
            self.q_delta = expert_bank()
            self.k_delta = expert_bank()
            self.v_delta = expert_bank()
            self.o_delta = expert_bank()
            self.lr_delta = expert_bank()

    def _validate_inputs(
        self,
        image_features: Tensor,
        text_features: Tensor,
        dataset_name: Optional[str],
        use_dataset_expert: bool,
    ) -> None:
        if image_features.ndim != 3:
            raise ValueError("image_features must have shape [batch, patches, dim]")
        if text_features.ndim != 2:
            raise ValueError("text_features must have shape [classes, dim]")
        if image_features.shape[-1] != self.d_model or text_features.shape[-1] != self.d_model:
            raise ValueError(f"feature dimension must be {self.d_model}")
        if use_dataset_expert and dataset_name not in self.dataset_names:
            raise ValueError(f"dataset_name must be one of {self.dataset_names}")

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        dataset_name: Optional[str] = None,
        alpha: float = 1.0,
        use_dataset_expert: bool = True,
    ) -> HarMoEOutput:
        self._validate_inputs(image_features, text_features, dataset_name, use_dataset_expert)
        batch_size = image_features.shape[0]
        memory = self.decoder_norm(image_features.transpose(0, 1))
        queries = self.decoder_norm(text_features.unsqueeze(1).expand(-1, batch_size, -1))

        shared = queries
        routed = queries
        routed_memory = memory
        for layer_idx, layer in enumerate(self.decoder.layers):
            shared, _ = layer(shared, memory)
            expert_active = use_dataset_expert and layer_idx >= self.expert_start_layer
            if not expert_active:
                routed, _ = layer(routed, routed_memory)
                continue

            assert dataset_name is not None
            if self.expert_mode == "qkvo":
                query = routed + alpha * self.q_delta[layer_idx][dataset_name](routed)
                key = routed_memory + alpha * self.k_delta[layer_idx][dataset_name](routed_memory)
                value = routed_memory + alpha * self.v_delta[layer_idx][dataset_name](routed_memory)
                routed, _ = layer(query, key, memory_value=value)
                routed = routed + alpha * self.o_delta[layer_idx][dataset_name](routed)
            elif self.expert_mode == "shared_qkvo":
                delta = self.lr_delta[layer_idx][dataset_name]
                query = routed + alpha * delta(routed)
                key = routed_memory + alpha * delta(routed_memory)
                value = routed_memory + alpha * delta(routed_memory)
                routed, _ = layer(query, key, memory_value=value)
                routed = routed + alpha * delta(routed)
            else:
                routed_memory = routed_memory + alpha * self.kv_delta[layer_idx][dataset_name](
                    routed_memory
                )
                routed, _ = layer(routed, routed_memory)

        shared = self.decoder.norm(shared)
        routed = self.decoder.norm(routed)
        shared_features = self.dropout_feas(shared).transpose(0, 1)
        routed_features = self.dropout_feas(routed).transpose(0, 1)
        logits = self.mlp_head(routed_features)
        shared_pooled = shared_features.mean(dim=1)
        residual_pooled = routed_features.mean(dim=1) - shared_pooled
        return HarMoEOutput(logits, shared_pooled, residual_pooled)

    def residual_l2(self, dataset_name: str) -> Tensor:
        if dataset_name not in self.dataset_names:
            raise ValueError(f"dataset_name must be one of {self.dataset_names}")
        terms = []
        for layer_idx in range(self.expert_start_layer, len(self.decoder.layers)):
            if self.expert_mode == "qkvo":
                banks = (self.q_delta, self.k_delta, self.v_delta, self.o_delta)
            elif self.expert_mode == "shared_qkvo":
                banks = (self.lr_delta,)
            else:
                banks = (self.kv_delta,)
            terms.extend(bank[layer_idx][dataset_name].squared_l2() for bank in banks)
        return torch.stack(terms).sum()

    @torch.no_grad()
    def predict_shared(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        """Return source-agnostic disease probabilities with all experts disabled."""
        return self(
            image_features,
            text_features,
            dataset_name=None,
            use_dataset_expert=False,
        ).probabilities
