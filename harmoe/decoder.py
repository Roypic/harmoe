"""Transformer decoder used by the HarMoE cross-modal head."""

import copy
from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        output = tgt
        weights = None
        for layer in self.layers:
            output, weights = layer(
                output,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(output), weights


class TransformerDecoderLayer(nn.Module):
    """Pre-norm self-attention, image cross-attention, and feed-forward block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_value: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        **_: object,
    ) -> tuple[Tensor, Tensor]:
        normalized = self.norm1(tgt)
        attended, _ = self.self_attn(normalized, normalized, normalized)
        tgt = tgt + self.dropout1(attended)

        normalized = self.norm2(tgt)
        attended, weights = self.multihead_attn(
            query=normalized,
            key=memory,
            value=memory if memory_value is None else memory_value,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(attended)

        normalized = self.norm3(tgt)
        feedforward = self.linear2(self.dropout(F.relu(self.linear1(normalized))))
        return tgt + self.dropout3(feedforward), weights
