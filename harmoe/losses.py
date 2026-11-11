"""Masked multi-source supervision and HarMoE regularizers."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import HarMoEOutput


def masked_binary_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Binary classification loss for labels in ``{-1, 0, 1}``.

    HarMoE uses two logits per disease. Unknown entries (``-1``) do not
    contribute to the loss, preventing unannotated classes from becoming false
    negatives when datasets are mapped into the global vocabulary.
    """
    if logits.shape[:-1] != labels.shape or logits.shape[-1] != 2:
        raise ValueError("expected logits [B, C, 2] and labels [B, C]")
    known = labels != -1
    if not known.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[known], labels[known].long())


def orthogonality_loss(shared: Tensor, dataset_residual: Tensor, eps: float = 1e-6) -> Tensor:
    shared = F.normalize(shared, dim=-1, eps=eps)
    dataset_residual = F.normalize(dataset_residual, dim=-1, eps=eps)
    return (shared.mul(dataset_residual).sum(dim=-1).square()).mean()


@dataclass
class HarMoELossOutput:
    total: Tensor
    classification: Tensor
    orthogonality: Tensor
    expert_regularization: Tensor


class HarMoELoss(nn.Module):
    def __init__(self, lambda_orth: float = 1.0, lambda_reg: float = 1.0) -> None:
        super().__init__()
        self.lambda_orth = lambda_orth
        self.lambda_reg = lambda_reg

    def forward(
        self,
        output: HarMoEOutput,
        labels: Tensor,
        expert_regularization: Tensor,
    ) -> HarMoELossOutput:
        classification = masked_binary_cross_entropy(output.logits, labels)
        orthogonality = orthogonality_loss(output.shared, output.dataset_residual)
        total = (
            classification
            + self.lambda_orth * orthogonality
            + self.lambda_reg * expert_regularization
        )
        return HarMoELossOutput(total, classification, orthogonality, expert_regularization)
