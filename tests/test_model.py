import pytest
import torch

from harmoe import HarMoEHead


@pytest.mark.parametrize("mode", ["qkvo", "kv", "shared_qkvo"])
def test_forward_shapes(mode):
    model = HarMoEHead(
        dim=32,
        num_layers=3,
        rank=4,
        expert_start_layer=1,
        dataset_names=("a", "b"),
        expert_mode=mode,
        num_heads=4,
        dropout=0.0,
    ).eval()
    output = model(torch.randn(2, 9, 32), torch.randn(5, 32), "a")
    assert output.logits.shape == (2, 5, 2)
    assert output.shared.shape == (2, 32)
    assert output.dataset_residual.shape == (2, 32)


def test_expert_free_inference_needs_no_dataset_name():
    model = HarMoEHead(dim=32, num_heads=4, dropout=0.0).eval()
    output = model(
        torch.randn(2, 9, 32),
        torch.randn(5, 32),
        dataset_name=None,
        use_dataset_expert=False,
    )
    assert torch.allclose(output.dataset_residual, torch.zeros_like(output.dataset_residual))


def test_unknown_dataset_is_rejected_during_routing():
    model = HarMoEHead(dim=32, num_heads=4)
    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 32), torch.randn(2, 32), "unknown")
