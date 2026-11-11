import torch

from harmoe import build_global_vocabulary, masked_binary_cross_entropy


def test_unknown_labels_do_not_change_loss():
    logits = torch.tensor([[[2.0, -1.0], [-4.0, 4.0], [1.0, 3.0]]])
    labels = torch.tensor([[0, 1, -1]])
    loss = masked_binary_cross_entropy(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits[:, :2].reshape(-1, 2), labels[:, :2].reshape(-1))
    assert torch.allclose(loss, expected)


def test_global_vocabulary_masks_unannotated_classes():
    vocabulary = build_global_vocabulary({"a": ["edema", "mass"], "b": ["mass", "nodule"]})
    expanded = vocabulary.expand_labels("a", torch.tensor([[1, 0]]))
    assert vocabulary.classes == ("edema", "mass", "nodule")
    assert expanded.tolist() == [[1, 0, -1]]
