"""Minimal feature-level HarMoE training step.

Replace the random tensors with image tokens from a ViT and disease-prompt
embeddings from a frozen clinical text encoder.
"""

import torch

from harmoe import HarMoEHead, HarMoELoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Small dimensions keep this example quick on CPU. Use dim=768, four layers,
# rank=8, and 196 image tokens for the paper configuration.
model = HarMoEHead(
    dim=64,
    num_layers=2,
    rank=4,
    expert_start_layer=1,
    expert_mode="kv",
).to(device)
criterion = HarMoELoss(lambda_orth=0.01, lambda_reg=1e-6)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.02)

image_tokens = torch.randn(2, 16, 64, device=device)
prompt_tokens = torch.randn(5, 64, device=device)
labels = torch.randint(0, 2, (2, 5), device=device)
labels[:, 4:] = -1

output = model(image_tokens, prompt_tokens, dataset_name="chexpert", alpha=1.0)
losses = criterion(output, labels, model.residual_l2("chexpert"))
optimizer.zero_grad()
losses.total.backward()
optimizer.step()

print({name: float(value.detach()) for name, value in vars(losses).items()})

# Source-agnostic inference discards every dataset expert.
probabilities = model.predict_shared(image_tokens, prompt_tokens)
print(probabilities.shape)
