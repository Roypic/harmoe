# HarMoE: Multi-Source Chest Radiograph Pretraining with Dataset-Disentangled Experts

Official implementation of **HarMoE**, a framework for learning a transferable
chest radiograph representation from heterogeneous datasets with partially
overlapping disease labels.

HarMoE combines four ideas:

1. a global disease vocabulary with unknown-label masking;
2. disease prompts encoded by a frozen clinical text encoder;
3. a shared image--text decoder plus lightweight dataset-routed residual experts;
4. shared-only inference, which removes every source-specific expert at test time.

## Installation

```bash
git clone https://github.com/Roypic/harmoe.git
cd harmoe
pip install -e ".[dev]"
pytest -q
```

The core package only requires PyTorch. A full experiment additionally needs
the image and text encoders used to produce `[B, P, D]` visual tokens and
`[C, D]` disease-prompt embeddings.

## Core usage

```python
from harmoe import HarMoEHead, HarMoELoss

head = HarMoEHead(
    dim=768,
    num_layers=4,
    expert_start_layer=2,
    rank=8,
    dataset_names=("mimic", "chexpert", "chestxray14", "padchest"),
    expert_mode="kv",
)

# Pretraining: route the sample through its source expert.
output = head(image_tokens, prompt_embeddings, dataset_name="chexpert")
losses = HarMoELoss(lambda_orth=0.01, lambda_reg=1e-6)(
    output,
    global_labels,  # 1 present, 0 absent, -1 unknown
    head.residual_l2("chexpert"),
)
losses.total.backward()

# Zero-shot/OOD inference: disable source experts.
probabilities = head.predict_shared(image_tokens, new_prompt_embeddings)
```

Run `python -m examples.minimal_train_step` for an executable feature-level
example. The paper configuration is in `configs/harmoe_vitb16.yaml`.

## Label harmonization

Each dataset keeps its native annotated classes. After synonym normalization,
the union of those class sets forms a global vocabulary. Labels absent from a
dataset's annotation schema are set to `-1` and excluded from the objective;
they are not treated as negatives. `harmoe.vocabulary` implements this mapping.

## Paper configuration

| Component | Setting |
|---|---|
| Pretraining sources | MIMIC-CXR, CheXpert, ChestX-ray14, PadChest |
| Image encoder | ViT-B/16, 224 x 224 input |
| Text encoder | BioClinicalMPBERT, frozen |
| Prompt | `There is {disease}.` |
| Decoder | 4 layers, 4 attention heads |
| Dataset experts | Low-rank visual-memory residuals, rank 8 |
| Expert insertion | Layers 3--4 (`expert_start_layer=2`) |
| Loss weights | `lambda_orth=0.01`, `lambda_reg=1e-6` |
| Optimizer | AdamW, lr `5e-5`, weight decay `0.02` |
| Schedule | 100 epochs, 20-epoch warm-up, cosine decay |
| Batch size | 32 |

Dataset paths are intentionally not embedded in the code. Configure local CSV
manifests in `configs/harmoe_vitb16.yaml`; images and checkpoints are not stored
in Git.

The released reference configuration uses `expert_mode="kv"`. The package also
contains the `qkvo` and shared-residual variants used in architectural ablations.

## Repository layout

```text
harmoe/                  Core model, objective, and vocabulary mapping
configs/                 Reproducible model/training configuration
examples/                Minimal integration example
tests/                   Shape, masking, and routing tests
```

## Citation

```bibtex
@inproceedings{luo2026harmoe,
  title={HarMoE: Multi-Source Chest Radiograph Pretraining with Dataset-Disentangled Experts},
  author={Luo, Haozhe and Zhou, Ziyu and Shu, Shelley Zixin and Reyes, Mauricio},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2026}
}
```
