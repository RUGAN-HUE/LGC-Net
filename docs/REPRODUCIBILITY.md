# Reproducibility guide

> This guide is a preparation draft. Do not represent the repository as a final public release until every release checklist item is resolved.

## 1. Clone the two sibling repositories

```text
workspace/
├─ LGC-Net/
└─ BookLayout-Bi/
```

The code consumes structured annotations only; original book-cover pixels are not required for the reported coordinate-generation training and evaluation pipeline.

## 2. Create an isolated Python environment

The reconstructed standalone LGC-Net package was verified under Python 3.12 with the dependencies listed in `requirements.txt`. Select the appropriate PyTorch accelerator build for the local operating system and CUDA version. The third-party baselines retain their own historical environment requirements; use isolated environments and follow their pinned upstream documentation.

## 3. Verify the fixed split files

From the `LGC-Net` repository root, first run both release validators:

```bash
python ../BookLayout-Bi/tools/validate_release.py
python tools/validate_release.py --booklayout-root ../BookLayout-Bi
```

The final protocol expects the following counts:

| Split | Records |
|---|---:|
| Stage 1 training | 8,971 |
| Stage 1 validation | 996 |
| Stage 2 training | 1,572 |
| Stage 2 validation | 174 |
| Held-out test | 194 |

The held-out test records must not enter either training or validation split.

LGC-Net uses `max_seq_len = 15` and retains the first 15 elements in stored record order for longer inputs, matching the original training code. Twenty-four dataset records exceed this limit: 21 in Stage 1 training and 3 in Stage 1 validation. No Stage 2 or test record is truncated.

The BookLayout-Bi data card documents a post-hoc duplicate audit. The fixed split is retained to reproduce the submitted results, but two test records have same-cover counterparts under different non-test identifiers. The manuscript must disclose this limitation rather than describe the split as image-level duplicate-free.

## 4. Train the final five-seed configuration

From the `LGC-Net` repository root:

```bash
python -m src.train \
  --dataset-json ../BookLayout-Bi/data/booklayout_bi_annotations.json \
  --split-root ../BookLayout-Bi/data/splits
```

The default configuration is `configs/final_protocol.json`. It trains seeds 0–4 with the mixed two-stage protocol and a fixed data-split seed of 42.

## 5. Evaluate one checkpoint

```bash
python -m src.evaluate \
  --checkpoint checkpoints/seed0/stage2_best.pt \
  --dataset-json ../BookLayout-Bi/data/booklayout_bi_annotations.json \
  --test-split ../BookLayout-Bi/data/splits/test.txt
```

The evaluation reports element-wise IoU, two-level Hungarian Max IoU in the shared four-class label space, overlap, maximum overlap, alignment error, and out-of-bounds rate.

## 6. Reproduce baseline interfaces

The `baselines/` directory exports the fixed data to four-class COCO-style JSON, remaps it for the LayoutGAN++ and LayoutDM PubLayNet loaders, records exact upstream commits and patches, converts generated outputs to a common identifier-based schema, and evaluates every method through one metric implementation. See `baselines/README.md` for the commands and reported-run qualifications.

## Repeated-run interpretation

Training and prior sampling are stochastic. The manuscript reports the mean and sample standard deviation over multiple seeds. A later five-checkpoint pipeline verification is retained separately from the original manuscript summary in `results/`; it must not be described as the original reported runs.
