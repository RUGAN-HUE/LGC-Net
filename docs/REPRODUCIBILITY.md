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

Use Python 3.12 and install the dependencies listed in `requirements.txt`. Select the appropriate PyTorch accelerator build for the local operating system and CUDA version.

## 3. Verify the fixed split files

The final protocol expects the following counts:

| Split | Records |
|---|---:|
| Stage 1 training | 8,971 |
| Stage 1 validation | 996 |
| Stage 2 training | 1,572 |
| Stage 2 validation | 174 |
| Held-out test | 194 |

The held-out test records must not enter either training or validation split.

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

## Repeated-run interpretation

Training and prior sampling are stochastic. The manuscript reports the mean and sample standard deviation over multiple seeds. A later five-checkpoint pipeline verification is retained separately from the original manuscript summary in `results/`; it must not be described as the original reported runs.

