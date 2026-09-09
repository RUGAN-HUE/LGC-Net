# Baseline reproduction

The manuscript compares LGC-Net with LayoutVAE, LayoutTransformer, LayoutGAN++, and LayoutDM. Complete copies of those third-party repositories are intentionally excluded. This directory contains only author-created data exporters, inference/output adapters, and small patches applied to separately cloned upstream repositories.

## Fixed upstream versions

| Method | Upstream repository and commit | License |
|---|---|---|
| LayoutVAE | `kampta/DeepLayout` at `60ebaca7f29a14d92c455f83e681c0bd8e2962fe` (`layout_vae`) | Apache-2.0 |
| LayoutTransformer | `kampta/DeepLayout` at `60ebaca7f29a14d92c455f83e681c0bd8e2962fe` | Apache-2.0 |
| LayoutGAN++ | `ktrk115/const_layout` at `5287480505939345543fff0b9f2e5d541e6f84e2` | AGPL-3.0 |
| LayoutDM | `CyberAgentAILab/layout-dm` at `873b5eebe4c61862e5c08a10859accf65a168dfd` | Apache-2.0 |

The repositories are cloned separately; their source files are not relicensed or copied here. See `../docs/THIRD_PARTY_NOTICE.md`.

## 1. Export the fixed data

```bash
python baselines/export_booklayout_to_coco.py \
  --dataset-json ../BookLayout-Bi/data/booklayout_bi_annotations.json \
  --split-root ../BookLayout-Bi/data/splits \
  --output-dir baseline_data
```

This produces the five fixed splits and `non_test.json` in the shared four-class space:

- `Title` remains `Title`;
- `Subtitle`, `Author`, `Publisher`, `Endorsement`, and `Series` become `Text`;
- `Main_Image` becomes `Figure`;
- `Logo` and `Badge` become `List`.

For LayoutGAN++ and LayoutDM, remap the files to the PubLayNet category identifiers:

```bash
python baselines/remap_coco_to_publaynet.py baseline_data/stage1_train.json upstream/train.json
python baselines/remap_coco_to_publaynet.py baseline_data/test.json upstream/val.json
```

The JSON files contain dimensions and boxes but no image pixels. The upstream loaders use the geometric annotations only.

## 2. Apply the recorded patches

From each upstream repository root:

```bash
git checkout <commit-listed-above>
git apply /path/to/LGC-Net/baselines/patches/<corresponding-patch>
```

The patches increase the LayoutTransformer epoch default, apply compatibility fixes for current NumPy/PyTorch versions, allow the observed BookLayout-Bi sequence length in LayoutGAN++, and disable loading an unrelated pretrained Layout-FID network. Layout FID is not reported in the manuscript and is not used for checkpoint selection; the manuscript metrics are computed by `src.evaluate_predictions`.

## 3. Method-specific execution

The complete settings are recorded in `../configs/baseline_protocols.json`.

### LayoutTransformer

Run the upstream trainer with the exported training and validation JSON files for each reported seed. The manuscript configuration uses 300 epochs, batch size 64, six layers, eight attention heads, 512-dimensional embeddings, and 8-bit coordinate quantization. Convert a checkpoint to the common prediction schema with:

```bash
python baselines/infer_layouttransformer.py \
  --upstream-root /path/to/DeepLayout \
  --checkpoint /path/to/checkpoint.pth \
  --test-coco baseline_data/test.json \
  --output predictions/layouttransformer_seed0.json \
  --seed 0
```

### LayoutVAE

The author-created training wrapper uses the official `layout_vae` model classes with the recorded reconstruction and KL settings:

```bash
python baselines/train_layoutvae_booklayout.py \
  --upstream-root /path/to/DeepLayout \
  --train-coco baseline_data/stage1_train.json \
  --validation-coco baseline_data/stage1_validation.json \
  --output-dir checkpoints/layoutvae_seed0 \
  --seed 0
```

Generate label-conditioned predictions without passing ground-truth coordinates:

```bash
python baselines/infer_layoutvae.py \
  --upstream-root /path/to/DeepLayout \
  --checkpoint checkpoints/layoutvae_seed0/epoch_080.pt \
  --test-coco baseline_data/test.json \
  --output predictions/layoutvae_seed0.json \
  --seed 0
```

### LayoutGAN++ and LayoutDM

Use the upstream training and generation entry points with the exact settings in `baseline_protocols.json`. Their PubLayNet loaders apply an `H < W` exclusion, so one landscape test cover is removed and 193 covers remain. Convert a trusted local pickle output as follows:

```bash
python baselines/adapt_layout_pickle.py \
  --pickle /path/to/generated.pkl \
  --dataset-json ../BookLayout-Bi/data/booklayout_bi_annotations.json \
  --test-split ../BookLayout-Bi/data/splits/test.txt \
  --output predictions/layoutganpp_seed2.json \
  --method LayoutGAN++ \
  --filter-landscape
```

Python pickle is unsafe for untrusted files. This adapter should only be used on outputs generated in a trusted local environment.

## 4. Unified evaluation

```bash
python -m src.evaluate_predictions \
  --predictions predictions/layouttransformer_seed0.json \
  --dataset-json ../BookLayout-Bi/data/booklayout_bi_annotations.json \
  --test-split ../BookLayout-Bi/data/splits/test.txt
```

Use `--require-all` for methods expected to cover all 194 test records. The command reports element-wise IoU, overlap, maximum overlap, alignment error, out-of-bounds rate, and the two-level Hungarian Max IoU used in the manuscript.

## Reported-run qualifications

- LayoutVAE, LayoutTransformer, and LGC-Net report five runs.
- LayoutDM reports three runs because of its higher training cost.
- LayoutGAN++ reports four valid converged runs. Seed 0 collapsed to degenerate layouts and seed 1 did not produce a usable checkpoint; these failures are documented rather than silently averaged.
- LayoutGAN++ and LayoutDM are evaluated on 193 covers because their original loaders exclude one landscape cover. The other methods use all 194.
