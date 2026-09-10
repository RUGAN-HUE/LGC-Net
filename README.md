# LGC-Net

LGC-Net is a lightweight conditional variational autoencoder for editable book-cover layout generation. This repository contains the standalone model, the final M-32-G two-stage training protocol, unified metrics, baseline adapters, and machine-readable reported results.

The repository metadata describe version 1.0.0. Citation metadata are provided in [`CITATION.cff`](CITATION.cff); the persistent DOI will be added after the GitHub release has been archived.

## Repository contents

- standalone LGC-Net model implementation;
- fixed data-split loader using public BookLayout-Bi identifiers;
- two-stage training procedure;
- five-seed evaluation protocol;
- evaluation metrics used in the manuscript;
- data exporters, patches, inference adapters, and instructions for the reported baselines;
- machine-readable final results and a separate later-checkpoint verification;
- reproducibility and third-party software documentation.

## Reproducibility targets

- final dataset size: 10,161 records;
- held-out test size: 194 manually annotated records;
- latent dimension: 32;
- maximum LGC-Net sequence length: 15 elements;
- training random seeds: 0, 1, 2, 3, and 4;
- reported LGC-Net element-wise IoU: 0.2027 +/- 0.0044;
- reported LGC-Net maximum IoU: 0.2500 +/- 0.0028;
- reported LayoutTransformer element-wise IoU: 0.1566 +/- 0.0031;
- reported relative element-wise IoU improvement: approximately 29%.

The `results` directory distinguishes the authoritative values transcribed from the final manuscript from a separate five-checkpoint pipeline verification. The latter checks the reconstructed public data and metric pipeline but is not represented as the original set of manuscript runs.

The original LGC-Net implementation retains the first 15 elements in stored record order when a training record is longer. This affects 24 of 10,161 records (21 in Stage 1 training and 3 in Stage 1 validation); no Stage 2 or held-out test record exceeds 15 elements. The released loader preserves that behavior for exact protocol compatibility.

## Quick verification

Place this repository beside the BookLayout-Bi repository and run:

```bash
python ../BookLayout-Bi/tools/validate_release.py
python tools/validate_release.py --booklayout-root ../BookLayout-Bi
```

To launch the full five-seed training protocol, run:

```bash
python -m src.train \
  --dataset-json ../BookLayout-Bi/data/booklayout_bi_annotations.json \
  --split-root ../BookLayout-Bi/data/splits
```

Training is intentionally not launched by the verification command because the five-seed, two-stage protocol is computationally expensive. See `docs/REPRODUCIBILITY.md` for evaluation and baseline commands.

## Release boundary

This repository does not include original cover images, credentials, private paths, virtual environments, checkpoints, unpublished negative-space experiments, or complete copies of third-party baseline repositories.

The separately maintained BookLayout-Bi repository contains only structured annotations, hashes, and fixed split identifiers. Original cover pixels are not needed by this coordinate-only generation pipeline.

## License

The author-created software is available for noncommercial use under the PolyForm Noncommercial License 1.0.0. Author-created documentation and result summaries are available under CC BY-NC 4.0. Commercial use is not licensed.

Third-party-derived patch files retain their corresponding upstream licenses. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md), [NOTICE](NOTICE), and [docs/THIRD_PARTY_NOTICE.md](docs/THIRD_PARTY_NOTICE.md) for the exact boundaries. No license in this repository applies to original book-cover artwork or other third-party content.
