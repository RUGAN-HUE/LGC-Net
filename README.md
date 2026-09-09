# LGC-Net

> Preparation copy: this repository is private and is not yet ready for public release.

LGC-Net is a lightweight conditional variational autoencoder for editable book-cover layout generation. The release package is being reconstructed from the exact final experimental protocol reported in the manuscript.

## Planned contents

- standalone LGC-Net model implementation;
- fixed data-split loader using public BookLayout-Bi identifiers;
- two-stage training procedure;
- five-seed evaluation protocol;
- evaluation metrics used in the manuscript;
- adapters and instructions for the reported baselines;
- machine-readable final results and ablation summaries;
- reproducibility and third-party software documentation.

## Reproducibility targets

- final dataset size: 10,161 records;
- held-out test size: 194 manually annotated records;
- latent dimension: 32;
- training random seeds: 0, 1, 2, 3, and 4;
- reported LGC-Net element-wise IoU: 0.2027 +/- 0.0044;
- reported LGC-Net maximum IoU: 0.2500 +/- 0.0028;
- reported LayoutTransformer element-wise IoU: 0.1566 +/- 0.0031;
- reported relative element-wise IoU improvement: approximately 29%.

The `results` directory distinguishes the authoritative values transcribed from the final manuscript from a separate five-checkpoint pipeline verification. The latter checks the reconstructed public data and metric pipeline but is not represented as the original set of manuscript runs.

## Release boundary

This repository will not include original cover images, credentials, private paths, virtual environments, checkpoints by default, unpublished negative-space experiments, or complete copies of third-party baseline repositories.

## License

The code license will be selected after author and institutional confirmation. No license is granted by this preparation copy.
