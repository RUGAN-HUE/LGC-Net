# Third-party software notice

The reported experiments use or adapt implementations of LayoutVAE, LayoutTransformer, LayoutGAN++, and LayoutDM. The exact source snapshots audited against the retained local experiment copies are:

| Implementation | Repository | Commit | License |
|---|---|---|---|
| LayoutVAE and LayoutTransformer | https://github.com/kampta/DeepLayout | `60ebaca7f29a14d92c455f83e681c0bd8e2962fe` | Apache License 2.0 |
| LayoutGAN++ | https://github.com/ktrk115/const_layout | `5287480505939345543fff0b9f2e5d541e6f84e2` | GNU AGPL version 3 |
| LayoutDM | https://github.com/CyberAgentAILab/layout-dm | `873b5eebe4c61862e5c08a10859accf65a168dfd` | Apache License 2.0 |

Complete third-party repositories are not copied into this project. Author-created adapters and the minimal patches necessary to reproduce the reported protocol are included with explicit license boundaries.

The local DeepLayout snapshot differed from the pinned upstream commit in three small places: the LayoutTransformer epoch default was set to 300, a method-name typo was corrected, and deprecated `np.int` usage was replaced. The LayoutDM snapshot contained two compatibility changes for current PyTorch releases. The LayoutGAN++ snapshot increased the maximum retained element count from 9 to 26, added a PyTorch loading compatibility argument, and bypassed an unrelated pretrained Layout-FID checkpoint. These changes are represented as patch files in `baselines/patches/`.

The patch change sets were prepared for this release on 2026-09-10 and make every modified upstream line visible. The upstream LayoutDM distribution's attribution notice is retained here: Copyright 2023 Naoto Inoue.

The author-created adapters in this repository do not incorporate or relicense upstream model source code. Users must obtain each upstream implementation under its own license. Patch files retain the license of the corresponding upstream source: the DeepLayout and LayoutDM patches are under Apache-2.0, and the LayoutGAN++ patch is under AGPL-3.0. Copies of those terms are provided in `LICENSES/`. Users who modify, run as a network service, or distribute the AGPL-covered upstream program remain responsible for complying with its terms.
