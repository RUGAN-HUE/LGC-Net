# Third-party software notice

The reported experiments use or adapt implementations of LayoutVAE, LayoutTransformer, LayoutGAN++, and LayoutDM. The exact source snapshots audited against the retained local experiment copies are:

| Implementation | Repository | Commit | License |
|---|---|---|---|
| LayoutVAE and LayoutTransformer | https://github.com/kampta/DeepLayout | `60ebaca7f29a14d92c455f83e681c0bd8e2962fe` | Apache License 2.0 |
| LayoutGAN++ | https://github.com/ktrk115/const_layout | `5287480505939345543fff0b9f2e5d541e6f84e2` | GNU AGPL version 3 |
| LayoutDM | https://github.com/CyberAgentAILab/layout-dm | `873b5eebe4c61862e5c08a10859accf65a168dfd` | Apache License 2.0 |

Complete third-party repositories will not be copied into this project. Only author-created adapters or patches that are necessary to reproduce the reported protocol will be included after license review.

The local DeepLayout snapshot differed from the pinned upstream commit in three small places: the LayoutTransformer epoch default was set to 300, a method-name typo was corrected, and deprecated `np.int` usage was replaced. The LayoutDM snapshot contained two compatibility changes for current PyTorch releases. The LayoutGAN++ snapshot increased the maximum retained element count from 9 to 26, added a PyTorch loading compatibility argument, and bypassed an unrelated pretrained Layout-FID checkpoint. These changes are represented as patch files in `baselines/patches/`.

The author-created adapters in this repository do not incorporate or relicense upstream model source code. Users must obtain each upstream implementation under its own license. In particular, LayoutGAN++ is distributed under AGPL-3.0; users who modify or distribute that upstream program remain responsible for complying with its terms.
