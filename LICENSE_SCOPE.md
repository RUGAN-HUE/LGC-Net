# License scope and non-commercial research boundary

Copyright 2026 Lei Zhu, Jiahao Li, Xiaoyan Xue, and Yuan Zhang.

## Author-created software

Except for the third-party-derived patch files listed below, the original LGC-Net source code, command-line tools, baseline adapters, and configuration files in this repository are made available under the [PolyForm Noncommercial License 1.0.0](LICENSE). That license permits use, modification, and redistribution for noncommercial purposes, including noncommercial scientific research, experimentation, teaching, and scholarly publication. This repository grants no permission for commercial use.

Commercial use, or use directed toward commercial advantage or monetary compensation, requires separate prior written permission from all applicable rights holders.

## Documentation and result summaries

Author-created documentation and machine-readable result summaries are licensed under [Creative Commons Attribution-NonCommercial 4.0 International](LICENSES/CC-BY-NC-4.0.txt). Reusers must provide attribution, link to the license, indicate changes, and must not use those materials for commercial purposes.

## Third-party-derived patches

The following patch files contain context from the identified upstream projects and remain governed by the corresponding upstream license:

- `baselines/patches/deeplayout_booklayout.patch`: Apache License 2.0;
- `baselines/patches/layoutdm_compatibility.patch`: Apache License 2.0;
- `baselines/patches/layoutganpp_booklayout.patch`: GNU Affero General Public License version 3.

Copies of the applicable terms are provided in `LICENSES/`. All other upstream code must be obtained from the repositories and commits identified in `docs/THIRD_PARTY_NOTICE.md`; this repository does not relicense it.

## Excluded materials

No original book-cover pixels, private checkpoints, credentials, or unpublished negative-space experiments are licensed or distributed by this repository. The BookLayout-Bi structured dataset is maintained and licensed separately.

## Scholarly attribution

Compliance with the software license does not replace normal scholarly citation. Publications using this repository should cite the associated LGC-Net paper and the archived v1.0.0 software release using its version DOI, [`10.5281/zenodo.22684889`](https://doi.org/10.5281/zenodo.22684889). Version 1.0.0 citation metadata are provided in `CITATION.cff`, and `.zenodo.json` supplies Zenodo-compatible software and license metadata.
