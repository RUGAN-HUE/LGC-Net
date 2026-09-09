# Baseline reproduction boundary

The manuscript compares LGC-Net with four baseline models. Complete copies of the third-party repositories are intentionally excluded. This directory will contain only author-created data converters, configuration notes, and evaluation adapters after each file has been checked for local paths and license compatibility.

## Upstream implementations

| Method | Upstream repository used as the implementation basis | Upstream license found in the local archive |
|---|---|---|
| LayoutVAE | https://github.com/kampta/DeepLayout (`layout_vae`) | Apache License 2.0 |
| LayoutTransformer | https://github.com/kampta/DeepLayout | Apache License 2.0 |
| LayoutGAN++ | https://github.com/ktrk115/const_layout | GNU AGPLv3 |
| LayoutDM | https://github.com/CyberAgentAILab/layout-dm | Apache License 2.0 |

## Pending before public release

- record the exact upstream commit or archive identifier used for each run;
- add the four-class BookLayout-Bi conversion script;
- add method-specific inference adapters and unified output conversion;
- document non-converged runs and the 193-versus-194 paired-sample difference;
- retain all notices required by the relevant upstream license.

