# Internal release checklist

- [x] Training and evaluation scripts encode the audited final paper protocol; the expensive full retraining is documented rather than rerun as a release check.
- [x] No dependency on the unpublished negative-space project.
- [x] Public source files use command-line paths rather than absolute local paths.
- [x] No passwords, API keys, tokens, personal emails, or account credentials.
- [x] No original cover images or embedded image data.
- [x] No virtual environments, caches, checkpoints, or large generated artifacts.
- [x] All third-party baselines are represented by links, version identifiers, adapters, patches, and notices rather than copied repositories.
- [x] Requirements contain pinned package versions verified in the local reconstruction environment.
- [x] Authoritative main-comparison JSON agrees with the final manuscript values, including LayoutTransformer IoU 0.1566 +/- 0.0031.
- [x] Reconstructed two-level Hungarian Max IoU produces the expected metric scale on five later checkpoints.
- [x] Later retraining checkpoints are explicitly separated from the original manuscript result archive.
- [x] Exact upstream baseline commits were audited against the retained local experiment copies.
- [x] LayoutGAN++ and LayoutDM 193-versus-194 test coverage is represented by stable sample identifiers.
- [x] Noncommercial software, documentation, result-summary, and third-party patch license scopes are explicit.
- [x] `CITATION.cff` contains version 1.0.0 metadata without an unissued DOI or release date.
- [x] `.zenodo.json` explicitly registers the archive as software, uses Zenodo's validated PolyForm Noncommercial identifier, and preserves the documentation/result and third-party patch boundaries in its description.
- [x] Public-release content review is complete; changing repository visibility remains an author-only step.
