# Internal release checklist

- [ ] Training and evaluation scripts reproduce the exact final paper protocol.
- [x] No dependency on the unpublished negative-space project.
- [x] Public source files use command-line paths rather than absolute local paths.
- [ ] No passwords, API keys, tokens, personal emails, or account credentials.
- [x] No original cover images or embedded image data.
- [x] No virtual environments, caches, checkpoints, or large generated artifacts.
- [x] All third-party baselines are represented by links, version identifiers, adapters, patches, and notices rather than copied repositories.
- [ ] Requirements contain pinned, verified package versions.
- [ ] Final result JSON files agree with every manuscript table and figure.
- [x] Reconstructed two-level Hungarian Max IoU produces the expected metric scale on five later checkpoints.
- [x] Later retraining checkpoints are explicitly separated from the original manuscript result archive.
- [x] Exact upstream baseline commits were audited against the retained local experiment copies.
- [x] LayoutGAN++ and LayoutDM 193-versus-194 test coverage is represented by stable sample identifiers.
- [ ] Repository remains private until the author approves publication.
