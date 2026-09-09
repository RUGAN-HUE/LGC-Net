# Internal release checklist

- [ ] Training and evaluation scripts reproduce the exact final paper protocol.
- [ ] No dependency on the unpublished negative-space project.
- [ ] No hard-coded `E:\\PycharmProjects` or other absolute local paths.
- [ ] No passwords, API keys, tokens, personal emails, or account credentials.
- [ ] No original cover images or embedded image data.
- [ ] No virtual environments, caches, checkpoints, or large generated artifacts.
- [ ] All third-party baselines are represented by links, version identifiers, adapters, and notices rather than copied repositories.
- [ ] Requirements contain pinned, verified package versions.
- [ ] Final result JSON files agree with every manuscript table and figure.
- [x] Reconstructed two-level Hungarian Max IoU produces the expected metric scale on five later checkpoints.
- [x] Later retraining checkpoints are explicitly separated from the original manuscript result archive.
- [ ] Repository remains private until the author approves publication.
