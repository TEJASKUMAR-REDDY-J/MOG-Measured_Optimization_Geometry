# Exp 1: geometry atlas

`atlas.py` trains an incumbent optimizer and, every `atlas_every` steps, logs per 2-D block the momentum's effective dimensions, the gradient-fit θ_c/C_c for every candidate geometry, the Newton–Schulz proxy error, the secant curvature (both the PDF's free version and the same-batch version), the selection scores E_c, the half-batch gradient SNR and the input-activation participation ratio. The first arm's run also writes the oracle checkpoints.

| Config | Stage | Evidence class |
|---|---|---|
| `configs/tiny/exp011_atlas_dryrun.yaml` | 2 | pilot |
| `configs/1m/exp100_atlas.yaml` | 3 | measured |
