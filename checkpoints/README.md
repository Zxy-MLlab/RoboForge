# Checkpoints and External Models

Large model checkpoints are intentionally excluded from this repository. The
generic kernel does not need any checkpoint; only the optional LIBERO adapter
does. Install the adapter dependencies first:

```bash
bash scripts/install_libero.sh
python scripts/download_libero_checkpoints.py
```

Download the public, task-independent models into this directory (or set the
environment variables below):

- GroundingDINO Swin-T OGC: `groundingdino_swint_ogc.pth`
- SAM ViT-B: `sam_vit_b_01ec64.pth`
- GraspNet RGB-D: `graspnet-checkpoint-rs.tar`

The adapter resolves these paths from:

```text
ROBOFORGE_GROUNDINGDINO_ROOT
ROBOFORGE_GROUNDINGDINO_CONFIG
ROBOFORGE_GROUNDINGDINO_CHECKPOINT
ROBOFORGE_SAM_ROOT
ROBOFORGE_SAM_CHECKPOINT
ROBOFORGE_GRASPNET_ROOT
ROBOFORGE_GRASPNET_CHECKPOINT
```

Before running LIBERO, verify the files and runtime without starting an episode:

```bash
python scripts/download_libero_checkpoints.py --verify-only
roboforge doctor --adapter libero
```

Keep the resulting hashes and upstream URLs in the run provenance. Never commit
large or privately sourced weights to this repository. The benchmark runner
performs its own provenance and contamination checks; these are not kernel
requirements.
