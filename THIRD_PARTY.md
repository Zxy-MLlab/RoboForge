# Third-party provenance

This list records upstreams inspected or used by RoboForge. Unless explicitly
described as a wrapper or port, RoboForge does not copy agent loops,
coordinators, repair loops, evolutionary searches, or training systems from
ASPIRE, CaP-X, HELIX, ENPIRE, or RHO.

| Upstream | Revision / state inspected | License | Reused mechanism | RoboForge location |
| --- | --- | --- | --- | --- |
| [OpenHands software-agent-sdk](https://github.com/OpenHands/software-agent-sdk) | `704cbe6015e3d59cabe04632175d99df2d448999`; installed `1.44.1` | MIT | Sole LLM conversation/agent substrate; public editor, terminal, planning and task extensions | `roboforge/runtime.py`, `roboforge/cli.py`; no source copied |
| [NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE) | `f4c8939aab0af9b97690c561bd80e282940f7886` | Apache-2.0 plus third-party notices | Primitive trace and trial artifact lifecycle concepts | `embodied_codex/deployments/libero.py`, `scripts/`, `evaluation/`; independently implemented |
| [capgym/CaP-X](https://github.com/capgym/cap-x) | `53e9966d7a8e2fa7494676772bccc35280f5c0ed` | MIT | Code-as-policy LIBERO environment/API boundary | `embodied_codex/deployments/libero.py`, `embodied_codex/adapters/libero_sdk.py` |
| [KE7/HELIX](https://github.com/KE7/HELIX) | `858b6bcbafd9bb1ca9226e1f03c83d8cbe3a0db6` | BSD-3-Clause | Fair baseline/candidate comparison concepts | `scripts/run_paired_libero_evaluation.py`, `evaluation/`; no loop copied |
| [NVIDIA ENPIRE](https://research.nvidia.com/labs/gear/enpire/) | Page inspected 2026-09-02; no checkout | Project/publication terms | External reset/execute/evaluate ownership reference | No code copied |
| [RHO](https://rho-robotics.github.io/) | Page inspected 2026-09-02; no checkout | Project/publication terms | External experiment/evaluation ownership reference | No code copied |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `8f1084e3132a39270c3a13ebe37270a43ece2a01`; external vendor compatibility edits | MIT | Real benchmark/tasks, initial states and official `check_success` | External vendor used by `embodied_codex/deployments/libero.py` |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | `856dde20aee659246248e20734ef9ba5214f5e44` | Apache-2.0 | Open-vocabulary RGB detector | Wrapper in `embodied_codex/capabilities/open_vocab_rgbd.py` |
| [Segment Anything](https://github.com/facebookresearch/segment-anything) | `dca509fe793f601edb92606367a655c15ac00fdf` | Apache-2.0 | SAM ViT-B mask prediction | Wrapper in `embodied_codex/capabilities/open_vocab_rgbd.py` |
| [GraspNet baseline](https://github.com/graspnet/graspnet-baseline) | `280c215129f759ed8649cb4e89fc5dfee55f4f80` | Academic/non-commercial research agreement | 6-DoF grasp proposals | Wrappers in `embodied_codex/capabilities/graspnet_*.py` |

## Checkpoint provenance

| Asset | Source | SHA-256 | Runtime path |
| --- | --- | --- | --- |
| GroundingDINO Swin-T OGC | Hugging Face `ShilongLiu/GroundingDINO` | `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799` | `/root/autodl-tmp/roboforge-assets/checkpoints/groundingdino_swint_ogc.pth` |
| SAM ViT-B | Meta public checkpoint | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` | `/root/autodl-tmp/roboforge-assets/checkpoints/sam_vit_b_01ec64.pth` |
| GraspNet checkpoint-rs | Google Drive source pinned by `scripts/download_libero_checkpoints.py` | `60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868` | `/root/autodl-tmp/roboforge-assets/checkpoints/graspnet-checkpoint-rs.tar` |

Upstream source checkouts and model weights are external runtime materials and
are not committed to this repository.
