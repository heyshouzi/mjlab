![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).
The framework provides composable building blocks for environment design,
with minimal dependencies and direct access to native MuJoCo data structures.

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/main/source/installation.html).

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions. See the [motion imitation guide](https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html) for preprocessing setup.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
uv run train Mjlab-Tracking-Flat-Booster-K1-No-State-Estimation \
  --registry-name wandb-registry-motions/k1_walk1_subject2.npz:latest \
  --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Booster-K1-No-State-Estimation  --registry-name wandb-registry-motions/k1_walk1_subject2.npz:latest 

```

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.


## Motion Replay

### Converting CSV to NPZ

To convert motion CSV files (e.g., from LAFAN dataset) to NPZ format for use with mjlab:

```bash
# Convert for G1 robot (29 DOF)
uv run python src/mjlab/scripts/csv_to_npz.py \
    --input-file /path/to/motion.csv \
    --output-name my_motion \
    --robot g1 \
    --input-fps 30 \
    --output-fps 50

# Convert for K1 robot (22 DOF)
uv run python src/mjlab/scripts/csv_to_npz.py \
    --input-file /path/to/motion.csv \
    --output-name my_motion \
    --robot k1 \
    --input-fps 30 \
    --output-fps 50

# Process a specific frame range (START END, both inclusive)
uv run python src/mjlab/scripts/csv_to_npz.py \
    --input-file /path/to/motion.csv \
    --output-name my_motion \
    --robot g1 \
    --line-range 122 722
```

**Arguments:**
- `--input-file`: Path to input CSV file (required)
- `--output-name`: Output name for W&B registry (required)
- `--robot`: Robot type: `g1` (Unitree G1, 29 DOF) or `k1` (Booster K1, 22 DOF) (default: g1)
- `--input-fps`: Input CSV framerate (default: 30)
- `--output-fps`: Output NPZ framerate (default: 50)
- `--line-range`: Frame range START END (both inclusive, 1-indexed)
- `--render`: Enable video rendering

The script automatically uploads the converted motion to W&B registry at `motions/<output-name>`.

**G1 joint order (29 joints):**

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0-5 | Left leg (hip_pitch/roll/yaw, knee, ankle_pitch/roll) | 15-21 | Left arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw) |
| 6-11 | Right leg | 22-28 | Right arm |
| 12-14 | Waist (yaw/roll/pitch) | | |

**K1 joint order (22 joints):**

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0 | Head_yaw | 11-16 | Left leg (Hip_pitch/roll/yaw, Knee, Ankle_pitch/roll) |
| 1 | Head_pitch | 17-22 | Right leg |
| 2-6 | Left arm (Shoulder_pitch/roll, Elbow_pitch/yaw) | | |
| 7-10 | Right arm | | |

### Replaying Motion

Replay a motion NPZ file to visualize robot behavior:

```bash
# Replay on G1 robot with viser viewer (default)
uv run python src/mjlab/scripts/replay_npz.py --robot g1 --motion-file /path/to/motion.npz

# Replay on K1 robot
uv run python src/mjlab/scripts/replay_npz.py --robot k1 --motion-file /path/to/motion.npz

# With native MuJoCo viewer (if DISPLAY is available)
uv run python src/mjlab/scripts/replay_npz.py --robot g1 --motion-file /path/to/motion.npz --viewer native

# Load motion from W&B registry (auto-downloads to ~/mjlab/motion/)
uv run python src/mjlab/scripts/replay_npz.py --robot g1 --registry-name my-org/motions/my-motion

# Adjust playback speed (1.0 = real-time, 2.0 = 2x faster)
uv run python src/mjlab/scripts/replay_npz.py --robot g1 --motion-file /path/to/motion.npz --speed 2.0
```

**Arguments:**
- `--robot`: Robot type: `k1` or `g1` (default: k1)
- `--motion-file`: Path to local motion NPZ file
- `--registry-name`: W&B registry artifact name (e.g., `my-org/motions/my-motion`)
- `--speed`: Playback speed multiplier (default: 1.0, 1.0 = real-time)
- `--viewer`: Viewer backend: `auto`, `native`, or `viser` (default: auto)


## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

## Citation

mjlab is used in published research and open-source robotics projects. See the [Research](https://mujocolab.github.io/mjlab/main/source/research.html) page for publications and projects, or share your own in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell).

If you use mjlab in your research, please consider citing:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
