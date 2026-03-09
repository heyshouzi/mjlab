"""Replay a motion NPZ file on a tracking task robot.

This script loads a motion ``.npz`` file (either from a local path or a W&B
registry artifact) and plays it back on the robot using a zero-action policy
so that the motion command drives the reference pose visualisation without
any learned control.

.. code-block:: bash

    # Local file – K1 robot (default)
    uv run python replay_motion.py --motion-file /path/to/motion.npz

    # W&B registry artifact – K1 robot
    uv run python replay_motion.py --registry-name my-org/motions/k1-walk

    # Explicit task selection
    uv run python replay_motion.py --task Mjlab-Tracking-Flat-Unitree-G1 \\
        --motion-file /path/to/g1_motion.npz
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

import mjlab
import mjlab.tasks  # noqa: F401  – populate the task registry
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import list_tasks

_DEFAULT_TASK = "Mjlab-Tracking-Flat-Booster-K1"


@dataclass(frozen=True)
class ReplayConfig:
  """Configuration for motion replay."""

  motion_file: str | None = None
  """Path to a local motion ``.npz`` file."""

  registry_name: str | None = None
  """W&B registry artifact name (e.g. ``my-org/motions/k1-walk``).
    ``:latest`` is appended automatically when no alias is specified."""

  task: str = _DEFAULT_TASK
  """Tracking task to use for replay.  Defaults to the Booster K1 task."""

  num_envs: int = 1
  """Number of parallel environments to spawn."""

  device: str | None = None
  """Compute device (e.g. ``cpu``, ``cuda:0``).  Auto-detected when omitted."""

  viewer: Literal["auto", "native", "viser"] = "auto"
  """Viewer backend to use."""


def main() -> None:
  cfg = tyro.cli(ReplayConfig, config=mjlab.TYRO_FLAGS)

  if cfg.task not in list_tasks():
    print(f"[ERROR] Unknown task: {cfg.task!r}")
    print(f"Available tasks: {list_tasks()}")
    sys.exit(1)

  if cfg.motion_file is None and cfg.registry_name is None:
    print(
      "[ERROR] Either --motion-file or --registry-name must be provided.\n"
      "  --motion-file /path/to/motion.npz\n"
      "  --registry-name my-org/motions/motion-name"
    )
    sys.exit(1)

  # Resolve W&B registry artifact to a local path when no local file given.
  motion_file = cfg.motion_file
  if motion_file is None:
    registry_name = cfg.registry_name
    assert registry_name is not None
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_file = str(Path(artifact.download()) / "motion.npz")
    print(f"[INFO]: Downloaded motion artifact to: {motion_file}")

  play_cfg = PlayConfig(
    agent="zero",
    motion_file=motion_file,
    num_envs=cfg.num_envs,
    device=cfg.device,
    viewer=cfg.viewer,
    no_terminations=True,
  )
  run_play(cfg.task, play_cfg)


if __name__ == "__main__":
  main()
