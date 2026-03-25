"""Replay a motion NPZ file on a robot.

This script loads a motion ``.npz`` file and plays it back on the robot
by directly writing root and joint states to the simulation, without using
the full RL task system.

.. code-block:: bash

    # K1 robot (default) - from local file
    uv run python src/mjlab/scripts/replay_npz.py --robot k1 --motion-file /path/to/motion.npz

    # G1 robot - from local file
    uv run python src/mjlab/scripts/replay_npz.py --robot g1 --motion-file /path/to/motion.npz

    # From W&B registry (downloads to mjlab/motion/)
    uv run python src/mjlab/scripts/replay_npz.py --robot k1 --registry-name my-org/motions/k1-walk

    # With viser viewer
    uv run python src/mjlab/scripts/replay_npz.py --robot k1 --motion-file /path/to/motion.npz --viewer viser
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.booster_k1.env_cfgs import (
  booster_k1_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.base import VerbosityLevel
from mjlab.viewer.native.viewer import NativeMujocoViewer
from mjlab.viewer.viser.viewer import ViserPlayViewer

# Path to the mjlab motion cache folder (~/mjlab/motion/)
MJLAB_MOTION_DIR = Path.home() / "mjlab" / "motion"


class _DummyRewardManager:
  """Dummy reward manager that returns empty terms for replay."""

  def get_active_iterable_terms(self, env_idx: int):
    return []


class _DummyCommandManager:
  """Dummy command manager that returns empty active terms for replay."""

  @property
  def active_terms(self):
    return []

  def create_debug_vis_gui(self, server):
    pass


@dataclass(frozen=True)
class ReplayConfig:
  """Configuration for motion replay."""

  motion_file: str | None = None
  """Path to a motion ``.npz`` file. Either --motion-file or --registry-name is required."""

  registry_name: str | None = None
  """W&B registry artifact name (e.g. ``my-org/motions/k1-walk``).
  ``:latest`` is appended automatically when no alias is specified.
  Downloads to ``mjlab/motion/<artifact-name>/motion.npz``."""

  robot: Literal["k1", "g1"] = "k1"
  """Robot type to replay on."""

  speed: float = 1.0
  """Playback speed multiplier. 1.0 = real-time, 2.0 = 2x faster, 0.5 = half speed."""

  device: str | None = None
  """Compute device (e.g. ``cpu``, ``cuda:0``). Auto-detected when omitted."""

  viewer: Literal["auto", "native", "viser"] = "auto"
  """Viewer backend to use."""


def resolve_motion_file(motion_file: str | None, registry_name: str | None) -> str:
  """Resolve motion file path from local file or W&B registry.

  Args:
    motion_file: Path to local motion npz file.
    registry_name: W&B registry artifact name.

  Returns:
    Resolved path to the motion npz file.

  Raises:
    ValueError: If neither or both arguments are provided.
  """
  if motion_file is None and registry_name is None:
    raise ValueError("Either --motion-file or --registry-name must be provided.")
  if motion_file is not None and registry_name is not None:
    raise ValueError("Cannot specify both --motion-file and --registry-name.")

  if motion_file is not None:
    if not os.path.isfile(motion_file):
      raise FileNotFoundError(f"Motion file not found: {motion_file}")
    return motion_file

  # Download from W&B registry
  assert registry_name is not None
  import wandb

  # Append :latest if no alias specified
  if ":" not in registry_name:
    registry_name = registry_name + ":latest"

  # Create download directory: mjlab/motion/<artifact-name>/
  artifact_dir = MJLAB_MOTION_DIR / registry_name.replace(":", "_")
  artifact_dir.mkdir(parents=True, exist_ok=True)
  motion_path = artifact_dir / "motion.npz"

  # Check if already downloaded
  if motion_path.exists():
    print(f"[INFO] Using cached motion file: {motion_path}")
    return str(motion_path)

  # Download from W&B
  api = wandb.Api()
  artifact = api.artifact(registry_name)
  print(f"[INFO] Downloading motion artifact: {registry_name}")
  temp_dir = Path(artifact.download())  # type: ignore[assignment]
  motion_file_path = temp_dir / "motion.npz"

  if not motion_file_path.exists():
    raise FileNotFoundError(f"motion.npz not found in artifact: {registry_name}")

  # Copy to cache directory
  import shutil

  shutil.copy2(motion_file_path, motion_path)  # type: ignore[arg-type]
  print(f"[INFO] Motion saved to: {motion_path}")

  return str(motion_path)


class NpzMotionLoader:
  """Loads motion data from an NPZ file."""

  def __init__(self, motion_file: str, device: str):
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self.body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self.body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self.body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self.body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self.time_step_total = self.joint_pos.shape[0]
    self.current_step = 0
    # Assume 50Hz motion (0.02s per frame) - standard for most motion capture data
    self.dt = 0.02


def run_replay(
  motion_file: str,
  robot_type: Literal["k1", "g1"],
  speed: float,
  device: str,
  viewer_type: Literal["auto", "native", "viser"],
) -> None:
  """Run motion replay with the specified robot and viewer."""
  configure_torch_backends()

  # Load motion data
  motion = NpzMotionLoader(motion_file, device)

  # Set up scene and simulation based on robot type
  if robot_type == "k1":
    env_cfg = booster_k1_flat_tracking_env_cfg()
  else:
    env_cfg = unitree_g1_flat_tracking_env_cfg()

  # Create scene and simulation
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()

  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]

  # Resolve viewer type
  if viewer_type == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = viewer_type

  if resolved_viewer == "native":
    run_native_viewer(env_cfg, robot, scene, sim, motion, speed)
  else:
    run_viser_viewer(env_cfg, robot, scene, sim, motion, speed)


def run_native_viewer(
  env_cfg: ManagerBasedRlEnvCfg,
  robot: Entity,
  scene: Scene,
  sim: Simulation,
  motion: NpzMotionLoader,
  speed: float,
) -> None:
  """Run replay with native MuJoCo viewer."""

  class ZeroPolicy:
    def __call__(self, obs):
      return torch.zeros(0, dtype=torch.float32, device=sim.device)

    def reset(self):
      pass

  # Create a minimal env wrapper for the viewer
  class MinimalEnv:
    def __init__(self, env_cfg: ManagerBasedRlEnvCfg, robot, scene, sim, motion, speed):
      self._env_cfg = env_cfg
      self.robot = robot
      self.scene = scene
      self.sim = sim
      self.motion = motion
      self.speed = speed
      self.num_envs = 1
      self.device = sim.device
      self.step_dt = sim.mj_model.opt.timestep
      self._step_count = 0
      self.reward_manager = _DummyRewardManager()
      self._last_real_time = time.perf_counter()
      self._motion_accumulator = 0.0

    @property
    def cfg(self) -> ManagerBasedRlEnvCfg:
      return self._env_cfg

    @property
    def unwrapped(self) -> MinimalEnv:
      return self

    def get_observations(self) -> Any:
      return torch.zeros(0, device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[Any, ...]:
      self._advance_motion()
      self._step_count += 1
      return (self.get_observations(), torch.zeros(1), torch.zeros(1), None)

    def reset(self) -> Any:
      self.motion.current_step = 0
      self._step_count = 0
      self._last_real_time = time.perf_counter()
      self._motion_accumulator = 0.0
      return self.get_observations()

    def _advance_motion(self):
      # Track real elapsed time to sync with motion playback
      current_real_time = time.perf_counter()
      real_dt = current_real_time - self._last_real_time
      self._last_real_time = current_real_time

      # Motion time advance based on speed (motion is recorded at 50Hz = 0.02s per frame)
      motion_dt = self.motion.dt / self.speed
      self._motion_accumulator += real_dt

      # Only advance motion when enough real time has passed
      if self._motion_accumulator < motion_dt:
        # Not enough time passed, still step simulation but don't advance motion
        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        return

      # Reset accumulator (keep remainder for accurate timing)
      self._motion_accumulator = self._motion_accumulator % motion_dt

      step = self.motion.current_step
      if step >= self.motion.time_step_total:
        step = 0
        self.motion.current_step = 0

      # Get motion data for current step
      root_pos = motion.body_pos_w[step, 0].clone()
      root_quat = motion.body_quat_w[step, 0].clone()
      root_lin_vel = motion.body_lin_vel_w[step, 0].clone()
      root_ang_vel = motion.body_ang_vel_w[step, 0].clone()
      joint_pos = motion.joint_pos[step].clone()
      joint_vel = motion.joint_vel[step].clone()

      # Write root state
      root_state = torch.zeros(13, dtype=torch.float32, device=self.device)
      root_state[0:3] = root_pos
      root_state[3:7] = root_quat
      root_state[7:10] = root_lin_vel
      root_state[10:] = root_ang_vel
      robot.write_root_state_to_sim(root_state.unsqueeze(0))

      # Write joint state
      robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0))

      # Step simulation
      sim.forward()
      scene.update(sim.mj_model.opt.timestep)

      self.motion.current_step += 1

    def close(self) -> None:
      pass

  env = MinimalEnv(env_cfg, robot, scene, sim, motion, speed)

  # Reset to initial pose
  env.reset()

  viewer = NativeMujocoViewer(env, ZeroPolicy(), verbosity=VerbosityLevel.SILENT)
  viewer.run()


def run_viser_viewer(
  env_cfg: ManagerBasedRlEnvCfg,
  robot: Entity,
  scene: Scene,
  sim: Simulation,
  motion: NpzMotionLoader,
  speed: float,
) -> None:
  """Run replay with Viser viewer."""

  class ZeroPolicy:
    def __call__(self, obs):
      return torch.zeros(0, dtype=torch.float32, device=sim.device)

    def reset(self):
      pass

  # Create a minimal env wrapper for the viewer
  class MinimalEnv:
    def __init__(self, env_cfg: ManagerBasedRlEnvCfg, robot, scene, sim, motion, speed):
      self._env_cfg = env_cfg
      self.robot = robot
      self.scene = scene
      self.sim = sim
      self.motion = motion
      self.speed = speed
      self.num_envs = 1
      self.device = sim.device
      self.step_dt = sim.mj_model.opt.timestep
      self._step_count = 0
      self.reward_manager = _DummyRewardManager()
      self.command_manager = _DummyCommandManager()
      self._last_real_time = time.perf_counter()
      self._motion_accumulator = 0.0

    @property
    def cfg(self) -> ManagerBasedRlEnvCfg:
      return self._env_cfg

    @property
    def unwrapped(self) -> MinimalEnv:
      return self

    def get_observations(self) -> Any:
      return torch.zeros(0, device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[Any, ...]:
      self._advance_motion()
      self._step_count += 1
      return (self.get_observations(), torch.zeros(1), torch.zeros(1), None)

    def reset(self) -> Any:
      self.motion.current_step = 0
      self._step_count = 0
      self._last_real_time = time.perf_counter()
      self._motion_accumulator = 0.0
      return self.get_observations()

    def _advance_motion(self):
      # Track real elapsed time to sync with motion playback
      current_real_time = time.perf_counter()
      real_dt = current_real_time - self._last_real_time
      self._last_real_time = current_real_time

      # Motion time advance based on speed (motion is recorded at 50Hz = 0.02s per frame)
      motion_dt = self.motion.dt / self.speed
      self._motion_accumulator += real_dt

      # Only advance motion when enough real time has passed
      if self._motion_accumulator < motion_dt:
        # Not enough time passed, still step simulation but don't advance motion
        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        return

      # Reset accumulator (keep remainder for accurate timing)
      self._motion_accumulator = self._motion_accumulator % motion_dt

      step = self.motion.current_step
      if step >= self.motion.time_step_total:
        step = 0
        self.motion.current_step = 0

      # Get motion data for current step
      root_pos = motion.body_pos_w[step, 0].clone()
      root_quat = motion.body_quat_w[step, 0].clone()
      root_lin_vel = motion.body_lin_vel_w[step, 0].clone()
      root_ang_vel = motion.body_ang_vel_w[step, 0].clone()
      joint_pos = motion.joint_pos[step].clone()
      joint_vel = motion.joint_vel[step].clone()

      # Write root state
      root_state = torch.zeros(13, dtype=torch.float32, device=self.device)
      root_state[0:3] = root_pos
      root_state[3:7] = root_quat
      root_state[7:10] = root_lin_vel
      root_state[10:] = root_ang_vel
      robot.write_root_state_to_sim(root_state.unsqueeze(0))

      # Write joint state
      robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0))

      # Step simulation
      sim.forward()
      scene.update(sim.mj_model.opt.timestep)

      self.motion.current_step += 1

    def close(self) -> None:
      pass

  env = MinimalEnv(env_cfg, robot, scene, sim, motion, speed)

  # Reset to initial pose
  env.reset()

  viewer = ViserPlayViewer(env, ZeroPolicy(), verbosity=VerbosityLevel.SILENT)
  viewer.run()


def main() -> None:
  cfg = tyro.cli(ReplayConfig, config=mjlab.TYRO_FLAGS)

  # Resolve motion file from local path or W&B registry
  motion_file = resolve_motion_file(cfg.motion_file, cfg.registry_name)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  run_replay(
    motion_file=motion_file,
    robot_type=cfg.robot,
    speed=cfg.speed,
    device=device,
    viewer_type=cfg.viewer,
  )


if __name__ == "__main__":
  main()
