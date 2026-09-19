# SDPC (Safe Diffusion Policy with Constraint)

## How SDPC works

Standard diffusion policies sample trajectories by iteratively denoising random noise. SDPC adds a **projection step** inside the denoising loop:

1. Diffusion model proposes a trajectory at each noise level, conditioned on the image history (and, for pose-conditioned models, the relative goal vector)
2. `Projector` (`diffuser/sampling/projection.py`) solves a constrained optimisation (SLSQP) to project the trajectory onto the feasible set — obstacle avoidance (against ground-truth or depth-perceived obstacle positions), altitude bounds, corridor bounds
3. The projected trajectory is passed to the next denoising step

Hard constraints are enforced at inference time without retraining.

## Models

| Model | File | Description |
|---|---|---|
| `GaussianDiffusion` | `diffuser/models/diffusion.py` | Diffusion process wrapper |
| `ImageCondTransformer1DModel` / `ImagePoseCondTransformer1DModel` | `diffuser/models/image_cond_transformer.py` | DiT-style transformer denoiser with image conditioning, optionally goal/pose-conditioned |
| `ImageCondUNet1DTemporalCondModel` / `ImagePoseCondUNet1DTemporalCondModel` | `diffuser/models/image_cond_unet.py` | UNet denoiser with image conditioning, optionally goal/pose-conditioned |
| `UNet1DTemporalCondModel` | `diffuser/models/unet1d_temporal_cond.py` | State-conditioned UNet denoiser |
| `ViTObsEncoder` | `diffuser/models/vit_obs_encoder.py` | ViT image encoder for observation conditioning |

## Installation

### Steps

**1. Install Isaac Sim and Isaac Lab** following NVIDIA's official guide (this creates the base conda environment):
```
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation
```

**2. Pull the robot asset submodule** (NTNU-ARL's `lmf2` quadcopter USD [ntnu-arl/robot_model](https://github.com/ntnu-arl/robot_model)):
```bash
git submodule update --init --recursive
```
If you cloned this repo without `--recurse-submodules`, run the command above once afterward; `robot_model/` stays empty (and every script importing `arl_robot__cfg.py` fails at startup) until you do.

**3. Activate the Isaac Lab conda environment and install extra packages:**
```bash
conda activate env_isaaclab

# Install project dependencies (includes warp 1.11.1, numpy 1.26.0, etc.)
pip install -r requirement.txt

# Make the project importable
export PYTHONPATH=$PWD:$PYTHONPATH
```

---

## Data Collection

`isaac/scripts/quadcopter.py` runs the quadcopter in IsaacLab using a cascaded PID controller, navigating to random targets while recording FPV camera images and state data. Episodes are saved as `.pkl`/`.npz` files and, via `isaac/scripts/zarr_episode_writer.py`, an appendable Zarr store under `isaac/dataset/avoiding_crazyflie/data/`.

### Running

```bash
# Basic (1 environment, RGB only)
python isaac/scripts/quadcopter.py

# Multiple parallel environments
python isaac/scripts/quadcopter.py --num_envs 4

# With depth camera (saves RGB + distance_to_camera)
python isaac/scripts/quadcopter.py --use_depth

# Combined
python isaac/scripts/quadcopter.py --num_envs 4 --use_depth
```

Any standard IsaacLab/AppLauncher flags (e.g. `--device cuda:0`) also work.

### Keyboard controls

| Key | Action |
|-----|--------|
| `SPACE` | Reset drone to start pose, clear buffered data |
| `ENTER` | Manually save all buffered episodes to disk |
| `R` | Toggle recording on/off |
| `C` | Clear buffered data (no files deleted) |

### Auto-save behaviour

The script saves and resets automatically whenever the drone holds within **5 cm** of its target for at least 1 simulation step. A new random target is then sampled and the episode restarts. No manual intervention is needed for continuous data collection.

### Output files

For each environment `N`, the script writes:

| File | Contents |
|------|----------|
| `env_NNN_XXXXX.pkl` | State list metadata |
| `env_NNN_XXXXX_images.npz` | Stacked RGB (and optionally depth) arrays |
| `env_NNN.zarr/` | Appendable Zarr store used for diffusion policy training |

---

## Training

```bash
conda activate env_isaaclab
python scripts/train.py
```

### Pretrained checkpoint

A trained checkpoint is published on Hugging Face at [`ashiqali98/SDPC_diffusionmodel`](https://huggingface.co/ashiqali98/SDPC_diffusionmodel). It's the flat contents of one `<exp_name>/7/` run directory: `dataset_config.pkl`, `model_config.pkl`, `diffusion_config.pkl`, `trainer_config.pkl`, `losses.pkl`, `state_best.pt` and exactly what `diffuser.utils.load_diffusion()` expects to find at `RUN_DIR`.

```bash
# Make sure the hf CLI is installed
curl -LsSf https://hf.co/cli/install.sh | bash

# Download straight into the RUN_DIR every eval script defaults to
hf download ashiqali98/SDPC_diffusionmodel \
  --local-dir isaac/logs/avoiding-crazyflie/diffusion/H8_K20_Dmodels.ImagePoseCondUNet1DTemporalCondModel_Evitp_L384/7
```

Downloading anywhere else works too and just point each script's `RUN_DIR` constant at wherever you put it. `git clone` (with [`git-xet`](https://hf.co/docs/hub/git-xet) installed) is the alternative if you'd rather not use the `hf` CLI:

```bash
curl -sSfL https://hf.co/git-xet/install.sh | sh
git clone https://huggingface.co/ashiqali98/SDPC_diffusionmodel
```

## Evaluation

Evaluation is split across several scripts depending on target (simulation vs. real hardware) and how thoroughly the loop actually runs (offline smoke test vs. full episode rollout).

### Offline smoke tests (no Isaac Sim, no env)

These load a trained checkpoint and run a handful of denoising steps against synthetic/random input, to sanity-check that a checkpoint loads and the projection math runs, without booting a simulator:

```bash
python scripts/eval_diffusion.py         # one unconditioned denoise on random noise
python scripts/eval_diffusionmpc.py      # SDPC-R/C/T + projection against a hardcoded obstacle list
python scripts/eval_diffusionmpc_depth.py  # same, obstacles detected from a synthetic depth frame
```

Each hardcodes `RUN_DIR` at the top of the file and edit it to point at your trained checkpoint directory (e.g. `isaac/logs/avoiding-crazyflie/diffusion/<exp_name>/<seed>`).

### Standalone alternative eval scripts (need Isaac Sim)

`scripts/diffusion_drone.py`, `scripts/diffusion_dronempc.py`, and `scripts/diffusion_dronempcdepth.py` are self-contained Isaac Lab scripts that build the ARL lmf2 environment inline (via `isaac/scripts/env_cfg.py`/`arl_robot__cfg.py`) rather than going through the `crazyflie_envpos.py` Gym env used by `scripts/eval_craziefliepos.py`. They progressively add SLSQP-projector MPC (`diffusion_dronempc.py`) and then depth-perceived obstacles (`diffusion_dronempcdepth.py`) on top of a plain policy rollout (`diffusion_drone.py`). Each hardcodes `RUN_DIR` at the top of the file, same as the smoke tests above.

```bash
conda activate env_isaaclab

python scripts/diffusion_drone.py        # plain policy rollout, GUI window (HEADLESS=False)
python scripts/diffusion_dronempc.py      # + SLSQP-projector MPC, GUI window (HEADLESS=False)
python scripts/diffusion_dronempcdepth.py # + depth-perceived obstacles, headless (HEADLESS=True)
```

Each file's own `HEADLESS` constant near the top overrides `--headless` — flip it there, not via the CLI, to switch between windowed and headless.

### Full simulation evaluation

```bash
conda activate env_isaaclab
python scripts/eval_craziefliepos.py
```

Runs a full episode rollout per variant in IsaacLab (`isaac/scripts/crazyflie_envpos.py`), for every entry in the `VARIANTS` list (`sdpc-r`, `sdpc-c`, `sdpc-t`, `diffuser`, repeated across random spawn/target pairs when `RANDOMIZE_SPAWN_TARGET=True`). Saves per-episode `.npz` trajectories, XY/Z plots, and a metrics summary (`metrics_logger.MetricsLogger`) under `<run_dir>/trajectories*/`, `<run_dir>/plots*/`, and `<run_dir>/results/`.

There is no CLI for this script and configuration is a block of module-level constants at the top of the file:

| Variable | Description |
|---|---|
| `RUN_DIR`, `SEEDS` | Checkpoint directory and which seed subfolder(s) to evaluate |
| `MAX_STEPS` | Max control steps per episode before giving up |
| `VARIANTS` / `VARIANT_CFG` | Which projection variants to run and how many episodes of each |
| `OBSTACLE_SOURCE` | `"ground_truth"` (reads `env.get_cylinder_positions()`) or `"depth"` (obstacles perceived from the onboard depth camera via `depth_obstacle_estimator.detect_obstacles_umap()`) |
| `RANDOMIZE_SPAWN_TARGET`, `SPAWN_X_RANGE`/`SPAWN_Y_RANGE`, `TARGET_X_RANGE`/`TARGET_Y_RANGE` | Per-episode randomised start/goal sampling |
| `PROJ_TIGHTEN`, `FLIGHT_Z_MIN`/`FLIGHT_Z_MAX` | Projector safety margin and altitude bounds |

### Real-hardware evaluation (ROS2 / MAVROS)

```bash
conda activate env_isaaclab
source /opt/ros/humble/setup.bash
python scripts/eval_crazieflieros2.py
```

A `rclpy` node (`Ros2HardwareRunner`) that runs the *same* trained diffusion policy + projector on a physical Crazyflie: subscribes to MAVROS pose (`/mavros/local_position/pose`) and a RealSense color/depth stream (`/camera/camera/...`), publishes position setpoints to `/mpc/set_pose`, and (if the chosen variant uses projection) perceives obstacles from the live depth stream the same way the sim path does. Configuration (`RUN_DIR`, `VARIANT`, topic names, camera intrinsics, control rate) is set via class attributes at the top of `Ros2HardwareRunner` — only a single variant is flown per run (no sweep, unlike the sim script).

### Depth-obstacle perception tools

`scripts/depth_obstacle_estimator.py` implements the obstacle detector shared by both eval paths above: a U-disparity-map + contour method (`detect_obstacles_umap()`) that turns a depth frame into a list of `(x, y, radius)` world-frame obstacle estimates. Companion tools for developing/debugging it, none of which need a live drone:

| Script | What it does |
|--------|-------------|
| `depth_camera_live_test.py` | Runs the detector against a live depth feed |
| `diag_umap_synthetic.py` | Pure numpy/opencv unit test of the detector's size/radius math against a synthetic depth frame |
| `diag_umap_visualize.py` | Renders detected bounding boxes onto real depth+color frames pulled from a rosbag |

and uses the obstacle perception uses this paper "Robust Vision-based Obstacle Avoidance for Micro Aerial Vehicles in Dynamic Environments" as reference.
