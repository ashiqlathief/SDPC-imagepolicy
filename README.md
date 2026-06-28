# SDPC Safe Diffusion Policy with Constraint

Trajectory-level diffusion policy for robot control with hard constraint enforcement via projection. Tested on:

- **Crazyflie quadrotor** in NVIDIA Isaac Lab (image-conditioned, Transformer or UNet denoiser)

## 🚧 Status: Work in Progress

This project is being developed as part of a master's thesis at Paderborn University. Expect frequent changes. Not all features are fully implemented or documented yet.

---

## Repository structure

```
dpcc-thesis/
├── config/
│   └── avoiding-crazyflie.py     # Experiment config (obstacles, halfspaces, model hyperparams)
├── diffuser/
│   ├── datasets/
│   │   ├── crazyflie.py          # CrazyflieImageDataset
│   │   ├── normalization.py
│   │   └── sequence.py
│   ├── models/
│   │   ├── diffusion.py          # GaussianDiffusion
│   │   ├── image_cond_transformer.py  # ImageCondTransformer1DModel
│   │   ├── image_cond_unet.py    # ImageCondUNet1DTemporalCondModel
│   │   ├── unet1d_temporal_cond.py
│   │   └── vit_obs_encoder.py    # ViTObsEncoder
│   ├── sampling/
│   │   ├── policies.py           # Candidate sampling, selection strategies
│   │   └── projection.py        # SLSQP-based Projector
│   └── utils/
│       ├── constraints_helpers.py
│       ├── training.py
│       └── ...
├── isaac/
│   ├── dataset/
│   │   └── avoiding_crazyflie/   # Recorded Crazyflie demonstration data
│   │   ├── avoiding_dataset.py
│   │   └── base_dataset.py
│   ├── logs/
│   │   └── avoiding-crazyflie/   # Training run outputs and eval results
│   └── scripts/
│       ├── crazyflie_env.py      # CrazyflieEnv (Isaac Lab gym env)
│       ├── crazyflie_env_cfg.py  # CrazyflieEnvCfg dataclass
│       ├── plotall.py            # Plot all variant trajectories on one XY figure
│       ├── table.py              # Generate LaTeX results table from saved npz files
│       └── view_traj.py          # Trajectory plot + metrics for one trajectories* folder
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── eval_crazieflie.py        # Evaluation across all projection variants
│   ├── make_traj_gif.py          # Animate a saved .npz trajectory as a GIF
│   └── metrics_logger.py         # Per-episode metrics and summary saving
├── requirements.txt
└── requirements_sdpc_isaac.txt
```

---

## Installation

### Steps

**1. Install Isaac Sim and Isaac Lab** following NVIDIA's official guide (this creates the base conda environment):
```
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation
```

**2. Activate the Isaac Lab conda environment and install extra packages:**
```bash
conda activate env_isaaclab

# Install project dependencies (includes warp 1.11.1, numpy 1.26.0, etc.)
pip install -r requirement.txt

# Make the project importable
export PYTHONPATH=$PWD:$PYTHONPATH
```
---

## Training

```bash
conda activate env_isaaclab
python scripts/train.py
```

Config: `config/avoiding-crazyflie.py`  
Checkpoints saved to: `isaac/logs/avoiding-crazyflie/`

### Configuration — `config/avoiding-crazyflie.py`

Everything about the environment layout and training hyperparameters lives here.

#### Obstacle layout

| Variable | Description |
|---|---|
| `CYLINDERS` | List of `(x, y)` positions for vertical cylinder obstacles in the corridor |
| `BOXES` | List of `(x, y)` positions for box obstacles (currently empty) |
| `SPHERES` | List of `(x, y, z)` rest positions for floating sphere obstacles (commented out by default) |
| `SPHERE_RADIUS` | Physical radius of each sphere in metres; effective exclusion zone = `SPHERE_RADIUS + drone_radius` |

#### Corridor halfspaces

`CORRIDOR_HALFSPACES` defines diagonal boundary constraints in the XY plane as line segments with a side (`'above'` or `'below'`). These are only enforced at eval time when `--use_halfspaces` is passed.

```python
# format: [p1, p2, side]  where p1/p2 are [x, y] endpoints of the boundary line
[np.array([0.0, 0.75]), np.array([4.0, 0.15]), 'below'],   # upper diagonal
[np.array([0.0, -0.75]), np.array([4.0, -0.15]), 'above'],  # lower diagonal
```

#### Depth sensing

| Variable | Default | Description |
|---|---|---|
| `USE_DEPTH` | `False` | Switch to RGBD input; must match how the dataset was collected |
| `DEPTH_NEAR` | `0.1` m | Near clip for depth normalisation |
| `DEPTH_FAR` | `10.0` m | Far clip for depth normalisation |

#### Training hyperparameters (`base['diffusion']`)

| Key | Default | Description |
|---|---|---|
| `model` | `ImageCondUNet1DTemporalCondModel` | Denoiser architecture — also supports `ImageCondTransformer1DModel` |
| `encoder_type` | `vitp` | Image encoder: `vit`, `vitp`, `cnn`, or `raw_pixels` |
| `horizon` | `16` | Number of action steps predicted per diffusion sample |
| `n_obs_steps` | `2` | Number of past observations fed as conditioning |
| `n_diffusion_steps` | `20` | Denoising steps at inference |
| `image_cond_dim` | `512` | Latent dim of image encoder output (`27648` for raw pixels at 96×96×3, `512` for ViT-P) |
| `stride` | `2` | Frame subsampling factor — how many raw sim frames between consecutive action steps |
| `dt` | `0.005` s | Raw sim physics timestep at which data was collected |
| `batch_size` | `8` | Training batch size |
| `learning_rate` | `1e-4` | Adam learning rate |
| `n_train_steps` | `1e5` | Total gradient steps |

> **`stride` and `dt` together define the effective control rate:**
> `control_dt = stride × dt`.
> With the defaults (`stride=2`, `dt=0.005`), the model acts every **10 ms**.
> The experiment folder name encodes these as `DT<stride>` and `DT<dt>` — e.g. `DT2` and `DT0.005`.
> If you change `stride` or `dt`, re-collect data at the matching rate or the actions will be scaled incorrectly.

---

## Evaluation

### Run all projection variants

```bash
conda activate env_isaaclab
python scripts/eval_crazieflie.py --run_dir isaac/logs/avoiding-crazyflie/diffusion/<exp_name>/<seed>
```

Saves per-episode `.npz` trajectories, XY plots, and a summary table under the `--run_dir`.

#### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--run_dir` | *(required)* | Path to a trained experiment directory |
| `--max_steps` | `1500` | Maximum environment steps per episode |
| `--action_scale` | `5.0` | Scalar multiplier applied to unnormalised actions |
| `--episodes` | `1` | Number of episodes to roll per variant |
| `--dynamic_obstacles` | off | Enable sinusoidal cylinder motion. Pass with no values for all cylinders on y-axis, or `idx:axis` tokens e.g. `--dynamic_obstacles 0:y 2:x 4:xy` |
| `--floating_spheres` | off | Enable floating 3-D sphere obstacles |
| `--dt` | auto | Override the control timestep (read from dataset metadata by default; see note below) |
| `--use_halfspaces` | off | Enforce corridor halfspace constraints from `CORRIDOR_HALFSPACES` in config |
| `--record_video` | off | Save `.mp4` video for each variant |
| `--camera` | `spectator` | Camera view(s) to record: `spectator`, `chase`, `fpv` (pass multiple) |
| `--record_variants` | all | Subset of variant names to record when `--record_video` is set |
| `--video_fps` | `20` | Playback fps for saved videos |


#### Hardcoded constants (edit in `scripts/eval_crazieflie.py` → `main()`)

| Variable | Value | Description |
|---|---|---|
| `device` | `cuda:0` | Torch device |
| `drone_radius` | `0.08` m | Minkowski expansion for obstacle constraints and env collision check |
| `obs_amplitude` | `0.35` m | Oscillation amplitude for dynamic cylinders |
| `obs_frequency` | `0.25` Hz | Oscillation frequency for dynamic cylinders |
| `sphere_amplitude` | `0.20` m | Per-axis oscillation amplitude for floating spheres |
| `sphere_frequency` | `0.20` Hz | Base oscillation frequency for floating spheres |

#### Projection variants

Variants are listed in `projection_variants` at the top of `scripts/eval_crazieflie.py`. Each maps to a config (num candidates, selection strategy, projection mode, tighten amount) via `variant_cfg()`. To run a subset, comment out unwanted entries in that list.

### Visualisation tools

```bash
# Plot all variants from a trajectories* folder
python isaac/scripts/plotall.py <path/to/trajectories*>

# Interactive trajectory plot + metrics table
python isaac/scripts/view_traj.py <path/to/trajectories*>

# Animate a single .npz trajectory as a GIF
python scripts/make_traj_gif.py <path/to/traj_*.npz>

# Generate LaTeX results table
python isaac/scripts/table.py <path/to/trajectories*>
```

---

## How SDPC works

Standard diffusion policies sample trajectories by iteratively denoising random noise. SDPC adds a **projection step** inside the denoising loop:

1. Diffusion model proposes a trajectory at each noise level
2. `Projector` (`diffuser/sampling/projection.py`) solves a constrained optimisation (SLSQP) to project the trajectory onto the feasible set — obstacle avoidance, corridor bounds, dynamics
3. The projected trajectory is passed to the next denoising step

Hard constraints are enforced at inference time without retraining.

---

## Models

| Model | File | Description |
|---|---|---|
| `GaussianDiffusion` | `diffuser/models/diffusion.py` | Diffusion process wrapper |
| `ImageCondTransformer1DModel` | `diffuser/models/image_cond_transformer.py` | Transformer denoiser with image conditioning |
| `ImageCondUNet1DTemporalCondModel` | `diffuser/models/image_cond_unet.py` | UNet denoiser with image conditioning |
| `UNet1DTemporalCondModel` | `diffuser/models/unet1d_temporal_cond.py` | State-conditioned UNet denoiser |
| `ViTObsEncoder` | `diffuser/models/vit_obs_encoder.py` | ViT image encoder for observation conditioning |
