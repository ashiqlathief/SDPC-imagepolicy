# DPCC — Diffusion Policy with Constraint Control

Trajectory-level diffusion policy for robot control with hard constraint enforcement via projection. Tested on two platforms:

- **Crazyflie quadrotor** in NVIDIA Isaac Lab (image-conditioned, Transformer or UNet denoiser)
- **D3IL robot arm** in MuJoCo (state-based, UNet denoiser)
## 🚧 Status: Work in Progress

This project is being developed as part of a master's thesis at Paderborn University. Expect frequent changes. Not all features are fully implemented or documented yet.
---

## Repository structure

```
dpcc-thesis/
├── diffuser/
│   ├── datasets/    # Dataset loaders
│   ├── models/      # UNet, Transformer, ViT encoder, GaussianDiffusion
│   ├── sampling/    # Policy rollout, DPCC projection
│   └── utils/       # Training loop, logging, normalisation
├── isaac/
│   ├── dataset/     # Crazyflie data collection scripts
│   └── scripts/     # Isaac Lab environment
├── scripts/         # Train and eval entry points
├── config/          # Experiment configs
├── plots/           # Generated figures
├── requirements.txt     # Packages for D3IL / base environment
└── requirements_dpcc_isaac.txt  # Extra packages for Isaac Lab environment
```

---

## Installation

### Experiment 1 — Crazyflie quadrotor (Isaac Lab)

Isaac Lab and IsaacSim must be installed first via NVIDIA's installer:
```
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation
```

Then, inside the Isaac Lab conda environment:
```bash
conda activate env_isaaclab

# Install extra packages on top of Isaac Lab
pip install -r requirements_dpcc_isaac.txt

# Install the local diffuser package
pip install -e .
```

**Packages in `requirements_dpcc_isaac.txt`:**
```
diffusers, einops, huggingface-hub, matplotlib,
numpy, PyYAML, regex, requests, scikit-learn,
scipy, tqdm, transformers, minari
```

---

## Training

### Crazyflie (image-conditioned, Transformer or UNet)
```bash
conda activate env_isaaclab
python scripts/traintransformer.py
```
Config: `config/avoiding-crazyflie.py`
Checkpoints saved to: `isaac/logs/avoiding-crazyflie/`

## Evaluation

### Crazyflie with constraints (DPCC)
```bash
conda activate env_isaaclab
python scripts/evalcopy.py \
  --run_dir isaac/scripts/logs/avoiding-crazyflie/diffusion/<exp_name>/<seed>
```

### Crazyflie without constraints (baseline)
```bash
python scripts/evalquad.py \
  --run_dir isaac/scripts/logs/avoiding-crazyflie/diffusion/<exp_name>/<seed>
```

---

## How DPCC works

Standard diffusion policies sample trajectories by iteratively denoising random noise. DPCC adds a **projection step** inside the denoising loop:

1. Diffusion model proposes a trajectory at each noise level
2. `Projector` (`diffuser/sampling/projection.py`) solves a constrained optimisation to project the trajectory onto the feasible set (obstacle avoidance, dynamics constraints)
3. The projected trajectory is passed to the next denoising step

This enforces hard constraints at inference time without retraining.

---

## Models

| Model | File | Used for |
|---|---|---|
| `GaussianDiffusion` | `diffuser/models/diffusion.py` | Diffusion process (both experiments) |
| `UNet1DTemporalCondModel` | `diffuser/models/unet1d_temporal_cond.py` | D3IL, Crazyflie UNet |
| `ImageCondTransformer1DModel` | `diffuser/models/image_cond_transformer.py` | Crazyflie Transformer |
| `ImageCondUNet1DTemporalCondModel` | `diffuser/models/image_cond_unet.py` | Crazyflie UNet (image) |
| `ViTObsEncoder` | `diffuser/models/vit_obs_encoder.py` | ViT image encoder |
