import os, pickle
import numpy as np
import matplotlib.pyplot as plt

RUN_DIR = "isaac/scripts/logs/avoiding-crazyflie/diffusion/H16_K20_Dmodels.GaussianDiffusion/9"

loss_path = os.path.join(RUN_DIR, "losses.pkl")
trainer_cfg_path = os.path.join(RUN_DIR, "trainer_config.pkl")

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

losses = load_pkl(loss_path)
print("losses.pkl type:", type(losses))
print("keys:", list(losses.keys()))

# ---- steps_per_epoch from trainer_config if possible ----
steps_per_epoch = 1000
if os.path.exists(trainer_cfg_path):
    trainer_cfg = load_pkl(trainer_cfg_path)
    if isinstance(trainer_cfg, dict) and "n_steps_per_epoch" in trainer_cfg:
        steps_per_epoch = int(trainer_cfg["n_steps_per_epoch"])
    elif hasattr(trainer_cfg, "n_steps_per_epoch"):
        steps_per_epoch = int(getattr(trainer_cfg, "n_steps_per_epoch"))

print("steps_per_epoch:", steps_per_epoch)

def unpack_step_loss(seq):
    """seq can be [[step, loss], ...] or dict-like. Returns (steps, losses)."""
    if seq is None:
        return None, None
    arr = np.asarray(seq, dtype=np.float32)
    if arr.size == 0:
        return None, None

    # Expect shape (N,2). If it’s flat, try to reshape.
    if arr.ndim == 1:
        if arr.shape[0] % 2 != 0:
            raise ValueError(f"Expected even-length flat array, got {arr.shape}")
        arr = arr.reshape(-1, 2)

    if arr.shape[1] != 2:
        raise ValueError(f"Expected shape (N,2), got {arr.shape}")

    steps = arr[:, 0]
    vals  = arr[:, 1]
    return steps, vals

train_steps, train_vals = unpack_step_loss(losses.get("training_losses"))
test_steps,  test_vals  = unpack_step_loss(losses.get("test_losses"))
train_a0_steps, train_a0_vals = unpack_step_loss(losses.get("training_a0_losses"))
test_a0_steps,  test_a0_vals  = unpack_step_loss(losses.get("test_a0_losses"))

def plot_curve(steps, vals, label):
    if steps is None or vals is None:
        return
    epochs = steps / float(steps_per_epoch)
    plt.plot(epochs, vals, label=label)

# ---- Plot 1: diffusion loss ----
plt.figure()
plot_curve(train_steps, train_vals, "train")
plot_curve(test_steps,  test_vals,  "test")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Diffusion loss (train vs test)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ---- Plot 2: a0 loss ----
plt.figure()
plot_curve(train_a0_steps, train_a0_vals, "train a0")
plot_curve(test_a0_steps,  test_a0_vals,  "test a0")
plt.xlabel("epoch")
plt.ylabel("a0 loss")
plt.title("First-action (a0) loss (train vs test)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
