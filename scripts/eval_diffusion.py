from pathlib import Path
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch

import diffuser.utils as utils

RUN_DIR = "isaac/logs/avoiding-crazyflie/diffusion/H8_K20_Dmodels.ImagePoseCondUNet1DTemporalCondModel_Evitp_L384/7"

print(f"\n[INFO] Loading run dir: {RUN_DIR}")
device = torch.device("cuda:0")

seedmodel = int(Path(RUN_DIR).name)
diff_exp = utils.load_diffusion(RUN_DIR, epoch="best", device=str(device))
dataset = diff_exp.dataset
diffusion = diff_exp.diffusion.to(device)
diffusion.eval()
horizon = int(getattr(diffusion, "horizon", 16))
action_dim = int(getattr(diffusion, "action_dim", 3))
To = int(getattr(dataset, "n_obs_steps", 2))
img_size = int(getattr(dataset, "img_size", 96))

obs_rgb = torch.rand(1, To, 3, img_size, img_size, device=device)
goal_rel = torch.zeros(1, 3, device=device)
cond = {"obs_rgb": obs_rgb, "goal_rel": goal_rel}

with torch.no_grad():
    x, _ = diffusion.conditional_sample(cond, horizon=horizon, projector=None)  # (1,H,D)

pred_actions = dataset.action_normalizer.unnormalize(x[0, :, :action_dim].detach().cpu().numpy())

print("action horizon:")
print(pred_actions)