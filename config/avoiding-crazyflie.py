import numpy as np
from diffuser.utils import watch

USE_DEPTH = False
DEPTH_NEAR = 0.1   # metres
DEPTH_FAR = 10.0    # metres

BOXES = [
]

CYLINDERS = [
    # (2, 0.7),
    (2.5, 0.5),
    # (2, 0.0),
    (2.5, -0.5),
    # (2, -0.8),
    (0, 0.8),
    # (0.5, 0.4),
    (0, 0.0),
    # (0.5, -0.4),
    (0, -0.9),
]

KEEPOUT_ZONES = [
    (-2.0, 0.0, 0.5),
]

CORRIDOR_HALFSPACES = [
    # # upper diagonal: y <= -0.1*x + 0.75  (ceiling slopes down toward the goal end)
    # [np.array([0.0, 0.75]), np.array([4.0, -0.15]), 'below'],

    # lower diagonal: y >= 0.1*x - 0.75   (floor slopes up toward the goal end)
    # [np.array([0.0, -0.75]), np.array([4.0, 0.15]), 'above'],

    # # # both
    # # upper diagonal: y <= -0.1*x + 0.75  (ceiling slopes down toward the goal end)
    # [np.array([0.0, 0.75]), np.array([4.0, 0.15]), 'below'],
    # # lower diagonal: y >= 0.1*x - 0.75   (floor slopes up toward the goal end)
    # [np.array([0.0, -0.75]), np.array([4.0, -0.15]), 'above'],
]

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
    ('model', 'D'),
    ('encoder_type', 'E'),
    ('image_cond_dim', 'L'),
    ('use_depth', 'DEPTH'),
]

logbase = 'isaac/logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.ImageCondTransformer1DModel', #ImageCondTransformer1DModel, ImageCondUNet1DTemporalCondModel
        'diffusion': 'models.GaussianDiffusion',
        'encoder_type': 'raw_pixels',   # "vit", "vitp", or "cnn" or "raw_pixels"
        'use_depth': USE_DEPTH,   # RGBD switch: set True only if the zarr dataset was collected with quadcopter.py --use_depth
        'horizon': 8,
        'n_obs_steps': 2,
        'image_cond_dim': 384,   # 96*96*3 for raw pixels (96*96*4 if use_depth=True) 27648 for vit, 512 for vitp
        'n_diffusion_steps': 20,
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 10,            
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        'test_ret': 0.9,
                
        # ── ViT knobs (these must live here so read_config sets them on args) ──
        'vit_img_size': 96,
        'vit_patch_size': 8,
        'vit_width': 512,
        'vit_depth': 6,
        'vit_heads': 8,
        'vit_mlp_ratio': 4.0,
        'vit_dropout': 0.1,
        'vit_attn_dropout': 0.0,
        
        # Transformer denoiser knobs (new)
        'd_model': 256,
        'n_heads': 8,
        'depth': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,

        ## dataset
        'loader': 'datasets.CrazyflieImageDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 1500,
        'include_returns': False,
        'returns_scale': 400,   # Determined using rewards from the dataset
        'discount': 0.99,


        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,            # Overwritten
    },

    'plan': {
        'policy': 'sampling.Policy',
        'max_episode_length': 1500,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 5,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),

        ## diffusion model
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 8,
        'n_obs_steps': 2,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,

        ## loading
        'diffusion_loadpath': 'f:diffusiondt/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },
}