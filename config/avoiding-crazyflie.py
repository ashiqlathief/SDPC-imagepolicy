from diffuser.utils import watch

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
]

logbase = 'isaac/logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.ImageCondTransformer1DModel', #ImageCondTransformer1DModel, ImageCondUNet1DTemporalCondModel
        'diffusion': 'models.GaussianDiffusion',
        'encoder_type': 'raw_pixels',   # "vit", "vitp", or "cnn" or "raw_pixels"
        'horizon': 16,
        'n_obs_steps': 2,
        'image_cond_dim': 27648,   # 96*96*3 for raw pixels 27648
        'n_diffusion_steps': 20,
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 10,            
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'dynamic_loss': False,
        'hidden_dim': 256,
        'attention': False,
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
        'horizon': 16,
        'n_obs_steps': 2,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },
}