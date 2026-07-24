import os
import numpy as np
import torch
import random
import diffuser.utils as utils

exp = 'avoiding-crazyflie'

seeds = [7,8,9,10] 

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    # disables auto-tuner (non-deterministic)
    print(f"[seed] All RNG sources set to {seed}")

class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

for seed in seeds:
    args = Parser().parse_args(experiment='diffusion', seed=seed)

    set_seed(args.seed)

    print(f"\n================= SEED {seed} =================")
    print(f"Dataset: {args.dataset}")
    print(f"Save path: {args.savepath}")
    print(f"Loader: {args.loader}")
    print(f"Model: {args.model}")
    print(f"Diffusion: {args.diffusion}")
    print("===============================================")

    # Get dataset
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=args.dataset,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        include_returns=args.include_returns,
        returns_scale=args.max_path_length,               # Because the reward is <= 1 in each timestep
        discount=args.discount,
        stride=getattr(args, 'stride', 1),
        dt=getattr(args, 'dt', 0.005),
        use_depth=getattr(args, 'use_depth', False),
    )

    dataset = dataset_config() # <class 'diffuser.datasets.sequence.SequenceDataset'>
    print(f"Dataset loaded.")
    print(f"  Observation dim: {dataset.observation_dim}")
    print(f"  Action dim:      {dataset.action_dim}")
    print(f"  Goal dim:        {dataset.goal_dim}")
    print(f"  Dataset length:  {len(dataset)} (if applicable)")

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    goal_dim = dataset.goal_dim

    print(f"Total episodes parsed: {len(dataset.episodes)}")
    print(f"Total training indices: {len(dataset.indices)}")
    print(f"Training samples: {int(0.9 * len(dataset.indices))}")
    print(f"Min/max action values pre-norm: {dataset.action_normalizer.mins}, {dataset.action_normalizer.maxs}")

    # -----------------------------------------------------------------------------#
    # ------------------------------ model ----------------------------------------#
    # -----------------------------------------------------------------------------#
    if args.model == 'models.ImageCondTransformer1DModel':
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            horizon=args.horizon,
            action_dim=action_dim,
            image_cond_dim=args.image_cond_dim,
            encoder_type=args.encoder_type,
            vit_img_size=args.vit_img_size,
            vit_patch_size=args.vit_patch_size,
            vit_width=args.vit_width,
            vit_depth=args.vit_depth,
            vit_heads=args.vit_heads,
            vit_mlp_ratio=args.vit_mlp_ratio,
            vit_dropout=args.vit_dropout,
            vit_attn_dropout=args.vit_attn_dropout,
            # transformer denoiser knobs
            d_model=args.d_model,
            n_heads=args.n_heads,
            depth=args.depth,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            returns_condition=args.returns_condition,
            condition_dropout=args.condition_dropout,
            device=args.device,
            use_depth=getattr(args, 'use_depth', False),
        )
        print("Using Transformer denoiser model.")
    else:
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            # ── architecture ──────────────────────────────
            horizon=args.horizon,
            action_dim=action_dim,
            image_cond_dim=args.image_cond_dim,
            dim_mults=args.dim_mults,
            returns_condition=args.returns_condition,
            dim=args.dim, #Base hidden dimension for the UNet’s first layer
            condition_dropout=args.condition_dropout,
            encoder_type=args.encoder_type,   # "cnn" or "vit"
            # NEW: ViT settings (used only if encoder_type == "vit")
            vit_img_size=args.vit_img_size,
            vit_patch_size=args.vit_patch_size,
            vit_width=args.vit_width,
            vit_depth=args.vit_depth,
            vit_heads=args.vit_heads,
            vit_mlp_ratio=args.vit_mlp_ratio,
            vit_dropout=args.vit_dropout,
            vit_attn_dropout=args.vit_attn_dropout,
            device=args.device,
            use_depth=getattr(args, 'use_depth', False),
        )
        print("Using image-conditioned Unet model")
    # -----------------------------------------------------------------------------#
    # ------------------------------ diffusion ------------------------------------#
    # -----------------------------------------------------------------------------#
    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_discount=args.loss_discount,
        returns_condition=args.returns_condition,
        condition_guidance_w=args.condition_guidance_w,
        device=args.device,
    )
    # -----------------------------------------------------------------------------#
    # ------------------------------ trainer --------------------------------------#
    # -----------------------------------------------------------------------------#
    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
        train_test_split=args.train_test_split,
        ema_decay=args.ema_decay,
        n_train_steps=args.n_train_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        results_folder=args.savepath,
    )
    print("\nCreating model, diffusion, and trainer objects...")
    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#
    # Step 1
    model = model_config() #creates the neural network.

    # Guard against a mismatched RGB/RGBD config silently corrupting training:
    # dataset.use_depth controls what's loaded from zarr, model.use_depth controls
    # the encoder's in_chans — both must agree with the requested args.use_depth.
    use_depth = getattr(args, 'use_depth', False)
    assert dataset.use_depth == use_depth, (
        f"Dataset use_depth={dataset.use_depth} != config use_depth={use_depth}"
    )
    assert model.use_depth == use_depth and model.in_chans == (4 if use_depth else 3), (
        f"Model use_depth={model.use_depth} (in_chans={model.in_chans}) != config use_depth={use_depth}"
    )
    # Step 2 — wrap it in the diffusion math
    diffusion = diffusion_config(model) #wraps the model in the diffusion process
    trainer = trainer_config(diffusion, dataset) #sets up the training routine
    trainer.train() #starts the actual training loop.