import torch
import diffuser.utils as utils

exp = 'avoiding-d3il-v1'

seeds = [5, 6, 7, 8, 9] 
# seeds = [6] 
#Repeat the entire training pipeline across 5 seeds to measure robustness 
#or pick the best checkpoint. Why? Diffusion models (and RL/data-driven models generally) 
#can vary with initialization; multiple seeds give a fairer estimate.

class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp   #points to a config module config.avoiding-d3il.

for seed in seeds:
    args = Parser().parse_args(experiment='diffusion', seed=seed)

    # args.seed = seed
    torch.manual_seed(args.seed)

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
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        include_returns=args.include_returns,
        # returns_scale=args.returns_scale,
        returns_scale=args.max_path_length,               # Because the reward is <= 1 in each timestep
        # returns_scale=args.n_diffusion_steps,
        discount=args.discount,
        stride=getattr(args, 'stride', 1),
        dt=getattr(args, 'dt', 0.005),
    )

    dataset = dataset_config() # <class 'diffuser.datasets.sequence.SequenceDataset'>
    print(f"Dataset loaded.")
    print(f"  Observation dim: {dataset.observation_dim}")
    print(f"  Action dim:      {dataset.action_dim}")
    print(f"  Goal dim:        {dataset.goal_dim}")
    print(f"  Dataset length:  {len(dataset)} (if applicable)")

    field_norms = dataset.normalizer.get_field_normalizers()
    obs_norm = field_norms.get('observations', None)
    act_norm = field_norms.get('actions', None)
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    goal_dim = dataset.goal_dim
    obs_norm = field_norms.get('observations', None)
    act_norm = field_norms.get('actions', None)

    if obs_norm is not None:
        print("\n[Workspace learned from DATASET (observations) via LimitsNormalizer]")
        print("mins:", obs_norm.mins)
        print("maxs:", obs_norm.maxs)

        # if your obs is [x,y,x,y], show x/y nicely
        if len(obs_norm.mins) == 4:
            print(f"x range ~ [{obs_norm.mins[0]:.3f}, {obs_norm.maxs[0]:.3f}]")
            print(f"y range ~ [{obs_norm.mins[1]:.3f}, {obs_norm.maxs[1]:.3f}]")

    if act_norm is not None:
        print("\n[Action range learned from DATASET (actions) via LimitsNormalizer]")
        print("mins:", act_norm.mins)
        print("maxs:", act_norm.maxs)
        if len(act_norm.mins) == 2:
            print(f"dx per step range ~ [{act_norm.mins[0]:.3f}, {act_norm.maxs[0]:.3f}]")
            print(f"dy per step range ~ [{act_norm.mins[1]:.3f}, {act_norm.maxs[1]:.3f}]")

    print("-----------------------------------------------------------------------\n")
    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#

    if args.diffusion == 'models.GaussianInvDynDiffusion':
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            horizon=args.horizon,
            transition_dim=observation_dim,
            cond_dim=observation_dim,
            dim_mults=args.dim_mults,
            returns_condition=args.returns_condition,
            dim=args.dim,
            condition_dropout=args.condition_dropout,
            device=args.device,
        )
        print("Using GaussianInvDynDiffusion model (Inverse Dynamics).")

        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, 'diffusion_config.pkl'),
            horizon=args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            goal_dim=dataset.goal_dim,
            n_timesteps=args.n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            hidden_dim=args.hidden_dim,
            ## loss weighting
            action_weight=args.action_weight,
            loss_discount=args.loss_discount,
            returns_condition=args.returns_condition,
            condition_guidance_w=args.condition_guidance_w,
            device=args.device,
        )
    else:
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            horizon=args.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=args.dim_mults,
            returns_condition=args.returns_condition,
            dim=args.dim,
            condition_dropout=args.condition_dropout,
            device=args.device,
        )
        print("Using standard GaussianDiffusion model (States + Actions).")
        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, 'diffusion_config.pkl'),
            horizon=args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            goal_dim=dataset.goal_dim,
            n_timesteps=args.n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            ## loss weighting
            action_weight=args.action_weight,
            loss_discount=args.loss_discount,
            returns_condition=args.returns_condition,
            condition_guidance_w=args.condition_guidance_w,
            device=args.device,
        )

    # Create trainer
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

    model = model_config() #actually creates the neural network.
    diffusion = diffusion_config(model) #wraps the model in the diffusion process (loss, sampling, etc.).
    trainer = trainer_config(diffusion, dataset) #sets up the training routine (optimizer, data loader, etc.).

    trainer.train() #starts the actual training loop.

    print(f"Model: {model.__class__.__name__}")
    print(f"Diffusion: {diffusion.__class__.__name__}")
    print(f"Trainer: {trainer.__class__.__name__}")
    print(f"Training parameters:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Train steps: {args.n_train_steps}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Device: {args.device}")

    print("\n✅ Everything loaded successfully — skipping trainer.train().")
    print("===========================================================\n")