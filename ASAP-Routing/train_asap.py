import os
import time
import torch

from utils.utils_for_model import (
    create_parser,
    generate_tsp_instance,
    generate_vrp_instance,
    compute_tsp_tour_length,
    compute_vrp_tour_length,
    load_stage_ckpt,
)
from tsp_env import TSPEnvironment
from tsp_policy_asap import TSPStage1Policy, TSPStage2Policy
from vrp_env import VRPEnvironment
from cvrp_policy_asap import VRPStage1Policy, VRPStage2Policy

try:
    import swanlab
    _HAS_SWANLAB = True
except ImportError:
    swanlab = None  # type: ignore[assignment]
    _HAS_SWANLAB = False


def _init_swanlab(args) -> bool:
    if not getattr(args, "use_swanlab", False) or not _HAS_SWANLAB or not hasattr(swanlab, "init"):
        return False
    problem = str(getattr(args, "problem", "tsp")).lower()
    project_name = f"{problem}{args.nb_nodes}_{args.stage}"
    exp_name = None
    if getattr(args, "save_dir", None):
        exp_name = os.path.basename(str(args.save_dir).rstrip(os.sep))
    if not exp_name:
        exp_name = project_name
    try:
        swanlab.init(
            project=project_name,
            experiment_name=exp_name,
            config=vars(args),
        )
    except Exception:
        return False
    return True


def _sanitize_vrp_inputs(x: dict) -> dict:
    """Ensure VRP demands are kept as integers inside the environment."""
    cleaned = dict(x)
    if "demand" in cleaned:
        demand = cleaned["demand"]
        if torch.is_floating_point(demand):
            demand = torch.round(demand)
        cleaned["demand"] = demand.long()
    return cleaned


def evaluate_batch_deterministic(
    args,
    stage1_policy,
    stage2_policy,
    baseline_policy,
    x_aug,
    x_repeat,
    mode: str,
) -> dict:
    """Greedy evaluation on a single batch with deterministic sampling.

    mode:
      - 'stage1': baseline_policy is a Stage1 policy; stage2_policy is shared.
      - 'stage2': baseline_policy is a Stage2 policy; stage1_policy is shared/fixed.
    """
    if mode not in ('stage1', 'stage2'):
        raise ValueError(f"Unsupported eval mode {mode}")
    problem = str(args.problem).lower()
    with torch.no_grad():
        if problem == 'cvrp':
            x_aug = _sanitize_vrp_inputs(x_aug)
            env_m = VRPEnvironment(x_aug, capacity=args.capacity, problem=problem)
            tours_m = []
            while not env_m.is_finished():
                sel_idx_m, _, _ = stage1_policy.select_k(env_m, k_promising=args.k_promising, deterministic=True)
                act_m, _, _ = stage2_policy.select_action(env_m, selected_global_idx=sel_idx_m, deterministic=True)
                env_m.step(act_m)
                tours_m.append(act_m)
            tours_m = env_m.get_tour_tensor(tours_m)

            env_b = VRPEnvironment(x_aug, capacity=args.capacity, problem=problem)
            tours_b = []
            while not env_b.is_finished():
                if mode == 'stage1':
                    sel_idx_b, _, _ = baseline_policy.select_k(env_b, k_promising=args.k_promising, deterministic=True)
                    act_b, _, _ = stage2_policy.select_action(env_b, selected_global_idx=sel_idx_b, deterministic=True)
                else:
                    sel_idx_b, _, _ = stage1_policy.select_k(env_b, k_promising=args.k_promising, deterministic=True)
                    act_b, _, _ = baseline_policy.select_action(env_b, selected_global_idx=sel_idx_b, deterministic=True)
                env_b.step(act_b)
                tours_b.append(act_b)
            tours_b = env_b.get_tour_tensor(tours_b)

            L_model = compute_vrp_tour_length(x_repeat, tours_m)
            L_baseline = compute_vrp_tour_length(x_repeat, tours_b)
        else:
            env_m = TSPEnvironment(x_aug)
            obs_m = env_m.observation()
            for _ in range(env_m.nb_nodes - 1):
                sel_idx_m, _, _ = stage1_policy.select_k(obs_m, k_promising=args.k_promising, deterministic=True)
                act_m, _, _ = stage2_policy.select_action(
                    obs_m,
                    selected_global_idx=sel_idx_m,
                    deterministic=True,
                )
                obs_m, done_m = env_m.step(act_m)
                if done_m:
                    break
            tours_m = env_m.get_tour_tensor()

            env_b = TSPEnvironment(x_aug)
            obs_b = env_b.observation()
            for _ in range(env_b.nb_nodes - 1):
                if mode == 'stage1':
                    sel_idx_b, _, _ = baseline_policy.select_k(obs_b, k_promising=args.k_promising, deterministic=True)
                    act_b, _, _ = stage2_policy.select_action(
                        obs_b,
                        selected_global_idx=sel_idx_b,
                        deterministic=True,
                    )
                else:  # stage2 baseline
                    sel_idx_b, _, _ = stage1_policy.select_k(obs_b, k_promising=args.k_promising, deterministic=True)
                    act_b, _, _ = baseline_policy.select_action(
                        obs_b,
                        selected_global_idx=sel_idx_b,
                        deterministic=True,
                    )
                obs_b, done_b = env_b.step(act_b)
                if done_b:
                    break
            tours_b = env_b.get_tour_tensor()

            L_model = compute_tsp_tour_length(x_aug, tours_m)
            L_baseline = compute_tsp_tour_length(x_aug, tours_b)

    return {
        'eval_L_model_mean': L_model.mean().item(),
        'eval_L_baseline_mean': L_baseline.mean().item(),
    }


def train_stage1_step(
    args,
    device,
    stage1_policy,
    stage2_policy,
    baseline_stage1_policy,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """Single training step for Stage 1 via REINFORCE with baseline.

    Builds two tours on the same batch:
      1) stage1_policy (with grad) + stage2_policy (no grad) → use Stage1 log-prob of final actions
      2) baseline_stage1_policy (no grad) + stage2_policy (no grad) → baseline tours
    Optimizes: E[(L_model - L_baseline) * sum_log_p_stage1].
    Returns a dict of scalar metrics.
    """
    stage1_policy.train()

    problem = str(args.problem).lower()
    # Generate batch
    if problem == 'cvrp':
        x_aug, x_repeat = generate_vrp_instance(args, device)
        x_aug = _sanitize_vrp_inputs(x_aug)
        env = VRPEnvironment(x_aug, capacity=args.capacity, problem=problem)
        sum_logp_stage1 = []
        tours_model_list = []
        while not env.is_finished():
            selected_idx, _, info1 = stage1_policy.select_k(env, k_promising=args.k_promising, deterministic=False)
            with torch.no_grad():
                chosen, _, _ = stage2_policy.select_action(env, selected_global_idx=selected_idx, deterministic=True)

            # Map Stage2 choice back to the Stage1 distribution (which already includes the depot slot).
            candidate_global_idx = info1["candidate_global_idx"]        # (bsz, k_candidates)
            candidate_probs = info1["candidate_probs"]                  # (bsz, k_candidates)
            choice_mask = candidate_global_idx == chosen.unsqueeze(1)
            chosen_prob = (candidate_probs * choice_mask).sum(dim=1)
            logp = torch.log(chosen_prob.clamp_min(1e-12))
            sum_logp_stage1.append(logp)
            env.step(chosen)
            tours_model_list.append(chosen)
        tours_model = env.get_tour_tensor(tours_model_list)
        sum_logp_stage1 = torch.stack(sum_logp_stage1, dim=1).sum(dim=1)

        with torch.no_grad():
            env_bl = VRPEnvironment(x_aug, capacity=args.capacity, problem=problem)
            tours_bl_list = []
            while not env_bl.is_finished():
                sel_idx_bl, _, _ = baseline_stage1_policy.select_k(env_bl, k_promising=args.k_promising, deterministic=True)
                chosen_bl, _, _ = stage2_policy.select_action(env_bl, selected_global_idx=sel_idx_bl, deterministic=True)
                env_bl.step(chosen_bl)
                tours_bl_list.append(chosen_bl)
            tours_baseline = env_bl.get_tour_tensor(tours_bl_list)

        L_model = compute_vrp_tour_length(x_repeat, tours_model)
        L_baseline = compute_vrp_tour_length(x_repeat, tours_baseline)
    else:
        x_aug, x_repeat = generate_tsp_instance(args, device)

        # Model rollout
        env = TSPEnvironment(x_aug)
        obs = env.observation()
        sum_logp_stage1 = []
        for _ in range(env.nb_nodes - 1):
            selected_idx, selected_probs, _ = stage1_policy.select_k(obs, k_promising=args.k_promising, deterministic=False)
            with torch.no_grad():
                chosen, _, info2 = stage2_policy.select_action(
                    obs,
                    selected_global_idx=selected_idx,
                    deterministic=True,
                )
            idx_in_k = info2["select_idx"]
            bsz = selected_probs.size(0)
            ar = torch.arange(bsz, device=selected_probs.device)
            logp = torch.log(selected_probs[ar, idx_in_k].clamp_min(1e-12))
            sum_logp_stage1.append(logp)
            obs, done = env.step(chosen)
            if done:
                break
        tours_model = env.get_tour_tensor()
        sum_logp_stage1 = torch.stack(sum_logp_stage1, dim=1).sum(dim=1)

        with torch.no_grad():
            env_bl = TSPEnvironment(x_aug)
            obs_bl = env_bl.observation()
            for _ in range(env_bl.nb_nodes - 1):
                sel_idx_bl, _, _ = baseline_stage1_policy.select_k(obs_bl, k_promising=args.k_promising, deterministic=True)
                chosen_bl, _, _ = stage2_policy.select_action(
                    obs_bl,
                    selected_global_idx=sel_idx_bl,
                    deterministic=True,
                )
                obs_bl, done_bl = env_bl.step(chosen_bl)
                if done_bl:
                    break
            tours_baseline = env_bl.get_tour_tensor()

        L_model = compute_tsp_tour_length(x_repeat, tours_model)
        L_baseline = compute_tsp_tour_length(x_repeat, tours_baseline)
    loss = ((L_model - L_baseline) * sum_logp_stage1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # eval_metrics = evaluate_batch_deterministic(
    #     args, stage1_policy, stage2_policy, baseline_stage1_policy, x_aug, x_repeat, mode='stage1'
    # )

    return {
        'loss': loss.item(),
        'L_model_mean': L_model.mean().item(),
        'L_baseline_mean': L_baseline.mean().item(),
        # 'eval_L_model_mean': eval_metrics['eval_L_model_mean'],
        # 'eval_L_baseline_mean': eval_metrics['eval_L_baseline_mean'],
    }


def train_stage2_step(
    args,
    device,
    stage1_policy,
    stage2_policy,
    baseline_stage2_policy,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """Single training step for Stage 2 via REINFORCE with baseline.

    Builds two tours on the same batch:
      1) stage1_policy (no grad) + stage2_policy (with grad) → Stage2 log-probs
      2) stage1_policy (no grad) + baseline_stage2_policy (no grad) → baseline tours
    Optimizes: E[(L_model - L_baseline) * sum_log_p_stage2].
    Returns a dict of scalar metrics.
    """
    stage2_policy.train()

    problem = str(args.problem).lower()
    if problem == 'cvrp':
        x_aug, x_repeat = generate_vrp_instance(args, device)
        x_aug = _sanitize_vrp_inputs(x_aug)
        env = VRPEnvironment(x_aug, capacity=args.capacity, problem=problem)
        sum_logp_stage2 = []
        tours_model_list = []
        while not env.is_finished():
            with torch.no_grad():
                selected_idx, _, _ = stage1_policy.select_k(env, k_promising=args.k_promising, deterministic=True)
            action, logp2, _ = stage2_policy.select_action(env, selected_global_idx=selected_idx, deterministic=False)
            sum_logp_stage2.append(logp2)
            env.step(action)
            tours_model_list.append(action)
        tours_model = env.get_tour_tensor(tours_model_list)
        sum_logp_stage2 = torch.stack(sum_logp_stage2, dim=1).sum(dim=1)

        with torch.no_grad():
            env_bl = VRPEnvironment(x_aug, capacity=args.capacity, problem=problem)
            tours_bl_list = []
            while not env_bl.is_finished():
                sel_idx_bl, _, _ = stage1_policy.select_k(env_bl, k_promising=args.k_promising, deterministic=True)
                action_bl, _, _ = baseline_stage2_policy.select_action(
                    env_bl, selected_global_idx=sel_idx_bl, deterministic=True
                )
                env_bl.step(action_bl)
                tours_bl_list.append(action_bl)
            tours_baseline = env_bl.get_tour_tensor(tours_bl_list)

        L_model = compute_vrp_tour_length(x_repeat, tours_model)
        L_baseline = compute_vrp_tour_length(x_repeat, tours_baseline)
    else:
        x_aug, x_repeat = generate_tsp_instance(args, device)
        env = TSPEnvironment(x_aug)
        obs = env.observation()
        sum_logp_stage2 = []
        for _ in range(env.nb_nodes - 1):
            with torch.no_grad():
                selected_idx, _, _ = stage1_policy.select_k(obs, k_promising=args.k_promising, deterministic=True)
            action, logp2, _ = stage2_policy.select_action(
                obs,
                selected_global_idx=selected_idx,
                deterministic=False,
            )
            sum_logp_stage2.append(logp2)
            obs, done = env.step(action)
            if done:
                break
        tours_model = env.get_tour_tensor()
        sum_logp_stage2 = torch.stack(sum_logp_stage2, dim=1).sum(dim=1)

        with torch.no_grad():
            env_bl = TSPEnvironment(x_aug)
            obs_bl = env_bl.observation()
            for _ in range(env_bl.nb_nodes - 1):
                sel_idx_bl, _, _ = stage1_policy.select_k(obs_bl, k_promising=args.k_promising, deterministic=True)
                action_bl, _, _ = baseline_stage2_policy.select_action(
                    obs_bl,
                    selected_global_idx=sel_idx_bl,
                    deterministic=True,
                )
                obs_bl, done_bl = env_bl.step(action_bl)
                if done_bl:
                    break
            tours_baseline = env_bl.get_tour_tensor()

        L_model = compute_tsp_tour_length(x_aug, tours_model)
        L_baseline = compute_tsp_tour_length(x_aug, tours_baseline)
    loss = ((L_model - L_baseline) * sum_logp_stage2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # eval_metrics = evaluate_batch_deterministic(
    #     args, stage1_policy, stage2_policy, baseline_stage2_policy, x_aug, x_repeat, mode='stage2'
    # )

    return {
        'loss': loss.item(),
        'L_model_mean': L_model.mean().item(),
        'L_baseline_mean': L_baseline.mean().item(),
        # 'eval_L_model_mean': eval_metrics['eval_L_model_mean'],
        # 'eval_L_baseline_mean': eval_metrics['eval_L_baseline_mean'],
    }


def train_stage1_epoch(
    args,
    device,
    stage1_policy,
    stage2_policy,
    baseline_stage1_policy,
    optimizer: torch.optim.Optimizer,
    epoch_idx: int,
    log_to_swanlab: bool = False,
    stage_label: str = "stage1",
) -> None:
    """Run one epoch of Stage 1 training with baseline replacement based on eval."""
    epoch_loss = 0.0
    for _ in range(args.nb_batch_per_epoch):
        metrics = train_stage1_step(args, device, stage1_policy, stage2_policy, baseline_stage1_policy, optimizer)
        epoch_loss += metrics['loss']
    eval_m = 0.0; eval_b = 0.0
    for _ in range(args.nb_batch_eval):
        if str(args.problem).lower() == 'cvrp':
            x_aug, x_repeat = generate_vrp_instance(args, device, if_test=True)
            x_aug = _sanitize_vrp_inputs(x_aug)
        else:
            x_aug, x_repeat = generate_tsp_instance(args, device, if_test=True)
        eval_metrics = evaluate_batch_deterministic(
            args, stage1_policy, stage2_policy, baseline_stage1_policy, x_aug, x_repeat, mode='stage1'
        )
        eval_m += eval_metrics['eval_L_model_mean']
        eval_b += eval_metrics['eval_L_baseline_mean']
    avg_loss = epoch_loss/args.nb_batch_per_epoch
    eval_m_avg = eval_m/args.nb_batch_eval
    eval_b_avg = eval_b/args.nb_batch_eval
    tol = 1e-4
    if eval_m_avg < eval_b_avg - tol:
        baseline_stage1_policy.load_state_dict(stage1_policy.state_dict())
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                {
                    'stage': 'pretrain',
                    'policy_state_dict': stage1_policy.state_dict(),
                    'args': vars(args),
                    'best_eval_Lm': eval_m_avg,
                    'best_epoch': epoch_idx,
                },
                os.path.join(args.save_dir, 'stage1.ckpt'),
            )
    print(f"[Stage1][Epoch {epoch_idx}] loss={epoch_loss/args.nb_batch_per_epoch:.4f} "
          f"evalLm={eval_m_avg:.4f} evalLb={eval_b_avg:.4f}")
    if log_to_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log"):
        swanlab.log({
            "stage": stage_label,
            "epoch": epoch_idx,
            "train/loss": avg_loss,
            "eval/L_model": eval_m_avg,
            "eval/L_baseline": eval_b_avg,
        })


def train_stage2_epoch(
    args,
    device,
    stage1_policy,
    stage2_policy,
    baseline_stage2_policy,
    optimizer: torch.optim.Optimizer,
    epoch_idx: int,
    log_to_swanlab: bool = False,
    stage_label: str = "stage2",
) -> None:
    """Run one epoch of Stage 2 training with baseline replacement based on eval."""
    epoch_loss = 0.0
    for _ in range(args.nb_batch_per_epoch):
        metrics = train_stage2_step(args, device, stage1_policy, stage2_policy, baseline_stage2_policy, optimizer)
        epoch_loss += metrics['loss']
    eval_m = 0.0; eval_b = 0.0
    for _ in range(args.nb_batch_eval):
        if str(args.problem).lower() == 'cvrp':
            x_aug, x_repeat = generate_vrp_instance(args, device, if_test=True)
            x_aug = _sanitize_vrp_inputs(x_aug)
        else:
            x_aug, x_repeat = generate_tsp_instance(args, device, if_test=True)
        eval_metrics = evaluate_batch_deterministic(
            args, stage1_policy, stage2_policy, baseline_stage2_policy, x_aug, x_repeat, mode='stage2'
        )
        eval_m += eval_metrics['eval_L_model_mean']
        eval_b += eval_metrics['eval_L_baseline_mean']
    avg_loss = epoch_loss/args.nb_batch_per_epoch
    eval_m_avg = eval_m/args.nb_batch_eval
    eval_b_avg = eval_b/args.nb_batch_eval
    tol = 1e-4
    if eval_m_avg < eval_b_avg - tol:
        baseline_stage2_policy.load_state_dict(stage2_policy.state_dict())
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                {
                    'stage': 'pretrain',
                    'policy_state_dict': stage2_policy.state_dict(),
                    'args': vars(args),
                    'best_eval_Lm': eval_m_avg,
                    'best_epoch': epoch_idx,
                },
                os.path.join(args.save_dir, 'stage2.ckpt'),
            )
    print(f"[Stage2][Epoch {epoch_idx}] loss={epoch_loss/args.nb_batch_per_epoch:.4f} "
          f"evalLm={eval_m_avg:.4f} evalLb={eval_b_avg:.4f}")
    if log_to_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log"):
        swanlab.log({
            "stage": stage_label,
            "epoch": epoch_idx,
            "train/loss": avg_loss,
            "eval/L_model": eval_m_avg,
            "eval/L_baseline": eval_b_avg,
        })


def train_stage1_policy(
    args,
    device,
    stage1_policy,
    stage2_policy,
    baseline_stage1_policy,
    optimizer: torch.optim.Optimizer,
    log_to_swanlab: bool = False,
) -> None:
    """Train Stage 1 across epochs with EMA baseline and quick eval per epoch."""
    for epoch in range(args.nb_epochs_stage1):
        train_stage1_epoch(args, device, stage1_policy, stage2_policy, baseline_stage1_policy, optimizer, epoch,
                           log_to_swanlab=log_to_swanlab, stage_label="stage1")


def train_stage2_policy(
    args,
    device,
    stage1_policy,
    stage2_policy,
    baseline_stage2_policy,
    optimizer: torch.optim.Optimizer,
    log_to_swanlab: bool = False,
) -> None:
    """Train Stage 2 across epochs with EMA baseline and quick eval per epoch."""
    # Ensure stage1 frozen
    for p in stage1_policy.parameters():
        p.requires_grad = False
    for epoch in range(args.nb_epochs_stage2):
        train_stage2_epoch(args, device, stage1_policy, stage2_policy, baseline_stage2_policy, optimizer, epoch,
                           log_to_swanlab=log_to_swanlab, stage_label="stage2")


def train_alternating(
    args,
    device,
    stage1_policy,
    stage2_policy,
    baseline_stage1_policy,
    baseline_stage2_policy,
    optimizer_stage1: torch.optim.Optimizer,
    optimizer_stage2: torch.optim.Optimizer,
    log_to_swanlab: bool = False,
) -> None:
    """Alternating training: Stage1 epoch, then Stage2 epoch, repeated."""
    for epoch in range(args.nb_epochs_alt):
        # Stage 1 epoch (stage2 used only for greedy/eval)
        train_stage1_epoch(
            args, device, stage1_policy, stage2_policy, baseline_stage1_policy, optimizer_stage1, epoch,
            log_to_swanlab=log_to_swanlab, stage_label="alt_stage1"
        )
        # Freeze Stage 1 while training Stage 2
        for p in stage1_policy.parameters():
            p.requires_grad = False
        train_stage2_epoch(
            args, device, stage1_policy, stage2_policy, baseline_stage2_policy, optimizer_stage2, epoch,
            log_to_swanlab=log_to_swanlab, stage_label="alt_stage2"
        )
        # Unfreeze for the next Stage 1 epoch
        for p in stage1_policy.parameters():
            p.requires_grad = True


def build_args():
    config_dict = {
        'problem': 'tsp',               # tsp or cvrp
        'stage': 'stage1',                # 'stage1', 'stage2', or 'alt'/'alternating'
        'bsz': 64,
        'nb_nodes': 50,
        'dim_input_nodes': 2,
        'dim_emb': 128,
        'dim_ff': 512,
        'nb_heads': 8,
        'nb_layers_action_encoder': 2,
        'nb_layers_decoder': 3,
        'batchnorm': False,
        'model_lr_stage1': 2e-5,
        'model_lr_stage2': 2e-5,
        'nb_epochs_stage1': 10,
        'nb_epochs_stage2': 10,
        'nb_batch_per_epoch': 300,
        'nb_batch_eval': 20,
        'aug': 'mix',
        'aug_num': 16,
        'test_aug_num': 16,
        'k_promising': 8,
        'action_k': 15,
        'state_k': '35,50,65',
        'knn_k': 25,
        'gamma': 0.99,
        'ema': 0.99,
        'data_path': './',
        'use_swanlab': False,
        'save_dir': './checkpoints/tsp_pretrain',
        'stage1_ckpt': '',
        'stage1_init_ckpt': '',
        'stage2_init_ckpt': '',
        'nb_epochs_alt': 5,
        'use_normalization_layer': True,
        'use_stage1_action_encoding': 'True',
        # VRP-specific
        'num_state_encoder': 1,
        'nb_layers_state_encoder': 2,
        'if_use_local_mask': False,
        'if_agg_whole_graph': False,
        'capacity': 50,
    }
    parser, args = create_parser(config_dict)
    args = parser.parse_args(namespace=args)
    args.capacity = float(args.capacity)

    if isinstance(args.state_k, str):
        args.state_k = [int(x) for x in args.state_k.split(',') if x.strip()]
    if len(args.state_k) < args.num_state_encoder:
        args.state_k = (args.state_k + args.state_k[-1:])[:args.num_state_encoder]
    else:
        args.state_k = args.state_k[:args.num_state_encoder]
    if isinstance(args.use_stage1_action_encoding, str):
        args.use_stage1_action_encoding = args.use_stage1_action_encoding.lower() not in ('false', '0', 'no')
    else:
        args.use_stage1_action_encoding = bool(args.use_stage1_action_encoding)

    base_dir = args.save_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{args.problem}{args.nb_nodes}/model_{timestamp}"
    args.save_dir = os.path.join(base_dir, run_dir_name)

    return args


def main():
    args = build_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_to_swanlab = _init_swanlab(args)

    proposal_class = VRPStage1Policy if args.problem == 'cvrp' else TSPStage1Policy
    selection_class = VRPStage2Policy if args.problem == 'cvrp' else TSPStage2Policy
    try:
        if args.stage == "stage1":
            stage1 = proposal_class(args).to(device)
            stage2 = selection_class(args).to(device)
            baseline_stage1 = proposal_class(args).to(device)
            if args.stage1_init_ckpt:
                load_stage_ckpt(stage1, args.stage1_init_ckpt, device, expected_stage=None)
            if args.stage2_init_ckpt:
                load_stage_ckpt(stage2, args.stage2_init_ckpt, device, expected_stage=None)
            baseline_stage1.load_state_dict(stage1.state_dict())
            opt1 = torch.optim.AdamW(stage1.parameters(), lr=args.model_lr_stage1)
            train_stage1_policy(args, device, stage1, stage2, baseline_stage1, opt1, log_to_swanlab=log_to_swanlab)
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(
                    {"stage": "stage1", "policy_state_dict": stage1.state_dict(), "args": vars(args)},
                    os.path.join(args.save_dir, "stage1.ckpt"),
                )
        elif args.stage in ("alt", "alternating"):
            stage1 = proposal_class(args).to(device)
            stage2 = selection_class(args).to(device)
            baseline_stage1 = proposal_class(args).to(device)
            baseline_stage2 = selection_class(args).to(device)
            if args.stage1_init_ckpt:
                load_stage_ckpt(stage1, args.stage1_init_ckpt, device, expected_stage=None)
            if args.stage2_init_ckpt:
                load_stage_ckpt(stage2, args.stage2_init_ckpt, device, expected_stage=None)
            baseline_stage1.load_state_dict(stage1.state_dict())
            baseline_stage2.load_state_dict(stage2.state_dict())
            opt1 = torch.optim.AdamW(stage1.parameters(), lr=args.model_lr_stage1)
            opt2 = torch.optim.AdamW(stage2.parameters(), lr=args.model_lr_stage2)
            train_alternating(
                args, device, stage1, stage2, baseline_stage1, baseline_stage2, opt1, opt2, log_to_swanlab=log_to_swanlab
            )
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(
                    {"stage": "stage1", "policy_state_dict": stage1.state_dict(), "args": vars(args)},
                    os.path.join(args.save_dir, "stage1.ckpt"),
                )
                torch.save(
                    {"stage": "stage2", "policy_state_dict": stage2.state_dict(), "args": vars(args)},
                    os.path.join(args.save_dir, "stage2.ckpt"),
                )
        elif args.stage == "stage2":
            stage1_fixed = proposal_class(args).to(device)
            if args.stage1_ckpt:
                load_stage_ckpt(stage1_fixed, args.stage1_ckpt, device, expected_stage=None)
            elif args.stage1_init_ckpt:
                load_stage_ckpt(stage1_fixed, args.stage1_init_ckpt, device, expected_stage=None)
            else:
                raise ValueError("Training Stage 2 requires --stage1_ckpt (trained) or --stage1_init_ckpt (pretrained).")
            for p in stage1_fixed.parameters():
                p.requires_grad = False
            stage2 = selection_class(args).to(device)
            baseline_stage2 = selection_class(args).to(device)
            if args.stage2_init_ckpt:
                load_stage_ckpt(stage2, args.stage2_init_ckpt, device, expected_stage=None)
            baseline_stage2.load_state_dict(stage2.state_dict())
            opt2 = torch.optim.AdamW(stage2.parameters(), lr=args.model_lr_stage2)
            train_stage2_policy(
                args, device, stage1_fixed, stage2, baseline_stage2, opt2, log_to_swanlab=log_to_swanlab
            )
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(
                    {"stage": "stage2", "policy_state_dict": stage2.state_dict(), "args": vars(args)},
                    os.path.join(args.save_dir, "stage2.ckpt"),
                )
        else:
            raise ValueError("Unknown stage: choose stage1, stage2, or alt/alternating")
    finally:
        if log_to_swanlab and _HAS_SWANLAB and hasattr(swanlab, "finish"):
            swanlab.finish()


if __name__ == "__main__":
    main()
