import os
import time
import torch
from typing import Any, Dict

from utils.utils_for_model import (
    create_parser,
    generate_vrp_instance,
    compute_vrp_tour_length,
    load_stage_ckpt,
)
from pomo_cvrp_policy_asap import POMOVRPStage1Policy, POMOVRPStage2Policy
from vrp_env import VRPEnvironment

try:
    import swanlab
    _HAS_SWANLAB = True
except ImportError:
    swanlab = None  # type: ignore[assignment]
    _HAS_SWANLAB = False


def _init_swanlab(args) -> bool:
    if not getattr(args, "use_swanlab", False) or not _HAS_SWANLAB or not hasattr(swanlab, "init"):
        return False
    project_name = f"pomo_vrp{args.nb_nodes}_{args.stage}"
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


def _model_kwargs(args) -> Dict[str, Any]:
    return {
        "embedding_dim": args.embedding_dim,
        "encoder_layer_num": args.encoder_layer_num,
        "qkv_dim": args.qkv_dim,
        "head_num": args.head_num,
        "ff_hidden_dim": args.ff_hidden_dim,
        "logit_clipping": args.logit_clipping,
        "eval_type": args.eval_type,
    }


def _sanitize_vrp_inputs(x: dict) -> dict:
    cleaned = dict(x)
    demand = cleaned.get("demand")
    if demand is not None:
        if torch.is_floating_point(demand):
            demand = torch.round(demand)
        cleaned["demand"] = demand.long()
    return cleaned


def evaluate_batch_deterministic(
    args,
    stage1_policy: POMOVRPStage1Policy,
    stage2_policy: POMOVRPStage2Policy,
    baseline_policy,
    x_aug: dict,
    x_repeat: dict,
    mode: str,
) -> Dict[str, float]:
    """Greedy evaluation on a single batch (CVRP only)."""
    if mode not in ("stage1", "stage2"):
        raise ValueError(f"Unsupported eval mode {mode}")
    with torch.no_grad():
        stage1_policy.reset(); stage2_policy.reset(); baseline_policy.reset()
        x_aug = _sanitize_vrp_inputs(x_aug)

        env_m = VRPEnvironment(x_aug, capacity=args.capacity, problem="cvrp")
        tours_m = []
        while not env_m.is_finished():
            sel_idx_m, _, _ = stage1_policy.select_k(env_m, k_promising=args.k_promising, deterministic=True)
            act_m, _, _ = stage2_policy.select_action(env_m, selected_global_idx=sel_idx_m, deterministic=True)
            env_m.step(act_m)
            tours_m.append(act_m)
        tours_m = env_m.get_tour_tensor(tours_m)

        env_b = VRPEnvironment(x_aug, capacity=args.capacity, problem="cvrp")
        tours_b = []
        while not env_b.is_finished():
            if mode == "stage1":
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

    return {
        "eval_L_model_mean": L_model.mean().item(),
        "eval_L_baseline_mean": L_baseline.mean().item(),
    }


def train_stage1_step(
    args,
    device,
    stage1_policy: POMOVRPStage1Policy,
    stage2_policy: POMOVRPStage2Policy,
    baseline_stage1_policy: POMOVRPStage1Policy,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    stage1_policy.train()
    stage1_policy.reset(); stage2_policy.reset(); baseline_stage1_policy.reset()

    x_aug, x_repeat = generate_vrp_instance(args, device)
    x_aug = _sanitize_vrp_inputs(x_aug)

    env = VRPEnvironment(x_aug, capacity=args.capacity, problem="cvrp")
    sum_logp_stage1 = []
    tours_model_list = []
    while not env.is_finished():
        selected_idx, selected_probs, _ = stage1_policy.select_k(env, k_promising=args.k_promising, deterministic=False)
        with torch.no_grad():
            chosen, _, _ = stage2_policy.select_action(
                env,
                selected_global_idx=selected_idx,
                deterministic=True,
            )
        choice_mask = selected_idx == chosen.unsqueeze(1)
        chosen_prob = (selected_probs * choice_mask).sum(dim=1).clamp_min(1e-12)
        logp = chosen_prob.log()
        sum_logp_stage1.append(logp)
        env.step(chosen)
        tours_model_list.append(chosen)
    tours_model = env.get_tour_tensor(tours_model_list)
    sum_logp_stage1 = torch.stack(sum_logp_stage1, dim=1).sum(dim=1)

    with torch.no_grad():
        env_bl = VRPEnvironment(x_aug, capacity=args.capacity, problem="cvrp")
        tours_bl_list = []
        while not env_bl.is_finished():
            sel_idx_bl, _, _ = baseline_stage1_policy.select_k(env_bl, k_promising=args.k_promising, deterministic=True)
            chosen_bl, _, _ = stage2_policy.select_action(
                env_bl,
                selected_global_idx=sel_idx_bl,
                deterministic=True,
            )
            env_bl.step(chosen_bl)
            tours_bl_list.append(chosen_bl)
        tours_baseline = env_bl.get_tour_tensor(tours_bl_list)

    L_model = compute_vrp_tour_length(x_repeat, tours_model)
    L_baseline = compute_vrp_tour_length(x_repeat, tours_baseline)
    loss = ((L_model - L_baseline) * sum_logp_stage1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "L_model_mean": L_model.mean().item(),
        "L_baseline_mean": L_baseline.mean().item(),
    }


def train_stage2_step(
    args,
    device,
    stage1_policy: POMOVRPStage1Policy,
    stage2_policy: POMOVRPStage2Policy,
    baseline_stage2_policy: POMOVRPStage2Policy,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    stage2_policy.train()
    stage1_policy.reset(); stage2_policy.reset(); baseline_stage2_policy.reset()

    x_aug, x_repeat = generate_vrp_instance(args, device)
    x_aug = _sanitize_vrp_inputs(x_aug)

    env = VRPEnvironment(x_aug, capacity=args.capacity, problem="cvrp")
    sum_logp_stage2 = []
    tours_model_list = []
    while not env.is_finished():
        with torch.no_grad():
            selected_idx, _, _ = stage1_policy.select_k(env, k_promising=args.k_promising, deterministic=True)
        action, logp2, _ = stage2_policy.select_action(
            env,
            selected_global_idx=selected_idx,
            deterministic=False,
        )
        sum_logp_stage2.append(logp2)
        env.step(action)
        tours_model_list.append(action)
    tours_model = env.get_tour_tensor(tours_model_list)
    sum_logp_stage2 = torch.stack(sum_logp_stage2, dim=1).sum(dim=1)

    with torch.no_grad():
        env_bl = VRPEnvironment(x_aug, capacity=args.capacity, problem="cvrp")
        tours_bl_list = []
        while not env_bl.is_finished():
            sel_idx_bl, _, _ = stage1_policy.select_k(env_bl, k_promising=args.k_promising, deterministic=True)
            action_bl, _, _ = baseline_stage2_policy.select_action(
                env_bl,
                selected_global_idx=sel_idx_bl,
                deterministic=True,
            )
            env_bl.step(action_bl)
            tours_bl_list.append(action_bl)
        tours_baseline = env_bl.get_tour_tensor(tours_bl_list)

    L_model = compute_vrp_tour_length(x_repeat, tours_model)
    L_baseline = compute_vrp_tour_length(x_repeat, tours_baseline)
    loss = ((L_model - L_baseline) * sum_logp_stage2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "L_model_mean": L_model.mean().item(),
        "L_baseline_mean": L_baseline.mean().item(),
    }


def train_stage1_epoch(
    args,
    device,
    stage1_policy: POMOVRPStage1Policy,
    stage2_policy: POMOVRPStage2Policy,
    baseline_stage1_policy: POMOVRPStage1Policy,
    optimizer: torch.optim.Optimizer,
    epoch_idx: int,
    log_to_swanlab: bool = False,
    stage_label: str = "stage1",
) -> None:
    epoch_loss = 0.0
    for _ in range(args.nb_batch_per_epoch):
        metrics = train_stage1_step(args, device, stage1_policy, stage2_policy, baseline_stage1_policy, optimizer)
        epoch_loss += metrics["loss"]
    eval_m = 0.0
    eval_b = 0.0
    for _ in range(args.nb_batch_eval):
        x_aug, x_repeat = generate_vrp_instance(args, device, if_test=True)
        x_aug = _sanitize_vrp_inputs(x_aug)
        eval_metrics = evaluate_batch_deterministic(
            args, stage1_policy, stage2_policy, baseline_stage1_policy, x_aug, x_repeat, mode="stage1"
        )
        eval_m += eval_metrics["eval_L_model_mean"]
        eval_b += eval_metrics["eval_L_baseline_mean"]
    avg_loss = epoch_loss / args.nb_batch_per_epoch
    eval_m_avg = eval_m / args.nb_batch_eval
    eval_b_avg = eval_b / args.nb_batch_eval
    tol = 1e-4
    if eval_m_avg < eval_b_avg - tol:
        baseline_stage1_policy.load_state_dict(stage1_policy.state_dict())
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                {
                    "stage": "stage1",
                    "policy_state_dict": stage1_policy.state_dict(),
                    "args": vars(args),
                    "best_eval_Lm": eval_m_avg,
                    "best_epoch": epoch_idx,
                },
                os.path.join(args.save_dir, "stage1.ckpt"),
            )
    print(f"[POMO VRP Stage1][Epoch {epoch_idx}] loss={avg_loss:.4f} evalLm={eval_m_avg:.4f} evalLb={eval_b_avg:.4f}")
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
    stage1_policy: POMOVRPStage1Policy,
    stage2_policy: POMOVRPStage2Policy,
    baseline_stage2_policy: POMOVRPStage2Policy,
    optimizer: torch.optim.Optimizer,
    epoch_idx: int,
    log_to_swanlab: bool = False,
    stage_label: str = "stage2",
) -> None:
    epoch_loss = 0.0
    for _ in range(args.nb_batch_per_epoch):
        metrics = train_stage2_step(args, device, stage1_policy, stage2_policy, baseline_stage2_policy, optimizer)
        epoch_loss += metrics["loss"]
    eval_m = 0.0
    eval_b = 0.0
    for _ in range(args.nb_batch_eval):
        x_aug, x_repeat = generate_vrp_instance(args, device, if_test=True)
        x_aug = _sanitize_vrp_inputs(x_aug)
        eval_metrics = evaluate_batch_deterministic(
            args, stage1_policy, stage2_policy, baseline_stage2_policy, x_aug, x_repeat, mode="stage2"
        )
        eval_m += eval_metrics["eval_L_model_mean"]
        eval_b += eval_metrics["eval_L_baseline_mean"]
    avg_loss = epoch_loss / args.nb_batch_per_epoch
    eval_m_avg = eval_m / args.nb_batch_eval
    eval_b_avg = eval_b / args.nb_batch_eval
    tol = 1e-4
    if eval_m_avg < eval_b_avg - tol:
        baseline_stage2_policy.load_state_dict(stage2_policy.state_dict())
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                {
                    "stage": "stage2",
                    "policy_state_dict": stage2_policy.state_dict(),
                    "args": vars(args),
                    "best_eval_Lm": eval_m_avg,
                    "best_epoch": epoch_idx,
                },
                os.path.join(args.save_dir, "stage2.ckpt"),
            )
    print(f"[POMO VRP Stage2][Epoch {epoch_idx}] loss={avg_loss:.4f} evalLm={eval_m_avg:.4f} evalLb={eval_b_avg:.4f}")
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
    stage1_policy: POMOVRPStage1Policy,
    stage2_policy: POMOVRPStage2Policy,
    baseline_stage1_policy: POMOVRPStage1Policy,
    optimizer: torch.optim.Optimizer,
    log_to_swanlab: bool = False,
) -> None:
    for epoch in range(args.nb_epochs_stage1):
        train_stage1_epoch(
            args, device, stage1_policy, stage2_policy, baseline_stage1_policy, optimizer, epoch,
            log_to_swanlab=log_to_swanlab, stage_label="stage1"
        )


def train_stage2_policy(
    args,
    device,
    stage1_policy: POMOVRPStage1Policy,
    stage2_policy: POMOVRPStage2Policy,
    baseline_stage2_policy: POMOVRPStage2Policy,
    optimizer: torch.optim.Optimizer,
    log_to_swanlab: bool = False,
) -> None:
    for p in stage1_policy.parameters():
        p.requires_grad = False
    for epoch in range(args.nb_epochs_stage2):
        train_stage2_epoch(
            args, device, stage1_policy, stage2_policy, baseline_stage2_policy, optimizer, epoch,
            log_to_swanlab=log_to_swanlab, stage_label="stage2"
        )


def train_alternating(
    args,
    device,
    stage1_policy: POMOVRPStage1Policy,
    stage2_policy: POMOVRPStage2Policy,
    baseline_stage1_policy: POMOVRPStage1Policy,
    baseline_stage2_policy: POMOVRPStage2Policy,
    optimizer_stage1: torch.optim.Optimizer,
    optimizer_stage2: torch.optim.Optimizer,
    log_to_swanlab: bool = False,
) -> None:
    for epoch in range(args.nb_epochs_alt):
        train_stage1_epoch(
            args, device, stage1_policy, stage2_policy, baseline_stage1_policy, optimizer_stage1, epoch,
            log_to_swanlab=log_to_swanlab, stage_label="alt_stage1"
        )
        for p in stage1_policy.parameters():
            p.requires_grad = False
        train_stage2_epoch(
            args, device, stage1_policy, stage2_policy, baseline_stage2_policy, optimizer_stage2, epoch,
            log_to_swanlab=log_to_swanlab, stage_label="alt_stage2"
        )
        for p in stage1_policy.parameters():
            p.requires_grad = True


def build_args():
    config_dict = {
        "stage": "stage1",            # 'stage1', 'stage2', or 'alt'
        "bsz": 64,
        "nb_nodes": 50,
        "capacity": 50.0,
        "dim_input_nodes": 2,
        "embedding_dim": 128,
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "ff_hidden_dim": 512,
        "logit_clipping": 10.0,
        "eval_type": "sampling",
        "model_lr_stage1": 2e-5,
        "model_lr_stage2": 2e-5,
        "nb_epochs_stage1": 10,
        "nb_epochs_stage2": 200,
        "nb_epochs_alt": 5,
        "nb_batch_per_epoch": 300,
        "nb_batch_eval": 20,
        "aug": "mix",
        "aug_num": 16,
        "test_aug_num": 16,
        "k_promising": 8,
        "use_swanlab": False,
        "save_dir": "./checkpoints/pomo_cvrp_asap",
        "stage1_ckpt": "",
        "stage1_init_ckpt": "",
        "stage2_init_ckpt": "",
    }
    parser, args = create_parser(config_dict)
    args = parser.parse_args(namespace=args)

    args.capacity = float(args.capacity)

    base_dir = args.save_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"cvrp{args.nb_nodes}/model_{timestamp}"
    args.save_dir = os.path.join(base_dir, run_dir_name)

    return args


def main():
    args = build_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_to_swanlab = _init_swanlab(args)

    model_kwargs = _model_kwargs(args)
    try:
        if args.stage == "stage1":
            stage1 = POMOVRPStage1Policy(**model_kwargs).to(device)
            stage2 = POMOVRPStage2Policy(**model_kwargs).to(device)
            baseline_stage1 = POMOVRPStage1Policy(**model_kwargs).to(device)
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
            stage1 = POMOVRPStage1Policy(**model_kwargs).to(device)
            stage2 = POMOVRPStage2Policy(**model_kwargs).to(device)
            baseline_stage1 = POMOVRPStage1Policy(**model_kwargs).to(device)
            baseline_stage2 = POMOVRPStage2Policy(**model_kwargs).to(device)
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
            stage1_fixed = POMOVRPStage1Policy(**model_kwargs).to(device)
            if args.stage1_ckpt:
                load_stage_ckpt(stage1_fixed, args.stage1_ckpt, device, expected_stage=None)
            elif args.stage1_init_ckpt:
                load_stage_ckpt(stage1_fixed, args.stage1_init_ckpt, device, expected_stage=None)
            else:
                raise ValueError("Training Stage 2 requires --stage1_ckpt (trained) or --stage1_init_ckpt (pretrained).")
            for p in stage1_fixed.parameters():
                p.requires_grad = False
            stage2 = POMOVRPStage2Policy(**model_kwargs).to(device)
            baseline_stage2 = POMOVRPStage2Policy(**model_kwargs).to(device)
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
