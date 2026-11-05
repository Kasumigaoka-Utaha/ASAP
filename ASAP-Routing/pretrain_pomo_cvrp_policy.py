import os
import time
import torch

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


def _best_over_augmented(lengths: torch.Tensor, aug_num: int) -> torch.Tensor:
    """Return per-base-instance best lengths across augmented variants."""
    if lengths.numel() % aug_num != 0:
        raise ValueError("Length tensor size must be divisible by aug_num for best-of-aug reduction.")
    base = lengths.numel() // aug_num
    return lengths.view(base, aug_num).min(dim=1).values


def _sanitize_vrp_inputs(x: dict) -> dict:
    """Ensure VRP demands are integer inside the environment."""
    cleaned = dict(x)
    demand = cleaned.get("demand")
    if demand is not None:
        if torch.is_floating_point(demand):
            demand = torch.round(demand)
        cleaned["demand"] = demand.long()
    return cleaned


def _model_kwargs(args) -> dict:
    return {
        "embedding_dim": args.embedding_dim,
        "encoder_layer_num": args.encoder_layer_num,
        "qkv_dim": args.qkv_dim,
        "head_num": args.head_num,
        "ff_hidden_dim": args.ff_hidden_dim,
        "logit_clipping": args.logit_clipping,
        "eval_type": args.eval_type,
    }


def _select_cvrp_candidates_knn(env: VRPEnvironment, args) -> torch.Tensor:
    """Select k nearest feasible nodes (by demand/capacity) and always append depot (-1)."""
    nodes = env.nodes  # (bsz, nb_nodes, dim)
    last = env.last_visited_node  # (bsz, 1, dim)
    bsz, nb_nodes, _ = nodes.shape

    demands = env.full_demands[:, :nb_nodes]
    remain_capacity_vec = (env.true_capacity_vec - env.true_used_capacity_vec).float() / env.capacity
    feasible = (demands > 0) & (demands <= remain_capacity_vec + 1e-6)

    dist = torch.norm(nodes - last, dim=2)
    dist_masked = dist.masked_fill(~feasible, float("inf"))

    k = max(1, min(args.k_promising, nb_nodes))
    neg_dist = -dist_masked
    _, idx = torch.topk(neg_dist, k=k, dim=1)

    # If no feasible nodes remain (need to return depot), fall back to nearest nodes ignoring feasibility.
    feasible_any = feasible.any(dim=1)
    if not feasible_any.all():
        fallback = torch.topk(-dist, k=k, dim=1).indices
        idx = torch.where(feasible_any.view(-1, 1), idx, fallback)

    candidates = torch.cat((idx, env.depot_idx), dim=1)
    return candidates


def rollout_stage1(env: VRPEnvironment, policy: POMOVRPStage1Policy, args, deterministic: bool = False):
    """Rollout Stage1 policy; returns (tours, sum_log_probs)."""
    policy.reset()
    tours = []
    log_probs = []
    while not env.is_finished():
        selected_idx, selected_probs, _ = policy.select_k(env, k_promising=args.k_promising, deterministic=deterministic)
        action = selected_idx[:, 0]
        log_prob = torch.log(selected_probs[:, 0].clamp_min(1e-12))
        env.step(action)
        tours.append(action)
        log_probs.append(log_prob)
    tours_tensor = env.get_tour_tensor(tours)
    sum_log_probs = torch.stack(log_probs, dim=1).sum(dim=1) if log_probs else torch.zeros(env.bsz, device=env.nodes.device)
    return tours_tensor, sum_log_probs


def rollout_stage2(
    env: VRPEnvironment,
    policy: POMOVRPStage2Policy,
    args,
    deterministic: bool = False,
):
    """Rollout Stage2 policy using heuristic KNN candidate selection."""
    policy.reset()
    tours = []
    log_probs = []
    while not env.is_finished():
        candidates = _select_cvrp_candidates_knn(env, args)
        action, logp, _ = policy.select_action(env, selected_global_idx=candidates, deterministic=deterministic)
        env.step(action)
        tours.append(action)
        log_probs.append(logp)
    tours_tensor = env.get_tour_tensor(tours)
    sum_log_probs = torch.stack(log_probs, dim=1).sum(dim=1) if log_probs else torch.zeros(env.bsz, device=env.nodes.device)
    return tours_tensor, sum_log_probs


def build_args():
    config_dict = {
        "problem": "cvrp",
        "stage": "stage1",
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
        "model_lr_pretrain": 2e-5,
        "nb_epochs_pretrain": 200,
        "nb_batch_per_epoch": 300,
        "nb_batch_eval": 20,
        "aug": "mix",
        "aug_num": 16,
        "test_aug_num": 16,
        "data_path": "./",
        "use_swanlab": False,
        "save_dir": "./checkpoints/pomo_vrp_pretrain",
        "resume_ckpt": "",
        "stage1_init_ckpt": "",
        "stage2_init_ckpt": "",
        "k_promising": 8,
    }
    parser, args = create_parser(config_dict)
    args = parser.parse_args(namespace=args)
    args.stage = str(args.stage).lower()

    args.capacity = float(args.capacity)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    args.save_dir = os.path.join(args.save_dir, f'cvrp{args.nb_nodes}', f'model_{timestamp}')
    return args


def train_pretrain_policy(args, device):
    if str(args.problem).lower() != 'cvrp':
        raise ValueError('pomo_cvrp pretraining only supports CVRP.')
    stage = str(args.stage).lower()
    if stage not in ('stage1', 'stage2'):
        raise ValueError("stage must be either 'stage1' or 'stage2'")
    model_kwargs = _model_kwargs(args)
    if stage == 'stage1':
        model = POMOVRPStage1Policy(**model_kwargs).to(device)
        baseline = POMOVRPStage1Policy(**model_kwargs).to(device)
        if args.stage1_init_ckpt:
            load_stage_ckpt(model, args.stage1_init_ckpt, device, expected_stage=None)
    else:
        model = POMOVRPStage2Policy(**model_kwargs).to(device)
        baseline = POMOVRPStage2Policy(**model_kwargs).to(device)
        if args.stage2_init_ckpt:
            load_stage_ckpt(model, args.stage2_init_ckpt, device, expected_stage=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.model_lr_pretrain)
    best_eval_len_m = float('inf')
    best_epoch = -1
    if getattr(args, 'resume_ckpt', ''):
        ckpt = load_stage_ckpt(model, args.resume_ckpt, device, expected_stage='pretrain')
        baseline.load_state_dict(model.state_dict())
        if isinstance(ckpt, dict):
            best_eval_len_m = ckpt.get('best_eval_Lm', best_eval_len_m)
            best_epoch = ckpt.get('best_epoch', best_epoch)
        print(f'Resumed pretraining from checkpoint: {args.resume_ckpt}')
    if getattr(args, 'use_swanlab', False) and _HAS_SWANLAB and hasattr(swanlab, 'init'):
        exp_name = None
        if getattr(args, 'save_dir', None):
            exp_name = os.path.basename(str(args.save_dir).rstrip(os.sep))
        if not exp_name:
            exp_name = f'pomo_cvrp{args.nb_nodes}_pretrain_{stage}'
        swanlab.init(project=f'pomo_cvrp{args.nb_nodes}_pretrain_{stage}', experiment_name=exp_name, config=vars(args))
    tol = 0.0001
    for epoch in range(args.nb_epochs_pretrain):
        model.train()
        loss_sum = 0.0
        len_m_sum = 0.0
        len_b_sum = 0.0
        for _ in range(args.nb_batch_per_epoch):
            (x_aug, x_repeat) = generate_vrp_instance(args, device)
            env = VRPEnvironment(_sanitize_vrp_inputs(x_aug), capacity=args.capacity, problem='cvrp')
            if stage == 'stage1':
                model.reset()
                baseline.reset()
                (tours_m, sum_logp) = rollout_stage1(env, model, args, deterministic=False)
                with torch.no_grad():
                    env_b = VRPEnvironment(_sanitize_vrp_inputs(x_aug), capacity=args.capacity, problem='cvrp')
                    (tours_b, _) = rollout_stage1(env_b, baseline, args, deterministic=True)
                L_m = compute_vrp_tour_length(torch.cat((x_aug['loc'], x_aug['depot'].unsqueeze(1)), dim=1), tours_m)
                L_b = compute_vrp_tour_length(torch.cat((x_aug['loc'], x_aug['depot'].unsqueeze(1)), dim=1), tours_b)
            else:
                model.reset()
                baseline.reset()
                (tours_m, sum_logp) = rollout_stage2(env, model, args, deterministic=False)
                with torch.no_grad():
                    env_b = VRPEnvironment(_sanitize_vrp_inputs(x_aug), capacity=args.capacity, problem='cvrp')
                    (tours_b, _) = rollout_stage2(env_b, baseline, args, deterministic=True)
                L_m = compute_vrp_tour_length(torch.cat((x_aug['loc'], x_aug['depot'].unsqueeze(1)), dim=1), tours_m)
                L_b = compute_vrp_tour_length(torch.cat((x_aug['loc'], x_aug['depot'].unsqueeze(1)), dim=1), tours_b)
            loss = ((L_m - L_b) * sum_logp).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            len_m_sum += L_m.mean().item()
            len_b_sum += L_b.mean().item()
        avg_loss = loss_sum / args.nb_batch_per_epoch
        avg_len_m = len_m_sum / args.nb_batch_per_epoch
        avg_len_b = len_b_sum / args.nb_batch_per_epoch
        model.eval()
        baseline.eval()
        eval_m = 0.0
        eval_b = 0.0
        with torch.no_grad():
            for _ in range(args.nb_batch_eval):
                (x_aug, x_repeat) = generate_vrp_instance(args, device, if_test=True)
                env_m = VRPEnvironment(_sanitize_vrp_inputs(x_aug), capacity=args.capacity, problem='cvrp')
                env_b = VRPEnvironment(_sanitize_vrp_inputs(x_aug), capacity=args.capacity, problem='cvrp')
                if stage == 'stage1':
                    (tours_m, _) = rollout_stage1(env_m, model, args, deterministic=True)
                    (tours_b, _) = rollout_stage1(env_b, baseline, args, deterministic=True)
                else:
                    (tours_m, _) = rollout_stage2(env_m, model, args, deterministic=True)
                    (tours_b, _) = rollout_stage2(env_b, baseline, args, deterministic=True)
                L_m_raw = compute_vrp_tour_length(x_repeat, tours_m)
                L_b_raw = compute_vrp_tour_length(x_repeat, tours_b)
                if args.test_aug_num > 1:
                    L_m = _best_over_augmented(L_m_raw, args.test_aug_num)
                    L_b = _best_over_augmented(L_b_raw, args.test_aug_num)
                else:
                    (L_m, L_b) = (L_m_raw, L_b_raw)
                eval_m += L_m.mean().item()
                eval_b += L_b.mean().item()
        eval_len_m = eval_m / args.nb_batch_eval
        eval_len_b = eval_b / args.nb_batch_eval
        print(f'[POMO Pretrain][Epoch {epoch}] loss={avg_loss:.4f} train Lm={avg_len_m:.4f} Lb={avg_len_b:.4f} eval Lm={eval_len_m:.4f} Lb={eval_len_b:.4f}')
        if eval_len_m < eval_len_b - tol:
            baseline.load_state_dict(model.state_dict())
            best_eval_len_m = eval_len_m
            best_epoch = epoch
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({'stage': 'pretrain', 'policy_state_dict': model.state_dict(), 'args': vars(args), 'best_eval_Lm': best_eval_len_m, 'best_epoch': best_epoch}, os.path.join(args.save_dir, f'pomo_pretrain_{stage}.ckpt'))
        if getattr(args, 'use_swanlab', False) and _HAS_SWANLAB and hasattr(swanlab, 'log'):
            swanlab.log({'epoch': epoch, 'train/loss': avg_loss, 'train/Lm': avg_len_m, 'train/Lb': avg_len_b, 'eval/Lm': eval_len_m, 'eval/Lb': eval_len_b, 'best/eval_Lm': best_eval_len_m, 'best/epoch': best_epoch})
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({
            'stage': 'pretrain', 'policy_state_dict': model.state_dict(),
            'args': vars(args), 'best_eval_Lm': best_eval_len_m,
            'best_epoch': best_epoch,
        }, os.path.join(args.save_dir, f'pomo_pretrain_{stage}_final.ckpt'))
    if getattr(args, 'use_swanlab', False) and _HAS_SWANLAB and hasattr(swanlab, 'finish'):
        swanlab.finish()


def main():
    args = build_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_pretrain_policy(args, device)


if __name__ == '__main__':
    main()
