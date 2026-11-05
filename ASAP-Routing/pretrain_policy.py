import os
import time
import torch
from utils.knn import knn

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

def _best_over_augmented(lengths: torch.Tensor, aug_num: int) -> torch.Tensor:
    """Return per-base-instance best lengths across augmented variants."""
    if lengths.numel() % aug_num != 0:
        raise ValueError("Length tensor size must be divisible by aug_num for best-of-aug reduction.")
    base = lengths.numel() // aug_num
    return lengths.view(base, aug_num).min(dim=1).values


def _sanitize_vrp_inputs(x: dict) -> dict:
    """Ensure VRP demands are integer inside the environment."""
    cleaned = dict(x)
    if "demand" in cleaned:
        demand = cleaned["demand"]
        if torch.is_floating_point(demand):
            demand = torch.round(demand)
        cleaned["demand"] = demand.long()
    return cleaned


def rollout_one_stage(env: TSPEnvironment, policy: TSPStage1Policy, args, deterministic: bool = False):
    log_probs = []
    obs = env.observation()
    for _ in range(env.nb_nodes - 1):
        action, logp, _ = policy.select_action(obs, deterministic=deterministic)
        obs, done = env.step(action)
        log_probs.append(logp)
        if done:
            break
    tours = env.get_tour_tensor()
    sum_log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
    return tours, sum_log_probs


def rollout_one_stage_vrp(env: VRPEnvironment, policy: VRPStage1Policy, args, deterministic: bool = False):
    """Rollout for VRP Stage1 policy; returns (tours, sum_log_probs)."""
    tours = []
    log_probs = []
    while not env.is_finished():
        selected_idx, selected_probs, _ = policy.select_k(
            env, k_promising=args.k_promising, deterministic=deterministic
        )
        prob = selected_probs[:, 0].clamp_min(1e-12)
        log_prob = torch.log(prob)
        action = selected_idx[:, 0]
        env.step(action)
        tours.append(action)
        log_probs.append(log_prob)
    sum_log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
    tours_tensor = env.get_tour_tensor(tours)
    return tours_tensor, sum_log_probs


def rollout_one_stage_stage2_tsp(env: TSPEnvironment, policy: TSPStage2Policy, args, deterministic: bool = False):
    """Rollout using Stage2 policy only (build candidates via KNN each step)."""
    log_probs = []
    obs = env.observation()
    for _ in range(env.nb_nodes - 1):
        x = obs["x"]
        last = obs["last_visited_node"]
        mask_global = obs["mask_global"]
        bsz, nb_nodes, _ = x.shape

        all_idx = torch.arange(nb_nodes, device=x.device).repeat((bsz, 1))
        unvisited_matrix = torch.reshape(all_idx[mask_global], (bsz, -1))
        num_nodes = unvisited_matrix.size(1)

        b_graph = torch.arange(0, bsz, device=x.device).repeat(num_nodes).sort()[0]
        graph = x[b_graph, unvisited_matrix.view(-1)].view((bsz, -1, policy.dim_input))

        k_action = min(args.k_promising, num_nodes)
        graph_for_knn = graph.view((-1, policy.dim_input))
        last_for_knn = last.view((-1, policy.dim_input))
        knn_output = knn(graph_for_knn, last_for_knn, k_action, b_graph, torch.arange(bsz, device=x.device))
        action_idx = (knn_output[1, :] % num_nodes).view(bsz, k_action).contiguous()
        selected_idx = unvisited_matrix.gather(1, action_idx)

        action, logp, _ = policy.select_action(obs, selected_global_idx=selected_idx, deterministic=deterministic)
        obs, done = env.step(action)
        log_probs.append(logp)
        if done:
            break
    tours = env.get_tour_tensor()
    sum_log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
    return tours, sum_log_probs


def rollout_one_stage_stage2_vrp(env: VRPEnvironment, policy: VRPStage2Policy, args, deterministic: bool = False):
    """Deprecated placeholder (VRP Stage2 rollout now uses heuristic KNN candidates)."""
    raise NotImplementedError("Use rollout_cvrp_stage2 instead.")


def rollout_cvrp_stage1(env: VRPEnvironment, policy: VRPStage1Policy, args, deterministic: bool = False):
    """Stage1 VRP rollout mirroring cvrp_policy_asap Stage1 behavior."""
    return rollout_one_stage_vrp(env, policy, args, deterministic=deterministic)


@torch.no_grad()
def _select_candidates_with_stage1(env: VRPEnvironment, stage1: VRPStage1Policy, args) -> torch.Tensor:
    """Use Stage1 (deterministic) to propose candidate set for Stage2."""
    selected_idx, _, _ = stage1.select_k(env, k_promising=args.k_promising, deterministic=True)
    return selected_idx


def _select_cvrp_candidates_knn(env: VRPEnvironment, args) -> torch.Tensor:
    """Select k_promising nearest feasible nodes (by demand/capacity); pad with infeasible nodes if needed."""
    nodes = env.nodes  # (bsz, nb_nodes, dim)
    last = env.last_visited_node  # (bsz, 1, dim)
    bsz, nb_nodes, _ = nodes.shape

    demands = env.full_demands[:, :nb_nodes]
    remain_capacity_vec = (env.true_capacity_vec - env.true_used_capacity_vec) / env.capacity
    feasible = (demands > 0) & (demands <= remain_capacity_vec + 1e-6)

    # Compute distances; make infeasible nodes very far so they are selected only if padding.
    dist = torch.norm(nodes - last, dim=2)
    dist_masked = dist.clone()
    dist_masked[~feasible] = float('inf')

    k = max(1, args.k_promising)
    neg_dist = -dist_masked  # topk over negative distances to get nearest
    _, idx = torch.topk(neg_dist, k=k, dim=1)

    return idx


def rollout_cvrp_stage2(
    env: VRPEnvironment,
    stage2: VRPStage2Policy,
    args,
    deterministic: bool = False,
):
    """Rollout VRP Stage2 policy using heuristic KNN candidate selection (no Stage1)."""
    tours = []
    log_probs = []
    while not env.is_finished():
        candidates = _select_cvrp_candidates_knn(env, args)
        action, logp, _ = stage2.select_action(env, selected_global_idx=candidates, deterministic=deterministic)
        env.step(action)
        tours.append(action)
        log_probs.append(logp)
    sum_log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
    tours_tensor = env.get_tour_tensor(tours)
    return tours_tensor, sum_log_probs


def build_args():
    config_dict = {
        'problem': 'tsp',  # tsp or cvrp
        'stage': 'stage1',  # stage1 or stage2
        'bsz': 64,
        'nb_nodes': 50,
        'dim_input_nodes': 2,
        'dim_emb': 128,
        'dim_ff': 512,
        'nb_heads': 8,
        'nb_layers_action_encoder': 2,
        'nb_layers_decoder': 3,
        'batchnorm': False,
        'model_lr_pretrain': 2e-5,
        'nb_epochs_pretrain': 200,
        'nb_batch_per_epoch': 300,
        'nb_batch_eval': 20,
        'aug': 'mix',
        'aug_num': 16,
        'test_aug_num': 16,
        'data_path': './',
        'use_swanlab': False,
        'save_dir': './checkpoints/tsp_pretrain',
        'resume_ckpt': '',
        'use_normalization_layer': False,
        'stage1_ckpt': '',
        'stage1_init_ckpt': '',
        'stage2_init_ckpt': '',
        # VRP-specific
        'num_state_encoder': 1,
        'nb_layers_state_encoder': 2,
        'action_k': 15,
        'knn_k': 25,
        'state_k': '35,50,65',
        'if_use_local_mask': False,
        'if_agg_whole_graph': False,
        'capacity': 50,
        'k_promising': 8,
    }
    parser, args = create_parser(config_dict)
    args = parser.parse_args(namespace=args)
    args.stage = str(args.stage).lower()
    args.capacity = float(args.capacity)
    if isinstance(args.state_k, str):
        args.state_k = [int(value) for value in args.state_k.split(',') if value.strip()]
    if not args.state_k:
        raise ValueError('state_k must contain at least one neighborhood size')
    args.state_k = (args.state_k + args.state_k[-1:] * args.num_state_encoder)[:args.num_state_encoder]
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    args.save_dir = os.path.join(args.save_dir, f'{args.problem}{args.nb_nodes}', f'model_{timestamp}')
    return args


def train_pretrain_policy(args, device):
    problem = str(args.problem).lower()
    if problem not in ('tsp', 'cvrp'):
        raise ValueError('problem must be tsp or cvrp')
    stage1_cls = VRPStage1Policy if problem == 'cvrp' else TSPStage1Policy
    stage2_cls = VRPStage2Policy if problem == 'cvrp' else TSPStage2Policy

    def generate_instances(args, device, if_test=False):
        generator = generate_vrp_instance if problem == 'cvrp' else generate_tsp_instance
        return generator(args, device, if_test=if_test)

    def make_environment(data):
        if problem == 'cvrp':
            return VRPEnvironment(_sanitize_vrp_inputs(data), capacity=args.capacity, problem=problem)
        return TSPEnvironment(data)

    def rollout_proposal(env, policy, deterministic=False):
        rollout = rollout_cvrp_stage1 if problem == 'cvrp' else rollout_one_stage
        return rollout(env, policy, args, deterministic=deterministic)

    def rollout_selection(env, policy, unused_args, deterministic=False):
        rollout = rollout_cvrp_stage2 if problem == 'cvrp' else rollout_one_stage_stage2_tsp
        return rollout(env, policy, args, deterministic=deterministic)

    def tour_length(data, tours):
        if problem == 'cvrp':
            if isinstance(data, dict):
                data = torch.cat((data['loc'], data['depot'].unsqueeze(1)), dim=1)
            return compute_vrp_tour_length(data, tours)
        return compute_tsp_tour_length(data, tours)
    stage = str(args.stage).lower()
    if stage not in ('stage1', 'stage2'):
        raise ValueError("stage must be either 'stage1' or 'stage2'")
    if stage == 'stage1':
        model = stage1_cls(args).to(device)
        baseline = stage1_cls(args).to(device)
        if args.stage1_init_ckpt:
            load_stage_ckpt(model, args.stage1_init_ckpt, device, expected_stage=None)
    else:
        model = stage2_cls(args).to(device)
        baseline = stage2_cls(args).to(device)
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
            exp_name = f'routing{args.nb_nodes}_pretrain_{stage}'
        swanlab.init(project=f'routing{args.nb_nodes}_pretrain_{stage}', experiment_name=exp_name, config=vars(args))
    tol = 0.0001
    for epoch in range(args.nb_epochs_pretrain):
        model.train()
        loss_sum = 0.0
        len_m_sum = 0.0
        len_b_sum = 0.0
        for _ in range(args.nb_batch_per_epoch):
            (x_aug, x_repeat) = generate_instances(args, device)
            env = make_environment(x_aug)
            if stage == 'stage1':
                (tours_m, sum_logp) = rollout_proposal(env, model, deterministic=False)
                with torch.no_grad():
                    env_b = make_environment(x_aug)
                    (tours_b, _) = rollout_proposal(env_b, baseline, deterministic=True)
                L_m = tour_length(x_aug, tours_m)
                L_b = tour_length(x_aug, tours_b)
            else:
                (tours_m, sum_logp) = rollout_selection(env, model, args, deterministic=False)
                with torch.no_grad():
                    env_b = make_environment(x_aug)
                    (tours_b, _) = rollout_selection(env_b, baseline, args, deterministic=True)
                L_m = tour_length(x_aug, tours_m)
                L_b = tour_length(x_aug, tours_b)
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
                (x_aug, x_repeat) = generate_instances(args, device, if_test=True)
                env_m = make_environment(x_aug)
                env_b = make_environment(x_aug)
                if stage == 'stage1':
                    (tours_m, _) = rollout_proposal(env_m, model, deterministic=True)
                    (tours_b, _) = rollout_proposal(env_b, baseline, deterministic=True)
                else:
                    (tours_m, _) = rollout_selection(env_m, model, args, deterministic=True)
                    (tours_b, _) = rollout_selection(env_b, baseline, args, deterministic=True)
                L_m_raw = tour_length(x_repeat, tours_m)
                L_b_raw = tour_length(x_repeat, tours_b)
                if args.test_aug_num > 1:
                    L_m = _best_over_augmented(L_m_raw, args.test_aug_num)
                    L_b = _best_over_augmented(L_b_raw, args.test_aug_num)
                else:
                    (L_m, L_b) = (L_m_raw, L_b_raw)
                eval_m += L_m.mean().item()
                eval_b += L_b.mean().item()
        eval_len_m = eval_m / args.nb_batch_eval
        eval_len_b = eval_b / args.nb_batch_eval
        print(f'[Pretrain][Epoch {epoch}] loss={avg_loss:.4f} train Lm={avg_len_m:.4f} Lb={avg_len_b:.4f} eval Lm={eval_len_m:.4f} Lb={eval_len_b:.4f}')
        if eval_len_m < eval_len_b - tol:
            baseline.load_state_dict(model.state_dict())
            best_eval_len_m = eval_len_m
            best_epoch = epoch
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({'stage': 'pretrain', 'policy_state_dict': model.state_dict(), 'args': vars(args), 'best_eval_Lm': best_eval_len_m, 'best_epoch': best_epoch}, os.path.join(args.save_dir, f'pretrain_{stage}.ckpt'))
        if getattr(args, 'use_swanlab', False) and _HAS_SWANLAB and hasattr(swanlab, 'log'):
            swanlab.log({'epoch': epoch, 'train/loss': avg_loss, 'train/Lm': avg_len_m, 'train/Lb': avg_len_b, 'eval/Lm': eval_len_m, 'eval/Lb': eval_len_b, 'best/eval_Lm': best_eval_len_m, 'best/epoch': best_epoch})
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({
            'stage': 'pretrain', 'policy_state_dict': model.state_dict(),
            'args': vars(args), 'best_eval_Lm': best_eval_len_m,
            'best_epoch': best_epoch,
        }, os.path.join(args.save_dir, f'pretrain_{stage}_final.ckpt'))
    if getattr(args, 'use_swanlab', False) and _HAS_SWANLAB and hasattr(swanlab, 'finish'):
        swanlab.finish()


def main():
    args = build_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_pretrain_policy(args, device)


if __name__ == '__main__':
    main()
