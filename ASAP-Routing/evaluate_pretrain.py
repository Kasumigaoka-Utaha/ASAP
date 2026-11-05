import os
from pathlib import Path
import torch

from load_data import load_instances_with_baselines
from utils.utils_for_model import (
    create_parser,
    compute_tsp_tour_length,
    compute_vrp_tour_length,
    load_stage_ckpt,
    run_aug,
    generate_tsp_instance,
)
from utils.utilities import (
    choose_bsz,
    normalize_nodes_to_unit_board,
    load_tsplib_file,
    tsplib_collections,
    parse_tsplib_name,
    load_cvrplib_file,
    cvrplib_collections,
    parse_cvrplib_name,
)
from tsp_env import TSPEnvironment
from tsp_policy_asap import TSPStage1Policy, TSPStage2Policy
from cvrp_policy_asap import VRPStage1Policy, VRPStage2Policy
from utils.knn import knn
from vrp_env import VRPEnvironment


def _best_over_augmented(lengths: torch.Tensor, aug_num: int) -> torch.Tensor:
    """Return per-base-instance best lengths across augmented variants."""
    if lengths.numel() % aug_num != 0:
        raise ValueError("Length tensor size must be divisible by aug_num for best-of-aug reduction.")
    base = lengths.numel() // aug_num
    return lengths.view(base, aug_num).min(dim=1).values


def _reduce_augmented_lengths(lengths: torch.Tensor, aug_num: int, use_best: bool = True) -> torch.Tensor:
    """Group augmented lengths back to the base-instance dimension."""
    if lengths.numel() % aug_num != 0:
        raise ValueError("Length tensor size must be divisible by aug_num for reduction.")
    grouped = lengths.view(-1, aug_num)
    if use_best:
        return grouped.min(dim=1).values
    return grouped.mean(dim=1)


def rollout_one_stage(env: TSPEnvironment, policy: TSPStage1Policy, deterministic: bool = True):
    """Greedy rollout for Stage 1 policy."""
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


def rollout_stage2_tsp(env: TSPEnvironment, policy: TSPStage2Policy, args, deterministic: bool = True):
    """Rollout using Stage 2 policy alone (candidate set built via KNN each step)."""
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


def rollout_stage1_vrp(env: VRPEnvironment, policy: VRPStage1Policy, args, deterministic: bool = True):
    """Rollout VRP Stage 1 policy (k-promising top-1)."""
    tours = []
    log_probs = []
    while not env.is_finished():
        step_ctx = env.build_step_context(args.action_k, args.state_k, args.if_use_local_mask)
        selected_idx, selected_probs, _ = policy.select_k(env, step_ctx, k_promising=args.k_promising, deterministic=deterministic)
        prob = selected_probs[:, 0].clamp_min(1e-12)
        log_prob = torch.log(prob)
        action = selected_idx[:, 0]
        env.step(action)
        tours.append(action)
        log_probs.append(log_prob)
    sum_log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
    tours_tensor = env.get_tour_tensor(tours)
    return tours_tensor, sum_log_probs


def rollout_stage2_vrp(env: VRPEnvironment, policy: VRPStage2Policy, args, deterministic: bool = True):
    """Rollout VRP Stage 2 policy alone (restrict to feasible nodes)."""
    tours = []
    log_probs = []
    while not env.is_finished():
        step_ctx = env.build_step_context(args.action_k, args.state_k, args.if_use_local_mask)
        candidate_global_idx = policy._build_embeddings(env)[2]
        action, logp, _ = policy.select_action(
            env, step_ctx, selected_global_idx=candidate_global_idx, deterministic=deterministic
        )
        env.step(action)
        tours.append(action)
        log_probs.append(logp)
    sum_log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
    tours_tensor = env.get_tour_tensor(tours)
    return tours_tensor, sum_log_probs


def rollout_tsp(env: TSPEnvironment, model, args, deterministic: bool = True):
    if str(args.stage).lower() == 'stage2':
        return rollout_stage2_tsp(env, model, args, deterministic=deterministic)
    return rollout_one_stage(env, model, deterministic=deterministic)


def rollout_vrp(env: VRPEnvironment, model, args, deterministic: bool = True):
    if str(args.stage).lower() == 'stage2':
        return rollout_stage2_vrp(env, model, args, deterministic=deterministic)
    return rollout_stage1_vrp(env, model, args, deterministic=deterministic)


def build_args():
    """Build CLI args with sensible defaults for pretrained Stage 1 evaluation."""
    config_dict = {
        'problem': 'tsp',  # tsp or cvrp (cvrp only supports cvrplib eval)
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
        'aug': 'mix',
        'aug_num': 16,
        'test_aug_num': 16,
        'data_path': './data/',
        'nb_batch_eval': 100,  # used for synthetic eval mode
        'use_normalization_layer': False,
        'ckpt': '',
        # VRP-only params
        'num_state_encoder': 2,
        'nb_layers_state_encoder': 2,
        'action_k': 15,
        'state_k': '35,50,65',
        'if_use_local_mask': False,
        'if_agg_whole_graph': False,
        'deterministic': True,
        'use_best_over_aug': True,
        'k_promising': 8,
        'eval_mode': 'dataset',  # dataset | synthetic | tsplib | cvrplib
        'eval_sizes': '100',  # comma-separated list
        'eval_distributions': 'uniform',  # comma-separated list
        'eval_num_instances': -1,  # -1 means all available
    }
    parser, args = create_parser(config_dict)
    args = parser.parse_args(namespace=args)
    args.eval_mode = str(args.eval_mode).lower()
    args.stage = str(args.stage).lower()
    args.eval_sizes = [int(s) for s in str(args.eval_sizes).split(',') if str(s).strip()]
    args.eval_distributions = [s.strip() for s in str(args.eval_distributions).split(',') if s.strip()]
    if not args.eval_distributions:
        args.eval_distributions = ['uniform']
    # Parse VRP-specific list arg
    if isinstance(args.state_k, str):
        args.state_k = [int(x) for x in args.state_k.split(',') if x.strip()]
    if len(args.state_k) < args.num_state_encoder:
        args.state_k = (args.state_k + args.state_k[-1:])[:args.num_state_encoder]
    else:
        args.state_k = args.state_k[:args.num_state_encoder]
    return args


def _require_dataset_root(data_path: str) -> None:
    expected = os.path.join(data_path, 'data_farm')
    if not os.path.isdir(expected):
        raise FileNotFoundError(
            f"Dataset directory '{expected}' is missing. Please unzip/download the dataset and point --data_path to its root."
        )


def _load_policy_state(policy: torch.nn.Module, ckpt_path: str, device: torch.device, expected_stage: str = None) -> None:
    """Load checkpoint with a best-effort fallback."""
    try:
        load_stage_ckpt(policy, ckpt_path, device, expected_stage=expected_stage, strict=False)
        return
    except Exception:
        pass
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['policy_state_dict'] if isinstance(ckpt, dict) and 'policy_state_dict' in ckpt else ckpt
    policy.load_state_dict(state_dict, strict=False)


@torch.no_grad()
def _evaluate_on_dataset(args, device, model):
    if args.bsz % args.test_aug_num != 0:
        raise ValueError("bsz must be a multiple of test_aug_num for dataset evaluation.")
    base_per_batch = args.bsz // args.test_aug_num
    _require_dataset_root(args.data_path)
    print(f"[Eval][Dataset] Root: {args.data_path} | Augmentation: {args.aug} | Reduce: {'best' if args.use_best_over_aug else 'mean'}")

    for size in args.eval_sizes:
        for distribution in args.eval_distributions:
            tsp_instances, _, opt_lens = load_instances_with_baselines(args.data_path, "tsp", size, distribution)
            total_available = tsp_instances.size(0)
            total_target = total_available if args.eval_num_instances < 0 else min(args.eval_num_instances, total_available)
            if total_target == 0:
                print(f"[Eval][Dataset][tsp{size}-{distribution}] No instances found, skipping.")
                continue

            tsp_instances = tsp_instances[:total_target].float()
            opt_lens_tensor = torch.tensor(opt_lens[:total_target], device=device, dtype=torch.float)
            gathered_lengths = []
            processed = 0
            while processed < total_target:
                cur_base = min(base_per_batch, total_target - processed)
                coords = tsp_instances[processed:processed + cur_base].to(device)
                x_repeat = coords.unsqueeze(1).repeat((1, args.test_aug_num, 1, 1)).view(
                    cur_base * args.test_aug_num, size, args.dim_input_nodes
                )
                x_aug = run_aug(args.aug, x_repeat, args.test_aug_num)
                env = TSPEnvironment(x_aug)
                tours, _ = rollout_tsp(env, model, args, deterministic=args.deterministic)
                lengths = compute_tsp_tour_length(x_repeat, tours)
                if args.use_best_over_aug:
                    base_lengths = _best_over_augmented(lengths, args.test_aug_num)
                else:
                    base_lengths = _reduce_augmented_lengths(lengths, args.test_aug_num, use_best=False)
                gathered_lengths.append(base_lengths.cpu())
                processed += cur_base

            all_lengths = torch.cat(gathered_lengths)
            avg_len = all_lengths.mean().item()
            opt_cpu = opt_lens_tensor[:all_lengths.size(0)].cpu()
            avg_gap = ((all_lengths - opt_cpu) / opt_cpu).mean().item()
            print(
                f"[Eval][Dataset][tsp{size}-{distribution}] "
                f"avg_len={avg_len:.4f}, avg_gap={avg_gap * 100:.3f}% over {all_lengths.size(0)} instances "
                f"(best-of-{args.test_aug_num} augments)."
            )


@torch.no_grad()
def _evaluate_on_tsplib(args, device, model):
    root = Path(args.data_path)
    names = sorted(tsplib_collections.keys(), key=lambda n: parse_tsplib_name(n)[1])
    buckets = {'1-100': [], '101-1000': [], '1001-10000': [], '>10000': []}

    print(f"[Eval][TSPLIB] Root: {root} | Augmentation: {args.aug} | Reduce: {'best' if args.use_best_over_aug else 'mean'}")
    for idx, name in enumerate(names):
        opt_len = tsplib_collections[name]
        instance, _ = load_tsplib_file(root, name)
        size = instance.size(0)
        base_bsz = choose_bsz(size)
        total_bsz = base_bsz * args.test_aug_num

        coords_norm = normalize_nodes_to_unit_board(instance).float()
        coords_norm_rep = coords_norm.unsqueeze(0).repeat((total_bsz, 1, 1)).to(device)
        coords_orig_rep = instance.float().unsqueeze(0).repeat((total_bsz, 1, 1)).to(device)

        x_aug = run_aug(args.aug, coords_norm_rep, args.test_aug_num)
        env = TSPEnvironment(x_aug)
        tours, _ = rollout_tsp(env, model, args, deterministic=args.deterministic)
        lengths = compute_tsp_tour_length(coords_orig_rep, tours)
        if args.use_best_over_aug:
            base_lengths = _best_over_augmented(lengths, args.test_aug_num)
        else:
            base_lengths = _reduce_augmented_lengths(lengths, args.test_aug_num, use_best=False)
        best_len = base_lengths.min().item()
        gap = best_len / opt_len - 1

        if size <= 100:
            buckets['1-100'].append(gap)
        elif size <= 1000:
            buckets['101-1000'].append(gap)
        elif size <= 10000:
            buckets['1001-10000'].append(gap)
        else:
            buckets['>10000'].append(gap)

        print(f"[Eval][TSPLIB][{idx:03d}] {name:12s} | size={size:5d} | len={best_len:.3f} | gap={gap * 100:.3f}%")

    def _avg(lst): return sum(lst) / len(lst) if lst else float('nan')
    print("[Eval][TSPLIB] Summary gaps (%): "
          f"1-100={_avg(buckets['1-100'])*100:.3f}, "
          f"101-1000={_avg(buckets['101-1000'])*100:.3f}, "
          f"1001-10000={_avg(buckets['1001-10000'])*100:.3f}, "
          f">10000={_avg(buckets['>10000'])*100:.3f}")


@torch.no_grad()
def _evaluate_on_cvrplib(args, device, model):
    root = Path(args.data_path)
    names = sorted(cvrplib_collections.keys(), key=lambda n: parse_cvrplib_name(n)[1])
    buckets = {'1-100': [], '101-200': [], '201-500': [], '>500': []}

    print(f"[Eval][CVRPLIB] Root: {root} | Augmentation: {args.aug} | Reduce: {'best' if args.use_best_over_aug else 'mean'}")
    for idx, name in enumerate(names):
        opt_len = cvrplib_collections[name]
        _, size = parse_cvrplib_name(name)
        depot, nodes, demands, capacity, _ = load_cvrplib_file(root, name)
        coords_orig = torch.cat((nodes, depot.unsqueeze(0)), dim=0).float()

        coords_norm = normalize_nodes_to_unit_board(coords_orig)
        base_bsz = choose_bsz(size)
        total_bsz = base_bsz * args.test_aug_num

        coords_norm_rep = coords_norm.unsqueeze(0).repeat((total_bsz, 1, 1)).to(device)
        coords_orig_rep = coords_orig.unsqueeze(0).repeat((total_bsz, 1, 1)).to(device)
        demand_rep = demands.to(device).unsqueeze(0).repeat((total_bsz, 1))

        x_aug = run_aug(args.aug, coords_norm_rep, args.test_aug_num)
        depot_aug = x_aug[:, -1, :]
        nodes_aug = x_aug[:, :-1, :]
        input_aug = {'loc': nodes_aug, 'demand': demand_rep, 'depot': depot_aug}

        env = VRPEnvironment(input_aug, capacity.item(), problem='cvrp')
        tours, _ = rollout_vrp(env, model, args, deterministic=args.deterministic)
        lengths = compute_vrp_tour_length(coords_orig_rep, tours)
        if args.use_best_over_aug:
            base_lengths = _best_over_augmented(lengths, args.test_aug_num)
        else:
            base_lengths = _reduce_augmented_lengths(lengths, args.test_aug_num, use_best=False)
        best_len = base_lengths.min().item()
        gap = best_len / opt_len - 1

        if size <= 100:
            buckets['1-100'].append(gap)
        elif size <= 200:
            buckets['101-200'].append(gap)
        elif size <= 500:
            buckets['201-500'].append(gap)
        else:
            buckets['>500'].append(gap)

        print(f"[Eval][CVRPLIB][{idx:03d}] {name:12s} | size={size:5d} | len={best_len:.3f} | gap={gap * 100:.3f}%")

    def _avg(lst): return sum(lst) / len(lst) if lst else float('nan')
    print("[Eval][CVRPLIB] Summary gaps (%): "
          f"1-100={_avg(buckets['1-100'])*100:.3f}, "
          f"101-200={_avg(buckets['101-200'])*100:.3f}, "
          f"201-500={_avg(buckets['201-500'])*100:.3f}, "
          f">500={_avg(buckets['>500'])*100:.3f}")


@torch.no_grad()
def _evaluate_on_synthetic(args, device, model):
    total_len = 0.0
    for _ in range(args.nb_batch_eval):
        x_aug, x_repeat = generate_tsp_instance(args, device, if_test=True)
        env = TSPEnvironment(x_aug)
        tours, _ = rollout_tsp(env, model, args, deterministic=args.deterministic)
        lengths = compute_tsp_tour_length(x_repeat, tours)
        if args.use_best_over_aug:
            lengths = _best_over_augmented(lengths, args.test_aug_num)
        total_len += lengths.mean().item()

    avg_len = total_len / args.nb_batch_eval
    print(f"[Eval][Synthetic][Pretrained Stage1] avg tour length over {args.nb_batch_eval} batches: {avg_len:.4f}")


@torch.no_grad()
def evaluate_pretrained(args, device):
    if not args.ckpt:
        raise ValueError("Please provide a pretrained checkpoint path via --ckpt")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    problem = str(args.problem).lower()
    stage = str(args.stage).lower()
    if stage not in ("stage1", "stage2"):
        raise ValueError("stage must be either 'stage1' or 'stage2'")

    if problem == 'cvrp':
        if args.eval_mode != 'cvrplib':
            raise ValueError("For CVRP, only eval_mode='cvrplib' is supported.")
        model_cls = VRPStage1Policy if stage == 'stage1' else VRPStage2Policy
        model = model_cls(args).to(device)
        _load_policy_state(model, args.ckpt, device, expected_stage='pretrain')
        model.eval()
        _evaluate_on_cvrplib(args, device, model)
        return

    # Default: TSP
    model_cls = TSPStage1Policy if stage == 'stage1' else TSPStage2Policy
    model = model_cls(args).to(device)
    _load_policy_state(model, args.ckpt, device, expected_stage='pretrain')
    model.eval()

    if args.eval_mode == 'dataset':
        _evaluate_on_dataset(args, device, model)
    elif args.eval_mode == 'synthetic':
        _evaluate_on_synthetic(args, device, model)
    elif args.eval_mode == 'tsplib':
        _evaluate_on_tsplib(args, device, model)
    elif args.eval_mode == 'cvrplib':
        raise ValueError("Use --problem cvrp with eval_mode cvrplib to evaluate CVRP checkpoints.")
    else:
        raise ValueError("eval_mode must be one of: dataset, synthetic, tsplib, cvrplib")


def main():
    args = build_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_pretrained(args, device)


if __name__ == '__main__':
    main()
