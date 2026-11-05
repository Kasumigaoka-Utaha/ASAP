import os
import time
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
try:
    import swanlab
    _HAS_SWANLAB = True
except ImportError:
    swanlab = None  # type: ignore[assignment]
    _HAS_SWANLAB = False

from generator.generate_instances import (
    generate_uniform_cvrp_instance,
    generate_clustered_cvrp_instance,
    generate_explosion_cvrp_instance,
    generate_implosion_cvrp_instance,
)
from vrp_env import VRPEnvironment
from cvrp_policy_asap import VRPStage1Policy, VRPStage2Policy
from utils.utils_for_model import create_parser, compute_vrp_tour_length, load_stage_ckpt, run_aug
from load_data import load_instances_with_baselines
from utils.utilities import (
    choose_bsz,
    normalize_nodes_to_unit_board,
    load_cvrplib_file,
    cvrplib_collections,
    parse_cvrplib_name,
)


def _best_over_augmented(lengths: torch.Tensor, aug_num: int) -> torch.Tensor:
    if lengths.numel() % aug_num != 0:
        raise ValueError("Length tensor size must be divisible by aug_num for best-of-aug reduction.")
    base = lengths.numel() // aug_num
    return lengths.view(base, aug_num).min(dim=1).values


def _normalize_str_list(val) -> List[str]:
    """Convert CLI string/list like 'a,b' or "['a']" to a clean list of strings."""
    if isinstance(val, str):
        cleaned = val.replace('[', '').replace(']', '')
        return [s.strip().strip("'").strip('"') for s in cleaned.split(',') if s.strip()]
    if isinstance(val, (list, tuple)):
        return [str(s).strip().strip("'").strip('"') for s in val]
    return [str(val).strip()]


def _build_args():
    """CLI defaults for finetuning VRP Stage 1 on cross distributions."""
    config = {
        'problem': 'cvrp',
        'stage': 'stage1',
        'bsz': 64,
        'nb_nodes': 50,
        'dim_input_nodes': 2,
        'dim_emb': 128,
        'dim_ff': 512,
        'nb_heads': 8,
        'nb_layers_action_encoder': 2,
        'nb_layers_state_encoder': 2,
        'nb_layers_decoder': 3,
        'batchnorm': False,
        'num_state_encoder': 1,
        'if_use_local_mask': False,
        'if_agg_whole_graph': False,
        'knn_k': 25,
        'k_promising': 8,
        'model_lr_stage1': 2e-5,
        'nb_epochs': 10,
        'nb_batch_per_epoch': 300,
        'nb_batch_eval': 50,
        'distribution': 'uniform',  # uniform | clustered | explosion | implosion
        'save_dir': './checkpoints/cvrp_stage1_finetune',
        'stage1_init_ckpt': '',
        'stage2_ckpt': '',
        'deterministic_eval': True,
        'use_swanlab': False,
        'data_path': './data/',
        'eval_mode': 'dataset',  # dataset | cvrplib
        'eval_sizes': '50',
        'eval_distributions': 'uniform',
        'eval_num_instances': -1,
        'aug': 'mix',
        'test_aug_num': 16,
        'use_best_over_aug': True,
        'capacity': 50,
        'measure_eval_time': False,
    }
    parser, args = create_parser(config)
    args = parser.parse_args(namespace=args)

    args.capacity = float(args.capacity)
    args.distribution = _normalize_str_list(getattr(args, 'distribution', 'uniform'))[0].lower()
    args.train_distributions = [args.distribution]
    if args.distribution == 'clustered':
        args.eval_distributions = ['clustered1', 'clustered2']
    else:
        args.eval_distributions = _normalize_str_list(getattr(args, 'eval_distributions', args.distribution))
    args.eval_sizes = [int(size) for size in _normalize_str_list(getattr(args, 'eval_sizes', '50'))]
    if not args.eval_distributions:
        args.eval_distributions = ['uniform']
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = f'cvrp{args.nb_nodes}_stage1_finetune_{args.distribution}_{timestamp}'
    args.save_dir = os.path.join(args.save_dir, run_dir)
    return args


def _get_generator(dist_name: str):
    """Map distribution name to generator callable."""
    name = dist_name.lower()
    if name == 'uniform':
        return generate_uniform_cvrp_instance
    if name == 'clustered':
        return generate_clustered_cvrp_instance
    if name == 'explosion':
        return generate_explosion_cvrp_instance
    if name == 'implosion':
        return generate_implosion_cvrp_instance
    raise ValueError(f"Unsupported distribution: {dist_name}")

def _build_batch(args, device, dist_name):
    gen_fn = _get_generator(dist_name)
    locs, depots, demands = [], [], []
    for _ in range(args.bsz):
        depot, nodes, demand, _ = gen_fn(args.nb_nodes, capacity=args.capacity)
        locs.append(nodes)
        depots.append(depot)
        demands.append(demand)
    loc = torch.stack(locs).to(device)
    depot = torch.stack(depots).to(device)
    demand_tensor = torch.stack(demands).to(device).long()
    coords_full = torch.cat((loc, depot.unsqueeze(1)), dim=1)
    return {'loc': loc, 'demand': demand_tensor, 'depot': depot}, coords_full


def _rollout_lengths(args, env_data, stage1, stage2, capacity, deterministic=True):
    # POMO adapters also reuse this evaluator; their embeddings belong to one rollout.
    for policy in (stage1, stage2):
        if hasattr(policy, 'reset'):
            policy.reset()
    env = VRPEnvironment(env_data, capacity=capacity, problem='cvrp')
    tours = []
    while not env.is_finished():
        selected_idx, _, _ = stage1.select_k(env, k_promising=args.k_promising, deterministic=True)
        action, _, _ = stage2.select_action(env, selected_global_idx=selected_idx, deterministic=deterministic)
        env.step(action)
        tours.append(action)
    return env.get_tour_tensor(tours)


@torch.no_grad()
def _evaluate_model(args, device, stage1, stage2):
    eval_mode = str(args.eval_mode).lower()
    if eval_mode not in ('dataset', 'cvrplib'):
        print(f"[VRP-Stage1] Unsupported eval_mode '{args.eval_mode}', skipping eval.")
        return {}
    eval_start_time = time.time() if getattr(args, 'measure_eval_time', False) else None
    stage1.eval()
    stage2.eval()
    eval_logs = {}
    if args.bsz % args.test_aug_num != 0:
        raise ValueError('bsz must be a multiple of test_aug_num for evaluation.')
    base_per_batch = args.bsz // args.test_aug_num
    if eval_mode == 'dataset':
        for size in args.eval_sizes:
            for distribution in args.eval_distributions:
                cvrp_instances, _, opt_lens = load_instances_with_baselines(args.data_path, 'cvrp', size, distribution)
                depot, nodes, demands, capacities = cvrp_instances
                total_available = depot.size(0)
                total_target = total_available if args.eval_num_instances < 0 else min(args.eval_num_instances, total_available)
                if total_target == 0:
                    print(f'[VRP-Stage1][cvrp{size}-{distribution}] No instances found, skipping.')
                    continue
                depot = depot[:total_target].float()
                nodes = nodes[:total_target].float()
                demands = demands[:total_target].long()
                capacities = capacities[:total_target]
                opt_tensor = torch.tensor(opt_lens[:total_target], device=device, dtype=torch.float)
                gathered = []
                processed = 0
                while processed < total_target:
                    cur_base = min(base_per_batch, total_target - processed)
                    depot_slice = depot[processed:processed + cur_base].to(device)
                    nodes_slice = nodes[processed:processed + cur_base].to(device)
                    demand_slice = demands[processed:processed + cur_base].to(device)
                    cap_slice = capacities[processed:processed + cur_base]
                    capacity_val = float(cap_slice[0].item()) if cap_slice.numel() else float(args.capacity)
                    coords = torch.cat((nodes_slice, depot_slice.unsqueeze(1)), dim=1)
                    x_repeat = coords.unsqueeze(1).repeat((1, args.test_aug_num, 1, 1)).view(
                        cur_base * args.test_aug_num, nodes_slice.size(1) + 1, args.dim_input_nodes)
                    x_aug = run_aug(args.aug, x_repeat, args.test_aug_num)
                    demand_rep = demand_slice.unsqueeze(1).repeat((1, args.test_aug_num, 1)).view(
                        cur_base * args.test_aug_num, nodes_slice.size(1))
                    env_input = {'loc': x_aug[:, :-1, :], 'demand': demand_rep, 'depot': x_aug[:, -1, :]}
                    tours = _rollout_lengths(args, env_input, stage1, stage2, capacity_val, deterministic=True)
                    lengths = compute_vrp_tour_length(x_repeat, tours)
                    base_lengths = (_best_over_augmented(lengths, args.test_aug_num) if args.use_best_over_aug
                                    else lengths.view(cur_base, args.test_aug_num).mean(dim=1))
                    gathered.append(base_lengths.cpu())
                    processed += cur_base
                all_lengths = torch.cat(gathered)
                avg_len = all_lengths.mean().item()
                opt_cpu = opt_tensor[:all_lengths.size(0)].cpu()
                avg_gap = ((all_lengths - opt_cpu) / opt_cpu).mean().item()
                tag = f'cvrp{size}-{distribution}'
                eval_logs[tag] = {'avg_len': avg_len, 'avg_gap': avg_gap}
                print(f'[VRP-Stage1-Eval][{tag}] avg_len={avg_len:.4f} avg_gap={avg_gap*100:.3f}% '
                      f'(best-of-{args.test_aug_num} aug).')
    else:
        names = sorted(cvrplib_collections.keys(), key=lambda name: parse_cvrplib_name(name)[1])
        for idx, name in enumerate(names):
            opt_len = cvrplib_collections[name]
            depot, nodes, demands, capacity, _ = load_cvrplib_file(args.data_path, name)
            size = nodes.size(0)
            base_bsz = choose_bsz(size)
            total_bsz = base_bsz * args.test_aug_num
            coords = torch.cat((nodes, depot.unsqueeze(0)), dim=0)
            coords_norm = normalize_nodes_to_unit_board(coords).float()
            coords_norm_rep = coords_norm.unsqueeze(0).repeat((total_bsz, 1, 1)).to(device)
            coords_orig_rep = coords.float().unsqueeze(0).repeat((total_bsz, 1, 1)).to(device)
            demand_rep = demands.long().unsqueeze(0).repeat((total_bsz, 1)).to(device)
            x_aug = run_aug(args.aug, coords_norm_rep, args.test_aug_num)
            env_input = {'loc': x_aug[:, :-1, :], 'demand': demand_rep, 'depot': x_aug[:, -1, :]}
            tours = _rollout_lengths(args, env_input, stage1, stage2, float(capacity.item()), deterministic=True)
            lengths = compute_vrp_tour_length(coords_orig_rep, tours)
            base_lengths = (_best_over_augmented(lengths, args.test_aug_num) if args.use_best_over_aug
                            else lengths.view(base_bsz, args.test_aug_num).mean(dim=1))
            best_len = base_lengths.min().item()
            gap = best_len / opt_len - 1
            tag = f'cvrplib-{name}'
            eval_logs[tag] = {'best_len': best_len, 'gap': gap}
            print(f'[VRP-Stage1-Eval][CVRPLIB][{idx:03d}] {name:12s} size={size:5d} '
                  f'len={best_len:.3f} gap={gap*100:.3f}%')
    if eval_start_time is not None:
        print(f'[VRP-Stage1-Eval] Total evaluation time: {time.time()-eval_start_time:.2f}s')
    return eval_logs


def _flatten_eval_logs(prefix: str, eval_logs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Flatten nested eval logs for swanlab logging."""
    payload: Dict[str, float] = {}
    for tag, vals in eval_logs.items():
        for k, v in vals.items():
            payload[f"{prefix}/{tag}/{k}"] = v
    return payload

def _compute_improvements(
    pre_logs: Dict[str, Dict[str, float]],
    post_logs: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Compute metric deltas (pre - post) for overlapping tags/metrics."""
    improvements: Dict[str, Dict[str, float]] = {}
    if not pre_logs or not post_logs:
        return improvements
    for tag, post_vals in post_logs.items():
        pre_vals = pre_logs.get(tag)
        if not pre_vals:
            continue
        deltas: Dict[str, float] = {}
        for metric, post_v in post_vals.items():
            if metric in pre_vals:
                deltas[metric] = pre_vals[metric] - post_v
        if deltas:
            improvements[tag] = deltas
    return improvements

def _train_stage1_step(args, device, dist_name, stage1, stage2, baseline_stage1, optimizer):
    env_input, coords_full = _build_batch(args, device, dist_name)
    env = VRPEnvironment(env_input, capacity=args.capacity, problem='cvrp')
    sum_logp, tours_model_list = [], []
    while not env.is_finished():
        selected_idx, _, info1 = stage1.select_k(env, k_promising=args.k_promising, deterministic=False)
        with torch.no_grad():
            chosen, _, _ = stage2.select_action(env, selected_global_idx=selected_idx, deterministic=True)
        candidate_global_idx = info1['candidate_global_idx']
        candidate_probs = info1['candidate_probs']
        choice_mask = candidate_global_idx == chosen.unsqueeze(1)
        chosen_prob = (candidate_probs * choice_mask).sum(dim=1)
        sum_logp.append(torch.log(chosen_prob.clamp_min(1e-12)))
        env.step(chosen)
        tours_model_list.append(chosen)
    tours_model = env.get_tour_tensor(tours_model_list)
    sum_logp_stage1 = torch.stack(sum_logp, dim=1).sum(dim=1)
    with torch.no_grad():
        env_bl = VRPEnvironment(env_input, capacity=args.capacity, problem='cvrp')
        tours_bl_list = []
        while not env_bl.is_finished():
            sel_idx_bl, _, _ = baseline_stage1.select_k(env_bl, k_promising=args.k_promising, deterministic=True)
            chosen_bl, _, _ = stage2.select_action(env_bl, selected_global_idx=sel_idx_bl, deterministic=True)
            env_bl.step(chosen_bl)
            tours_bl_list.append(chosen_bl)
        tours_baseline = env_bl.get_tour_tensor(tours_bl_list)
    L_model = compute_vrp_tour_length(coords_full, tours_model)
    L_baseline = compute_vrp_tour_length(coords_full, tours_baseline)
    loss = ((L_model - L_baseline) * sum_logp_stage1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {'loss': loss.item(), 'L_model': L_model.mean().item(), 'L_baseline': L_baseline.mean().item()}


@torch.no_grad()
def _evaluate_distributions(args, device, stage1, stage2, distribution):
    lengths = []
    for _ in range(args.nb_batch_eval):
        env_input, coords_full = _build_batch(args, device, distribution)
        tours = _rollout_lengths(args, env_input, stage1, stage2, args.capacity,
                                 deterministic=args.deterministic_eval)
        lengths.append(compute_vrp_tour_length(coords_full, tours).mean().item())
    return {distribution: sum(lengths) / len(lengths)}


def finetune_stage1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stage1 = VRPStage1Policy(args).to(device)
    stage2 = VRPStage2Policy(args).to(device)
    baseline_stage1 = VRPStage1Policy(args).to(device)

    if args.stage1_init_ckpt:
        load_stage_ckpt(stage1, args.stage1_init_ckpt, device, expected_stage='stage1')
    if args.stage2_ckpt:
        load_stage_ckpt(stage2, args.stage2_ckpt, device, expected_stage='stage2')
    baseline_stage1.load_state_dict(stage1.state_dict())

    for p in stage2.parameters():
        p.requires_grad = False

    stage2.eval()
    optimizer = AdamW(stage1.parameters(), lr=args.model_lr_stage1)

    best_eval = float('inf')
    best_state = None
    os.makedirs(args.save_dir, exist_ok=True)

    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "init"):
        exp_name = os.path.basename(str(args.save_dir).rstrip(os.sep))
        swanlab.init(
            project=f"cvrp{args.nb_nodes}_stage1_finetune_{args.distribution}",
            experiment_name=exp_name,
            config=vars(args),
        )

    # Evaluate before finetuning to establish a baseline on the same evaluation set.
    pre_eval_logs = _evaluate_model(args, device, stage1, stage2)
    if pre_eval_logs:
        print("[VRP-Stage1-PreEval] Finished evaluation before finetuning.")
    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log") and pre_eval_logs:
        swanlab.log(_flatten_eval_logs("pre_finetune", pre_eval_logs))
    stage1.train()

    for epoch in range(args.nb_epochs):
        epoch_loss = 0.0
        for _ in range(args.nb_batch_per_epoch):
            dist = args.distribution
            metrics = _train_stage1_step(args, device, dist, stage1, stage2, baseline_stage1, optimizer)
            epoch_loss += metrics["loss"]
        avg_loss = epoch_loss / args.nb_batch_per_epoch

        eval_results = _evaluate_distributions(args, device, stage1, stage2, args.distribution)
        print(f"[Epoch {epoch}] loss={avg_loss:.4f} | " +
              " ".join([f"{d}:{l:.4f}" for d, l in eval_results.items()]))

        mean_eval = sum(eval_results.values()) / len(eval_results)
        if mean_eval < best_eval:
            best_eval = mean_eval
            best_state = {k: v.detach().cpu().clone() for k, v in stage1.state_dict().items()}
            torch.save(
                {
                    "stage": "stage1",
                    "policy_state_dict": stage1.state_dict(),
                    "args": vars(args),
                    "best_eval": best_eval,
                },
                os.path.join(args.save_dir, f"stage1_finetune_best_{args.distribution}.ckpt"),
            )
        # Update EMA baseline if improved on aggregate
        baseline_stage1.load_state_dict(stage1.state_dict())

        if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log"):
            log_payload = {
                "epoch": epoch,
                "train/loss": avg_loss,
                "eval/mean_len": mean_eval,
            }
            for d, v in eval_results.items():
                log_payload[f"eval/{d}_len"] = v
            swanlab.log(log_payload)

    # Final evaluation using the best checkpoint
    eval_stage1 = VRPStage1Policy(args).to(device)
    if best_state is not None:
        eval_stage1.load_state_dict(best_state, strict=False)
    else:
        eval_stage1.load_state_dict(stage1.state_dict())
    post_eval_logs = _evaluate_model(args, device, eval_stage1, stage2)

    # Log final eval metrics to swanlab
    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log") and post_eval_logs:
        swanlab.log(_flatten_eval_logs("final_eval", post_eval_logs))

    improvements = _compute_improvements(pre_eval_logs, post_eval_logs)
    if improvements:
        for tag, vals in improvements.items():
            print(f"[VRP-Stage1-Improvement][{tag}] " + " ".join([f"{k}:{v:.4f}" for k, v in vals.items()]))
        if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log"):
            swanlab.log(_flatten_eval_logs("improvement", improvements))

    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "finish"):
        swanlab.finish()

def main():
    args = _build_args()
    finetune_stage1(args)


if __name__ == '__main__':
    main()
