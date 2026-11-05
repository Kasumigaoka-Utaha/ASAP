import os
import time
from typing import Dict, List

import torch
from torch.optim import AdamW
try:
    import swanlab
    _HAS_SWANLAB = True
except ImportError:
    swanlab = None  # type: ignore[assignment]
    _HAS_SWANLAB = False

from generator.generate_instances import (
    generate_uniform_tsp_instance,
    generate_clustered_tsp_instance,
    generate_explosion_tsp_instance,
    generate_implosion_tsp_instance,
)
from tsp_env import TSPEnvironment
from tsp_policy_asap import TSPStage1Policy
from utils.utils_for_model import create_parser, compute_tsp_tour_length, load_stage_ckpt, run_aug
from load_data import load_instances_with_baselines
from utils.utilities import (
    choose_bsz,
    normalize_nodes_to_unit_board,
    load_tsplib_file,
    tsplib_collections,
    parse_tsplib_name,
)


def _best_over_augmented(lengths: torch.Tensor, aug_num: int) -> torch.Tensor:
    if lengths.numel() % aug_num != 0:
        raise ValueError("Length tensor size must be divisible by aug_num for best-of-aug reduction.")
    base = lengths.numel() // aug_num
    return lengths.view(base, aug_num).min(dim=1).values


def _normalize_str_list(val):
    """Convert a CLI string/list like 'a,b' or "['a']" to a clean list of strings."""
    if isinstance(val, str):
        cleaned = val.replace('[', '').replace(']', '')
        return [s.strip().strip("'").strip('"') for s in cleaned.split(',') if s.strip()]
    if isinstance(val, (list, tuple)):
        return [str(s).strip().strip("'").strip('"') for s in val]
    return [str(val).strip()]


def _build_args():
    """CLI defaults for finetuning a one-stage (Stage 1) policy on cross distributions."""
    config = {
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
        'nb_epochs': 10,
        'nb_batch_per_epoch': 300,
        'nb_batch_eval': 50,
        'distribution': 'uniform',  # uniform | clustered | explosion | implosion
        'save_dir': './checkpoints/tsp_stage1_finetune',
        'stage1_init_ckpt': '',
        'deterministic_eval': True,
        'use_swanlab': False,
        'data_path': './data/',
        'eval_mode': 'dataset',  # dataset | tsplib
        'eval_sizes': '100',
        'eval_distributions': 'uniform',
        'eval_num_instances': -1,
        'aug': 'mix',
        'test_aug_num': 16,
        'use_best_over_aug': True,
        'measure_eval_time': False,
    }
    parser, args = create_parser(config)
    args = parser.parse_args(namespace=args)
    dist_list = _normalize_str_list(getattr(args, "distribution", "uniform"))
    args.distribution = dist_list[0].lower() if dist_list else "uniform"
    args.train_distributions = [args.distribution]
    if args.distribution == "clustered":
        args.eval_distributions = ['clustered1', 'clustered2']
    else:
        args.eval_distributions = _normalize_str_list(getattr(args, "eval_distributions", args.distribution))
    args.eval_sizes = [int(s) for s in _normalize_str_list(getattr(args, "eval_sizes", "100"))]
    if not args.eval_distributions:
        args.eval_distributions = ['uniform']

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = f"tsp{args.nb_nodes}_stage1_finetune_{args.distribution}_{timestamp}"
    args.save_dir = os.path.join(args.save_dir, run_dir)
    return args


def _get_generator(dist_name: str):
    """Map distribution name to generator callable."""
    name = dist_name.lower()
    if name == 'uniform':
        return generate_uniform_tsp_instance
    if name == 'clustered':
        return generate_clustered_tsp_instance
    if name == 'explosion':
        return generate_explosion_tsp_instance
    if name == 'implosion':
        return generate_implosion_tsp_instance
    raise ValueError(f"Unsupported distribution: {dist_name}")


def _build_batch(args, device: torch.device, dist_name: str) -> torch.Tensor:
    """Generate a batch of TSP coordinates for a given distribution."""
    gen_fn = _get_generator(dist_name)
    coords = [gen_fn(args.nb_nodes) for _ in range(args.bsz)]
    return torch.stack(coords).to(device)


def _rollout_lengths(
    coords: torch.Tensor,
    policy: TSPStage1Policy,
    deterministic: bool = True,
) -> torch.Tensor:
    """Greedy rollout and return tour lengths."""
    env = TSPEnvironment(coords)
    obs = env.observation()
    for _ in range(env.nb_nodes - 1):
        action, _, _ = policy.select_action(obs, deterministic=deterministic)
        obs, done = env.step(action)
        if done:
            break
    tours = env.get_tour_tensor()
    return tours


@torch.no_grad()
def _evaluate_model(args, device, policy: TSPStage1Policy):
    """Evaluate Stage1 on dataset or TSPLIB with augmentation."""
    eval_mode = str(args.eval_mode).lower()
    if eval_mode not in ("dataset", "tsplib"):
        print(f"[Stage1-Finetune] Unsupported eval_mode '{args.eval_mode}', skipping final eval.")
        return {}
    eval_start_time = time.time() if getattr(args, "measure_eval_time", False) else None
    policy.eval()
    eval_logs = {}

    if args.bsz % args.test_aug_num != 0:
        raise ValueError("bsz must be a multiple of test_aug_num for evaluation.")
    base_per_batch = args.bsz // args.test_aug_num

    if eval_mode == "dataset":
        for size in args.eval_sizes:
            for distribution in args.eval_distributions:
                tsp_instances, _, opt_lens = load_instances_with_baselines(args.data_path, "tsp", size, distribution)
                total_available = tsp_instances.size(0)
                total_target = total_available if args.eval_num_instances < 0 else min(args.eval_num_instances, total_available)
                if total_target == 0:
                    print(f"[Stage1-Eval][tsp{size}-{distribution}] No instances found, skipping.")
                    continue

                tsp_instances = tsp_instances[:total_target].float()
                opt_lens_tensor = torch.tensor(opt_lens[:total_target], device=device, dtype=torch.float)
                gathered = []
                processed = 0
                while processed < total_target:
                    cur_base = min(base_per_batch, total_target - processed)
                    coords = tsp_instances[processed:processed + cur_base].to(device)
                    x_repeat = coords.unsqueeze(1).repeat((1, args.test_aug_num, 1, 1)).view(
                        cur_base * args.test_aug_num, size, args.dim_input_nodes
                    )
                    x_aug = run_aug(args.aug, x_repeat, args.test_aug_num)
                    tours = _rollout_lengths(x_aug, policy, deterministic=True)
                    L = compute_tsp_tour_length(x_repeat,tours)
                    if args.use_best_over_aug:
                        base_L = _best_over_augmented(L, args.test_aug_num)
                    else:
                        base_L = L.view(cur_base, args.test_aug_num).mean(dim=1)
                    gathered.append(base_L.cpu())
                    processed += cur_base

                all_L = torch.cat(gathered)
                avg_len = all_L.mean().item()
                opt_cpu = opt_lens_tensor[:all_L.size(0)].cpu()
                avg_gap = ((all_L - opt_cpu) / opt_cpu).mean().item()
                tag = f"tsp{size}-{distribution}"
                eval_logs[tag] = {"avg_len": avg_len, "avg_gap": avg_gap}
                print(f"[Stage1-Eval][{tag}] avg_len={avg_len:.4f} avg_gap={avg_gap*100:.3f}% "
                      f"(best-of-{args.test_aug_num} aug).")
    else:  # tsplib
        names = sorted(tsplib_collections.keys(), key=lambda n: parse_tsplib_name(n)[1])
        for idx, name in enumerate(names):
            opt_len = tsplib_collections[name]
            instance, _ = load_tsplib_file(args.data_path, name)
            size = instance.size(0)
            base_bsz = choose_bsz(size)
            total_bsz = base_bsz * args.test_aug_num

            coords_norm = normalize_nodes_to_unit_board(instance).float()
            coords_norm_rep = coords_norm.unsqueeze(0).repeat((total_bsz, 1, 1)).to(device)
            coords_orig_rep = instance.float().unsqueeze(0).repeat((total_bsz, 1, 1)).to(device)

            x_aug = run_aug(args.aug, coords_norm_rep, args.test_aug_num)
            tours = _rollout_lengths(x_aug, policy, deterministic=True)
            L = compute_tsp_tour_length(coords_orig_rep,tours)
            base_L = _best_over_augmented(L, args.test_aug_num) if args.use_best_over_aug else \
                L.view(base_bsz, args.test_aug_num).mean(dim=1)
            best_len = base_L.min().item()
            gap = best_len / opt_len - 1
            tag = f"tsplib-{name}"
            eval_logs[tag] = {"best_len": best_len, "gap": gap}
            print(f"[Stage1-Eval][TSPLIB][{idx:03d}] {name:12s} size={size:5d} len={best_len:.3f} gap={gap*100:.3f}%")

    if eval_start_time is not None:
        elapsed = time.time() - eval_start_time
        print(f"[Stage1-Eval] Total evaluation time: {elapsed:.2f}s")
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


def _train_stage1_step(
    args,
    device: torch.device,
    dist_name: str,
    policy: TSPStage1Policy,
    baseline: TSPStage1Policy,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """One REINFORCE step for Stage 1 on a chosen distribution."""
    x_aug = _build_batch(args, device, dist_name)
    env = TSPEnvironment(x_aug)
    obs = env.observation()
    logp_list = []
    for _ in range(env.nb_nodes - 1):
        action, logp, _ = policy.select_action(obs, deterministic=False)
        logp_list.append(logp)
        obs, done = env.step(action)
        if done:
            break
    tours_model = env.get_tour_tensor()
    sum_logp = torch.stack(logp_list, dim=1).sum(dim=1)

    with torch.no_grad():
        env_bl = TSPEnvironment(x_aug)
        obs_bl = env_bl.observation()
        for _ in range(env_bl.nb_nodes - 1):
            action_bl, _, _ = baseline.select_action(obs_bl, deterministic=True)
            obs_bl, done_bl = env_bl.step(action_bl)
            if done_bl:
                break
        tours_baseline = env_bl.get_tour_tensor()

    L_model = compute_tsp_tour_length(x_aug, tours_model)
    L_baseline = compute_tsp_tour_length(x_aug, tours_baseline)
    loss = ((L_model - L_baseline) * sum_logp).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "L_model": L_model.mean().item(),
        "L_baseline": L_baseline.mean().item(),
    }


@torch.no_grad()
def _evaluate_distributions(
    args,
    device: torch.device,
    policy: TSPStage1Policy,
    distribution: str,
) -> Dict[str, float]:
    """Evaluate average tour length per distribution (deterministic)."""
    results: Dict[str, float] = {}
    lengths = []
    for _ in range(args.nb_batch_eval):
        coords = _build_batch(args, device, distribution)
        tours = _rollout_lengths(coords, policy, deterministic=args.deterministic_eval)
        L = compute_tsp_tour_length(coords, tours)
        lengths.append(L.mean().item())
    results[distribution] = sum(lengths) / len(lengths)
    return results


def finetune_stage1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = TSPStage1Policy(args).to(device)
    baseline = TSPStage1Policy(args).to(device)
    if args.stage1_init_ckpt:
        load_stage_ckpt(policy, args.stage1_init_ckpt, device, expected_stage='stage1')
    baseline.load_state_dict(policy.state_dict())

    optimizer = AdamW(policy.parameters(), lr=args.model_lr_stage1)

    best_eval = float('inf')
    best_state = None
    os.makedirs(args.save_dir, exist_ok=True)

    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "init"):
        exp_name = os.path.basename(str(args.save_dir).rstrip(os.sep))
        swanlab.init(
            project=f"tsp{args.nb_nodes}_stage1_finetune_{args.distribution}",
            experiment_name=exp_name,
            config=vars(args),
        )

    # Evaluate before finetuning so we can measure the improvement on the same dataset.
    pre_eval_logs = _evaluate_model(args, device, policy)
    if pre_eval_logs:
        print("[Stage1-PreEval] Finished evaluation before finetuning.")
    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log") and pre_eval_logs:
        swanlab.log(_flatten_eval_logs("pre_finetune", pre_eval_logs))
    policy.train()

    for epoch in range(args.nb_epochs):
        epoch_loss = 0.0
        for _ in range(args.nb_batch_per_epoch):
            dist = args.distribution
            metrics = _train_stage1_step(args, device, dist, policy, baseline, optimizer)
            epoch_loss += metrics["loss"]
        avg_loss = epoch_loss / args.nb_batch_per_epoch

        eval_results = _evaluate_distributions(args, device, policy, args.distribution)
        print(f"[Stage1-Finetune][Epoch {epoch}] loss={avg_loss:.4f} | " +
              " ".join([f"{d}:{l:.4f}" for d, l in eval_results.items()]))

        mean_eval = sum(eval_results.values()) / len(eval_results)
        if mean_eval < best_eval:
            best_eval = mean_eval
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            torch.save(
                {
                    "stage": "stage1",
                    "policy_state_dict": policy.state_dict(),
                    "args": vars(args),
                    "best_eval": best_eval,
                },
                os.path.join(args.save_dir, f"stage1_finetune_best_{args.distribution}.ckpt"),
            )
        baseline.load_state_dict(policy.state_dict())

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
    eval_policy = TSPStage1Policy(args).to(device)
    if best_state is not None:
        eval_policy.load_state_dict(best_state, strict=False)
    else:
        eval_policy.load_state_dict(policy.state_dict())
    post_eval_logs = _evaluate_model(args, device, eval_policy)

    # Log final eval metrics to swanlab
    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log") and post_eval_logs:
        swanlab.log(_flatten_eval_logs("final_eval", post_eval_logs))

    improvements = _compute_improvements(pre_eval_logs, post_eval_logs)
    if improvements:
        for tag, vals in improvements.items():
            print(f"[Stage1-Improvement][{tag}] " + " ".join([f"{k}:{v:.4f}" for k, v in vals.items()]))
        if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log"):
            swanlab.log(_flatten_eval_logs("improvement", improvements))

    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "finish"):
        swanlab.finish()


def main():
    args = _build_args()
    finetune_stage1(args)


if __name__ == "__main__":
    main()
