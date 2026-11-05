import os
import time
from typing import Any, Dict, List

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
from pomo_tsp_policy_asap import POMOTSPStage1Policy, POMOTSPStage2Policy
from tsp_env import TSPEnvironment
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


def _normalize_str_list(val) -> List[str]:
    if isinstance(val, str):
        cleaned = val.replace("[", "").replace("]", "")
        return [s.strip().strip("'").strip('"') for s in cleaned.split(",") if s.strip()]
    if isinstance(val, (list, tuple)):
        return [str(s).strip().strip("'").strip('"') for s in val]
    return [str(val).strip()]


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


def _build_args():
    """CLI defaults for finetuning POMO TSP Stage 2 on cross distributions."""
    config = {
        "bsz": 64,
        "nb_nodes": 50,
        "dim_input_nodes": 2,
        "embedding_dim": 128,
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "ff_hidden_dim": 512,
        "logit_clipping": 10.0,
        "eval_type": "sampling",
        "k_promising": 8,
        "model_lr_stage2": 2e-5,
        "nb_epochs": 10,
        "nb_batch_per_epoch": 300,
        "nb_batch_eval": 50,
        "distribution": "uniform",  # uniform | clustered | explosion | implosion
        "save_dir": "./checkpoints/pomo_tsp_stage2_finetune",
        "stage1_ckpt": "",
        "stage2_init_ckpt": "",
        "deterministic_eval": True,
        "use_swanlab": False,
        "data_path": "./data/",
        "eval_mode": "dataset",  # dataset | tsplib
        "eval_sizes": "100",
        "eval_distributions": "uniform",
        "eval_num_instances": -1,
        "aug": "mix",
        "test_aug_num": 16,
        "use_best_over_aug": True,
        "measure_eval_time": False,
    }
    parser, args = create_parser(config)
    args = parser.parse_args(namespace=args)

    args.distribution = _normalize_str_list(getattr(args, "distribution", "uniform"))[0].lower()
    args.train_distributions = [args.distribution]
    if args.distribution == "clustered":
        args.eval_distributions = ["clustered1", "clustered2"]
    else:
        args.eval_distributions = _normalize_str_list(getattr(args, "eval_distributions", args.distribution))
    args.eval_sizes = [int(s) for s in _normalize_str_list(getattr(args, "eval_sizes", "100"))]
    if not args.eval_distributions:
        args.eval_distributions = ["uniform"]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"pomo_tsp{args.nb_nodes}_stage2_finetune_{args.distribution}_{timestamp}"
    args.save_dir = os.path.join(args.save_dir, run_dir)
    return args


def _get_generator(dist_name: str):
    name = dist_name.lower()
    if name == "uniform":
        return generate_uniform_tsp_instance
    if name == "clustered":
        return generate_clustered_tsp_instance
    if name == "explosion":
        return generate_explosion_tsp_instance
    if name == "implosion":
        return generate_implosion_tsp_instance
    raise ValueError(f"Unsupported distribution: {dist_name}")


def _build_batch(args, device: torch.device, dist_name: str) -> torch.Tensor:
    gen_fn = _get_generator(dist_name)
    coords = [gen_fn(args.nb_nodes) for _ in range(args.bsz)]
    return torch.stack(coords).to(device)


def _rollout_lengths(
    args,
    coords: torch.Tensor,
    stage1: POMOTSPStage1Policy,
    stage2: POMOTSPStage2Policy,
    deterministic: bool = True,
) -> torch.Tensor:
    stage1.reset()
    stage2.reset()
    env = TSPEnvironment(coords)
    obs = env.observation()
    for _ in range(env.nb_nodes - 1):
        sel_idx, _, _ = stage1.select_k(env, k_promising=args.k_promising, deterministic=deterministic)
        action, _, _ = stage2.select_action(
            env,
            selected_global_idx=sel_idx,
            deterministic=deterministic,
        )
        obs, done = env.step(action)
        if done:
            break
    return env.get_tour_tensor()


@torch.no_grad()
def _evaluate_model(
    args,
    device: torch.device,
    stage1: POMOTSPStage1Policy,
    stage2: POMOTSPStage2Policy,
):
    """Evaluate two-stage POMO TSP policy on dataset or TSPLIB with augmentation."""
    eval_mode = str(args.eval_mode).lower()
    if eval_mode not in ("dataset", "tsplib"):
        print(f"[POMO-Stage2] Unsupported eval_mode '{args.eval_mode}', skipping final eval.")
        return {}
    eval_start_time = time.time() if getattr(args, "measure_eval_time", False) else None
    stage1.eval()
    stage2.eval()
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
                    print(f"[POMO-Stage2][tsp{size}-{distribution}] No instances found, skipping.")
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
                    tours = _rollout_lengths(args, x_aug, stage1, stage2, deterministic=True)
                    L = compute_tsp_tour_length(x_repeat, tours)
                    base_L = _best_over_augmented(L, args.test_aug_num) if args.use_best_over_aug else \
                        L.view(cur_base, args.test_aug_num).mean(dim=1)
                    gathered.append(base_L.cpu())
                    processed += cur_base

                all_L = torch.cat(gathered)
                avg_len = all_L.mean().item()
                opt_cpu = opt_lens_tensor[:all_L.size(0)].cpu()
                avg_gap = ((all_L - opt_cpu) / opt_cpu).mean().item()
                tag = f"tsp{size}-{distribution}"
                eval_logs[tag] = {"avg_len": avg_len, "avg_gap": avg_gap}
                print(f"[POMO-Stage2-Eval][{tag}] avg_len={avg_len:.4f} avg_gap={avg_gap*100:.3f}% "
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
            tours = _rollout_lengths(args, x_aug, stage1, stage2, deterministic=True)
            L = compute_tsp_tour_length(coords_orig_rep, tours)
            base_L = _best_over_augmented(L, args.test_aug_num) if args.use_best_over_aug else \
                L.view(base_bsz, args.test_aug_num).mean(dim=1)
            best_len = base_L.min().item()
            gap = best_len / opt_len - 1
            tag = f"tsplib-{name}"
            eval_logs[tag] = {"best_len": best_len, "gap": gap}
            print(f"[POMO-Stage2-Eval][TSPLIB][{idx:03d}] {name:12s} size={size:5d} len={best_len:.3f} gap={gap*100:.3f}%")

    if eval_start_time is not None:
        elapsed = time.time() - eval_start_time
        print(f"[POMO-Stage2] Total evaluation time: {elapsed:.2f}s")
    return eval_logs


def _flatten_eval_logs(prefix: str, eval_logs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for tag, vals in eval_logs.items():
        for k, v in vals.items():
            payload[f"{prefix}/{tag}/{k}"] = v
    return payload


def _compute_improvements(
    pre_logs: Dict[str, Dict[str, float]],
    post_logs: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
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


def _train_stage2_step(
    args,
    device: torch.device,
    dist_name: str,
    stage1: POMOTSPStage1Policy,
    stage2: POMOTSPStage2Policy,
    baseline_stage2: POMOTSPStage2Policy,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """One REINFORCE step for Stage 2 with fixed Stage 1 (POMO TSP)."""
    stage2.train()
    stage1.reset(); stage2.reset(); baseline_stage2.reset()

    x_aug = _build_batch(args, device, dist_name)
    env = TSPEnvironment(x_aug)
    obs = env.observation()
    logp_list = []
    for _ in range(env.nb_nodes - 1):
        with torch.no_grad():
            sel_idx, _, _ = stage1.select_k(env, k_promising=args.k_promising, deterministic=True)
        action, logp2, _ = stage2.select_action(
            env,
            selected_global_idx=sel_idx,
            deterministic=False,
        )
        logp_list.append(logp2)
        obs, done = env.step(action)
        if done:
            break
    tours_model = env.get_tour_tensor()
    sum_logp_stage2 = torch.stack(logp_list, dim=1).sum(dim=1)

    with torch.no_grad():
        env_bl = TSPEnvironment(x_aug)
        obs_bl = env_bl.observation()
        for _ in range(env_bl.nb_nodes - 1):
            sel_idx_bl, _, _ = stage1.select_k(env_bl, k_promising=args.k_promising, deterministic=True)
            action_bl, _, _ = baseline_stage2.select_action(
                env_bl,
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

    return {
        "loss": loss.item(),
        "L_model": L_model.mean().item(),
        "L_baseline": L_baseline.mean().item(),
    }


@torch.no_grad()
def _evaluate_distributions(
    args,
    device: torch.device,
    stage1: POMOTSPStage1Policy,
    stage2: POMOTSPStage2Policy,
    distribution: str,
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    lengths = []
    for _ in range(args.nb_batch_eval):
        coords = _build_batch(args, device, distribution)
        tours = _rollout_lengths(args, coords, stage1, stage2, deterministic=args.deterministic_eval)
        L = compute_tsp_tour_length(coords, tours)
        lengths.append(L.mean().item())
    results[distribution] = sum(lengths) / len(lengths)
    return results


def finetune_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.stage1_ckpt:
        raise ValueError("stage1_ckpt is required to finetune Stage 2.")

    model_kwargs = _model_kwargs(args)
    stage1 = POMOTSPStage1Policy(**model_kwargs).to(device)
    stage2 = POMOTSPStage2Policy(**model_kwargs).to(device)
    baseline_stage2 = POMOTSPStage2Policy(**model_kwargs).to(device)

    load_stage_ckpt(stage1, args.stage1_ckpt, device, expected_stage=None)
    if args.stage2_init_ckpt:
        load_stage_ckpt(stage2, args.stage2_init_ckpt, device, expected_stage=None)
    baseline_stage2.load_state_dict(stage2.state_dict())
    for p in stage1.parameters():
        p.requires_grad = False

    optimizer = AdamW(stage2.parameters(), lr=args.model_lr_stage2)

    best_eval = float("inf")
    best_state = None
    os.makedirs(args.save_dir, exist_ok=True)

    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "init"):
        exp_name = os.path.basename(str(args.save_dir).rstrip(os.sep))
        swanlab.init(
            project=f"pomo_tsp{args.nb_nodes}_stage2_finetune_{args.distribution}",
            experiment_name=exp_name,
            config=vars(args),
        )

    pre_eval_logs = _evaluate_model(args, device, stage1, stage2)
    if pre_eval_logs:
        print("[POMO-Stage2-PreEval] Finished evaluation before finetuning.")
    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log") and pre_eval_logs:
        swanlab.log(_flatten_eval_logs("pre_finetune", pre_eval_logs))
    stage2.train()

    for epoch in range(args.nb_epochs):
        epoch_loss = 0.0
        for _ in range(args.nb_batch_per_epoch):
            dist = args.distribution
            metrics = _train_stage2_step(args, device, dist, stage1, stage2, baseline_stage2, optimizer)
            epoch_loss += metrics["loss"]
        avg_loss = epoch_loss / args.nb_batch_per_epoch

        eval_results = _evaluate_distributions(args, device, stage1, stage2, args.distribution)
        print(f"[POMO-Stage2][Epoch {epoch}] loss={avg_loss:.4f} | " +
              " ".join([f"{d}:{l:.4f}" for d, l in eval_results.items()]))

        mean_eval = sum(eval_results.values()) / len(eval_results)
        if mean_eval < best_eval:
            best_eval = mean_eval
            best_state = {k: v.detach().cpu().clone() for k, v in stage2.state_dict().items()}
            torch.save(
                {
                    "stage": "stage2",
                    "policy_state_dict": stage2.state_dict(),
                    "args": vars(args),
                    "best_eval": best_eval,
                },
                os.path.join(args.save_dir, f"stage2_finetune_best_{args.distribution}.ckpt"),
            )
        baseline_stage2.load_state_dict(stage2.state_dict())

        if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log"):
            log_payload = {
                "epoch": epoch,
                "train/loss": avg_loss,
                "eval/mean_len": mean_eval,
            }
            for d, v in eval_results.items():
                log_payload[f"eval/{d}_len"] = v
            swanlab.log(log_payload)

    eval_stage2 = POMOTSPStage2Policy(**model_kwargs).to(device)
    if best_state is not None:
        eval_stage2.load_state_dict(best_state, strict=False)
    else:
        eval_stage2.load_state_dict(stage2.state_dict())
    post_eval_logs = _evaluate_model(args, device, stage1, eval_stage2)

    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log") and post_eval_logs:
        swanlab.log(_flatten_eval_logs("final_eval", post_eval_logs))

    improvements = _compute_improvements(pre_eval_logs, post_eval_logs)
    if improvements:
        for tag, vals in improvements.items():
            print(f"[POMO-Stage2-Improvement][{tag}] " + " ".join([f"{k}:{v:.4f}" for k, v in vals.items()]))
        if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log"):
            swanlab.log(_flatten_eval_logs("improvement", improvements))

    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "finish"):
        swanlab.finish()


def main():
    args = _build_args()
    finetune_stage2(args)


if __name__ == "__main__":
    main()
