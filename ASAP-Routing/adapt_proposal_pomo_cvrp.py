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
    generate_uniform_cvrp_instance,
    generate_clustered_cvrp_instance,
    generate_explosion_cvrp_instance,
    generate_implosion_cvrp_instance,
)
from pomo_cvrp_policy_asap import POMOVRPStage1Policy, POMOVRPStage2Policy
from vrp_env import VRPEnvironment
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
    if isinstance(val, str):
        cleaned = val.replace('[', '').replace(']', '')
        return [s.strip().strip("'").strip('"') for s in cleaned.split(',') if s.strip()]
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
    """CLI defaults for finetuning POMO CVRP Stage 1 on cross distributions."""
    config = {
        "bsz": 64,
        "capacity": 50.0,
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
        "model_lr_stage1": 2e-5,
        "nb_epochs": 10,
        "nb_batch_per_epoch": 300,
        "nb_batch_eval": 50,
        "distribution": "uniform",  # uniform | clustered | explosion | implosion
        "save_dir": "./checkpoints/pomo_cvrp_stage1_finetune",
        "stage1_init_ckpt": "",
        "stage2_ckpt": "",
        "deterministic_eval": True,
        "use_swanlab": False,
        "data_path": "./data/",
        "eval_mode": "dataset",  # dataset | cvrplib
        "eval_sizes": "50",
        "eval_distributions": "uniform",
        "eval_num_instances": -1,
        "aug": "mix",
        "test_aug_num": 16,
        "use_best_over_aug": True,
        "measure_eval_time": False,
    }
    parser, args = create_parser(config)
    args = parser.parse_args(namespace=args)
    args.capacity = float(args.capacity)

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
    run_dir = f"pomo_cvrp{args.nb_nodes}_stage1_finetune_{args.distribution}_{timestamp}"
    args.save_dir = os.path.join(args.save_dir, run_dir)
    return args


def _get_generator(dist_name: str):
    name = dist_name.lower()
    if name == "uniform":
        return generate_uniform_cvrp_instance
    if name == "clustered":
        return generate_clustered_cvrp_instance
    if name == "explosion":
        return generate_explosion_cvrp_instance
    if name == "implosion":
        return generate_implosion_cvrp_instance
    raise ValueError(f"Unsupported distribution: {dist_name}")


def _build_batch(args, device, dist_name):
    generator = _get_generator(dist_name)
    instances = [generator(args.nb_nodes, capacity=args.capacity) for _ in range(args.bsz)]
    inputs = {
        'loc': torch.stack([item[1] for item in instances]).to(device),
        'depot': torch.stack([item[0] for item in instances]).to(device),
        'demand': torch.stack([item[2] for item in instances]).long().to(device),
    }
    return inputs, torch.cat((inputs['loc'], inputs['depot'].unsqueeze(1)), 1)



def _rollout_lengths(args, env_data, stage1, stage2, capacity, deterministic=True):
    stage1.reset()
    stage2.reset()
    env = VRPEnvironment(env_data, capacity=capacity, problem='cvrp')
    tours = []
    for _ in range(2 * env.nb_nodes + 2):
        if env.is_finished():
            return env.get_tour_tensor(tours)
        candidates, _, _ = stage1.select_k(env, k_promising=args.k_promising, deterministic=deterministic)
        action, _, _ = stage2.select_action(env, selected_global_idx=candidates, deterministic=deterministic)
        env.step(action)
        tours.append(action)
    raise RuntimeError('CVRP rollout exceeded 2*n+2 steps')



@torch.no_grad()
def _evaluate_model(args, device, stage1, stage2):
    from adapt_selection_cvrp import _evaluate_model as evaluate_cvrp
    return evaluate_cvrp(args, device, stage1, stage2)



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


def _train_stage1_step(args, device, dist_name, stage1, stage2, baseline_stage1, optimizer):
    stage1.train()
    stage1.reset(); stage2.reset(); baseline_stage1.reset()
    inputs, coords = _build_batch(args, device, dist_name)
    env = VRPEnvironment(inputs, capacity=args.capacity, problem='cvrp')
    actions, log_probs = [], []
    for _ in range(2 * env.nb_nodes + 2):
        if env.is_finished():
            break
        candidates, probabilities, _ = stage1.select_k(env, k_promising=args.k_promising, deterministic=False)
        with torch.no_grad():
            action, _, _ = stage2.select_action(env, selected_global_idx=candidates, deterministic=True)
        chosen_probability = (probabilities * (candidates == action.unsqueeze(1))).sum(1).clamp_min(1e-12)
        log_probs.append(chosen_probability.log()); actions.append(action)
        env.step(action)
    else:
        raise RuntimeError('CVRP training rollout exceeded 2*n+2 steps')
    tours = env.get_tour_tensor(actions)
    with torch.no_grad():
        baseline_tours = _rollout_lengths(args, inputs, baseline_stage1, stage2, args.capacity, True)
    length = compute_vrp_tour_length(coords, tours)
    baseline = compute_vrp_tour_length(coords, baseline_tours)
    loss = ((length - baseline).detach() * torch.stack(log_probs, 1).sum(1)).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {'loss': loss.item(), 'L_model': length.mean().item(), 'L_baseline': baseline.mean().item()}



@torch.no_grad()
def _evaluate_distributions(args, device, stage1, stage2, distribution):
    stage1.eval()
    stage2.eval()
    lengths = []
    for _ in range(args.nb_batch_eval):
        inputs, coords = _build_batch(args, device, distribution)
        tours = _rollout_lengths(args, inputs, stage1, stage2, args.capacity, args.deterministic_eval)
        lengths.append(compute_vrp_tour_length(coords, tours).mean().item())
    return {distribution: sum(lengths) / len(lengths)}



def finetune_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = _model_kwargs(args)
    stage1 = POMOVRPStage1Policy(**model_kwargs).to(device)
    baseline_stage1 = POMOVRPStage1Policy(**model_kwargs).to(device)
    stage2_fixed = POMOVRPStage2Policy(**model_kwargs).to(device)
    if args.stage1_init_ckpt:
        load_stage_ckpt(stage1, args.stage1_init_ckpt, device, expected_stage=None)
    if args.stage2_ckpt:
        load_stage_ckpt(stage2_fixed, args.stage2_ckpt, device, expected_stage=None)
    baseline_stage1.load_state_dict(stage1.state_dict())
    for p in stage2_fixed.parameters():
        p.requires_grad = False
    stage2_fixed.eval()

    optimizer = AdamW(stage1.parameters(), lr=args.model_lr_stage1)

    best_eval = float("inf")
    best_state = None
    os.makedirs(args.save_dir, exist_ok=True)

    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "init"):
        exp_name = os.path.basename(str(args.save_dir).rstrip(os.sep))
        swanlab.init(
            project=f"pomo_cvrp{args.nb_nodes}_stage1_finetune_{args.distribution}",
            experiment_name=exp_name,
            config=vars(args),
        )

    pre_eval_logs = _evaluate_model(args, device, stage1, stage2_fixed)
    if pre_eval_logs:
        print("[POMO-Stage1-PreEval] Finished evaluation before finetuning.")
    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log") and pre_eval_logs:
        swanlab.log(_flatten_eval_logs("pre_finetune", pre_eval_logs))
    stage1.train()

    for epoch in range(args.nb_epochs):
        epoch_loss = 0.0
        for _ in range(args.nb_batch_per_epoch):
            dist = args.distribution
            metrics = _train_stage1_step(args, device, dist, stage1, stage2_fixed, baseline_stage1, optimizer)
            epoch_loss += metrics["loss"]
        avg_loss = epoch_loss / args.nb_batch_per_epoch

        eval_results = _evaluate_distributions(args, device, stage1, stage2_fixed, args.distribution)
        print(f"[POMO-Stage1][Epoch {epoch}] loss={avg_loss:.4f} | " +
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

    eval_stage1 = POMOVRPStage1Policy(**model_kwargs).to(device)
    if best_state is not None:
        eval_stage1.load_state_dict(best_state, strict=False)
    else:
        eval_stage1.load_state_dict(stage1.state_dict())
    post_eval_logs = _evaluate_model(args, device, eval_stage1, stage2_fixed)

    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log") and post_eval_logs:
        swanlab.log(_flatten_eval_logs("final_eval", post_eval_logs))

    improvements = _compute_improvements(pre_eval_logs, post_eval_logs)
    if improvements:
        for tag, vals in improvements.items():
            print(f"[POMO-Stage1-Improvement][{tag}] " + " ".join([f"{k}:{v:.4f}" for k, v in vals.items()]))
        if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "log"):
            swanlab.log(_flatten_eval_logs("improvement", improvements))

    if args.use_swanlab and _HAS_SWANLAB and hasattr(swanlab, "finish"):
        swanlab.finish()


def main():
    args = _build_args()
    finetune_stage1(args)


if __name__ == "__main__":
    main()
