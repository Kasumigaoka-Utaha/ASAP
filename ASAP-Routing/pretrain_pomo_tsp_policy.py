import os
import time
import torch
from utils.knn import knn

from utils.utils_for_model import (
    create_parser,
    generate_tsp_instance,
    compute_tsp_tour_length,
    load_stage_ckpt,
)
from pomo_tsp_policy_asap import POMOTSPStage1Policy, POMOTSPStage2Policy
from tsp_env import TSPEnvironment

try:
    import swanlab
    _HAS_SWANLAB = True
except ImportError:
    swanlab = None  # type: ignore[assignment]
    _HAS_SWANLAB = False


def _best_over_augmented(lengths: torch.Tensor, aug_num: int) -> torch.Tensor:
    if lengths.numel() % aug_num != 0:
        raise ValueError("Length tensor size must be divisible by aug_num for best-of-aug reduction.")
    base = lengths.numel() // aug_num
    return lengths.view(base, aug_num).min(dim=1).values


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


def rollout_stage1(env: TSPEnvironment, policy: POMOTSPStage1Policy, deterministic: bool = False):
    log_probs = []
    for _ in range(env.nb_nodes - 1):
        obs = env.observation()
        action, logp, _ = policy.select_action(env, deterministic=deterministic)
        obs, done = env.step(action)
        log_probs.append(logp)
        if done:
            break
    print(log_probs)
    tours = env.get_tour_tensor()
    sum_log_probs = torch.stack(log_probs, dim=1).sum(dim=1) if log_probs else torch.zeros(env.bsz, device=env.device)
    return tours, sum_log_probs


def _build_candidates(obs: dict, k_promising: int, dim_input: int) -> torch.Tensor:
    x: torch.Tensor = obs["x"]
    last: torch.Tensor = obs["last_visited_node"]
    mask_global: torch.Tensor = obs["mask_global"]

    bsz, nb_nodes, _ = x.shape
    all_idx = torch.arange(nb_nodes, device=x.device).repeat((bsz, 1))
    unvisited_matrix = torch.reshape(all_idx[mask_global], (bsz, -1))
    num_nodes = unvisited_matrix.size(1)
    k_action = max(1, min(k_promising, num_nodes))

    b_graph = torch.arange(0, bsz, device=x.device).repeat(num_nodes).sort()[0]
    graph = x[b_graph, unvisited_matrix.view(-1)].view((bsz, -1, dim_input))

    graph_for_knn = graph.view((-1, dim_input))
    last_for_knn = last.view((-1, dim_input))
    knn_out = knn(graph_for_knn, last_for_knn, k_action, b_graph, torch.arange(bsz, device=x.device))
    action_idx = (knn_out[1, :] % num_nodes).view(bsz, k_action).contiguous()
    selected_idx = unvisited_matrix.gather(1, action_idx)
    return selected_idx


def rollout_stage2(env: TSPEnvironment, policy: POMOTSPStage2Policy, args, deterministic: bool = False):
    log_probs = []
    for _ in range(env.nb_nodes - 1):
        obs = env.observation()
        candidates = _build_candidates(obs, args.k_promising, env.dim_input)
        action, logp, _ = policy.select_action(env, selected_global_idx=candidates, deterministic=deterministic)
        # print(logp)
        obs, done = env.step(action)
        log_probs.append(logp)
        if done:
            break
    # print(log_probs)
    tours = env.get_tour_tensor()
    sum_log_probs = torch.stack(log_probs, dim=1).sum(dim=1) if log_probs else torch.zeros(env.bsz, device=env.device)
    return tours, sum_log_probs


def build_args():
    config_dict = {
        "problem": "tsp",
        "stage": "stage1",
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
        "model_lr_pretrain": 2e-5,
        "nb_epochs_pretrain": 200,
        "nb_batch_per_epoch": 300,
        "nb_batch_eval": 20,
        "aug": "mix",
        "aug_num": 16,
        "test_aug_num": 16,
        "data_path": "./",
        "use_swanlab": False,
        "save_dir": "./checkpoints/pomo_tsp_pretrain",
        "resume_ckpt": "",
        "stage1_init_ckpt": "",
        "stage2_init_ckpt": "",
        "k_promising": 8,
    }
    parser, args = create_parser(config_dict)
    args = parser.parse_args(namespace=args)
    args.stage = str(args.stage).lower()

    base_dir = args.save_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"tsp{args.nb_nodes}/model_{timestamp}"
    args.save_dir = os.path.join(base_dir, run_dir_name)
    return args


def train_pretrain_policy(args, device):
    if str(args.problem).lower() != "tsp":
        raise ValueError("pomo_tsp pretraining only supports TSP.")
    stage = str(args.stage).lower()
    if stage not in ("stage1", "stage2"):
        raise ValueError("stage must be either 'stage1' or 'stage2'")

    model_kwargs = _model_kwargs(args)
    if stage == "stage1":
        model = POMOTSPStage1Policy(**model_kwargs).to(device)
        baseline = POMOTSPStage1Policy(**model_kwargs).to(device)
        if args.stage1_init_ckpt:
            load_stage_ckpt(model, args.stage1_init_ckpt, device, expected_stage=None)
    else:
        model = POMOTSPStage2Policy(**model_kwargs).to(device)
        baseline = POMOTSPStage2Policy(**model_kwargs).to(device)
        if args.stage2_init_ckpt:
            load_stage_ckpt(model, args.stage2_init_ckpt, device, expected_stage=None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.model_lr_pretrain)
    best_eval_len_m = float("inf")
    best_epoch = -1

    if getattr(args, "resume_ckpt", ""):
        ckpt = load_stage_ckpt(model, args.resume_ckpt, device, expected_stage="pretrain")
        baseline.load_state_dict(model.state_dict())
        if isinstance(ckpt, dict):
            best_eval_len_m = ckpt.get("best_eval_Lm", best_eval_len_m)
            best_epoch = ckpt.get("best_epoch", best_epoch)
        print(f"Resumed pretraining from checkpoint: {args.resume_ckpt}")

    if getattr(args, "use_swanlab", False) and _HAS_SWANLAB and hasattr(swanlab, "init"):
        exp_name = None
        if getattr(args, "save_dir", None):
            exp_name = os.path.basename(str(args.save_dir).rstrip(os.sep))
        if not exp_name:
            exp_name = f"pomo_tsp{args.nb_nodes}_pretrain_{stage}"
        swanlab.init(
            project=f"pomo_tsp{args.nb_nodes}_pretrain_{stage}",
            experiment_name=exp_name,
            config=vars(args),
        )

    tol = 1e-4
    for epoch in range(args.nb_epochs_pretrain):
        model.train()
        loss_sum = 0.0
        len_m_sum = 0.0
        len_b_sum = 0.0
        for _ in range(args.nb_batch_per_epoch):
            x_aug, x_repeat = generate_tsp_instance(args, device)
            env = TSPEnvironment(x_aug)

            if stage == "stage1":
                model.reset(); baseline.reset()
                tours_m, sum_logp = rollout_stage1(env, model, deterministic=False)
                with torch.no_grad():
                    env_b = TSPEnvironment(x_aug)
                    tours_b, _ = rollout_stage1(env_b, baseline, deterministic=True)
                L_m = compute_tsp_tour_length(x_aug, tours_m)
                L_b = compute_tsp_tour_length(x_aug, tours_b)
            else:
                model.reset(); baseline.reset()
                tours_m, sum_logp = rollout_stage2(env, model, args, deterministic=False)
                with torch.no_grad():
                    env_b = TSPEnvironment(x_aug)
                    tours_b, _ = rollout_stage2(env_b, baseline, args, deterministic=True)
                L_m = compute_tsp_tour_length(x_aug, tours_m)
                L_b = compute_tsp_tour_length(x_aug, tours_b)

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

        model.eval(); baseline.eval()
        eval_m = 0.0
        eval_b = 0.0
        with torch.no_grad():
            for _ in range(args.nb_batch_eval):
                x_aug, x_repeat = generate_tsp_instance(args, device, if_test=True)
                env_m = TSPEnvironment(x_aug)
                env_b = TSPEnvironment(x_aug)
                if stage == "stage1":
                    tours_m, _ = rollout_stage1(env_m, model, deterministic=True)
                    tours_b, _ = rollout_stage1(env_b, baseline, deterministic=True)
                else:
                    tours_m, _ = rollout_stage2(env_m, model, args, deterministic=True)
                    tours_b, _ = rollout_stage2(env_b, baseline, args, deterministic=True)

                L_m_raw = compute_tsp_tour_length(x_repeat, tours_m)
                L_b_raw = compute_tsp_tour_length(x_repeat, tours_b)
                if args.test_aug_num > 1:
                    L_m = _best_over_augmented(L_m_raw, args.test_aug_num)
                    L_b = _best_over_augmented(L_b_raw, args.test_aug_num)
                else:
                    L_m, L_b = L_m_raw, L_b_raw
                eval_m += L_m.mean().item()
                eval_b += L_b.mean().item()

        eval_len_m = eval_m / args.nb_batch_eval
        eval_len_b = eval_b / args.nb_batch_eval

        print(f"[POMO Pretrain][Epoch {epoch}] loss={avg_loss:.4f} "
              f"train Lm={avg_len_m:.4f} Lb={avg_len_b:.4f} "
              f"eval Lm={eval_len_m:.4f} Lb={eval_len_b:.4f}")

        if eval_len_m < eval_len_b - tol:
            baseline.load_state_dict(model.state_dict())
            best_eval_len_m = eval_len_m
            best_epoch = epoch
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(
                    {
                        "stage": "pretrain",
                        "policy_state_dict": model.state_dict(),
                        "args": vars(args),
                        "best_eval_Lm": best_eval_len_m,
                        "best_epoch": best_epoch,
                    },
                    os.path.join(args.save_dir, f"pomo_pretrain_{stage}.ckpt"),
                )

        if getattr(args, "use_swanlab", False) and _HAS_SWANLAB and hasattr(swanlab, "log"):
            swanlab.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/Lm": avg_len_m,
                "train/Lb": avg_len_b,
                "eval/Lm": eval_len_m,
                "eval/Lb": eval_len_b,
                "best/eval_Lm": best_eval_len_m,
                "best/epoch": best_epoch,
            })

    if getattr(args, "use_swanlab", False) and _HAS_SWANLAB and hasattr(swanlab, "finish"):
        swanlab.finish()


def main():
    args = build_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pretrain_policy(args, device)


if __name__ == "__main__":
    main()
