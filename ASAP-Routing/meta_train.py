"""Distribution-task MAML initialization for INViT/POMO on TSP and CVRP.

This implements Appendix D's task construction with independent support/query
instances. Second-order differentiation is through the sampled REINFORCE
surrogate and inner SGD updates; discrete sampled tours are treated as fixed.
It is a newly supplied training path, not a recovered paper-result checkpoint.
"""

import argparse
import random
from pathlib import Path

import torch
from torch import nn

from maml import DistributionTask, call_with_parameters, maml_task_loss


DISTRIBUTIONS = ("uniform", "clustered", "explosion", "implosion")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("tsp", "cvrp"), default="tsp")
    parser.add_argument("--backbone", choices=("invit", "pomo"), default="invit")
    parser.add_argument("--phase", "--policy-role", dest="phase",
                        choices=("proposal", "selection"), default="proposal",
                        help="Policy role to initialize (not a training phase)")
    parser.add_argument("--task-distributions", nargs="+", choices=DISTRIBUTIONS,
                        default=list(DISTRIBUTIONS))
    parser.add_argument("--tasks-per-step", type=int, default=4)
    parser.add_argument("--meta-steps", type=int, default=1000)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--support-size", type=int, default=64)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument("--inner-lr", type=float, default=1e-4)
    parser.add_argument("--outer-lr", type=float, default=2e-5)
    parser.add_argument("--first-order", action="store_true",
                        help="Use first-order MAML; default differentiates through inner gradients")
    parser.add_argument("--nb-nodes", type=int, default=None,
                        help="Default: TSP-100 or CVRP-50")
    parser.add_argument("--capacity", type=int, default=50,
                        help="CVRP vehicle capacity Q (paper: 50)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/maml"))
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--proposal-ckpt", type=Path,
                        help="Frozen proposal for selection MAML; otherwise use KNN candidates")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--k-promising", type=int, default=8)
    # Defaults match the released training architecture, not a changed model.
    parser.add_argument("--dim-emb", type=int, default=128)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--nb-heads", type=int, default=8)
    parser.add_argument("--nb-layers-action-encoder", type=int, default=2)
    parser.add_argument("--nb-layers-state-encoder", type=int, default=2)
    parser.add_argument("--nb-layers-decoder", type=int, default=3)
    parser.add_argument("--num-state-encoder", type=int, default=1)
    parser.add_argument("--knn-k", type=int, default=25)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--encoder-layer-num", type=int, default=6)
    parser.add_argument("--qkv-dim", type=int, default=16)
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--ff-hidden-dim", type=int, default=512)
    parser.add_argument("--logit-clipping", type=float, default=10.0)
    return parser


def validate_args(args):
    args.nb_nodes = args.nb_nodes or (100 if args.problem == "tsp" else 50)
    positive = ("nb_nodes", "capacity", "meta_steps", "inner_steps", "support_size",
                "query_size", "tasks_per_step", "save_every", "k_promising",
                "inner_lr", "outer_lr", "max_grad_norm", "knn_k")
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.nb_nodes < 3:
        raise ValueError("Use at least three nodes for routing meta-training")
    if args.problem == "cvrp" and args.capacity < 10:
        raise ValueError("CVRP generation uses demands 1..10; capacity must be at least 10")
    args.task_distributions = list(dict.fromkeys(args.task_distributions))
    if args.tasks_per_step > len(args.task_distributions):
        raise ValueError("tasks-per-step exceeds the number of distinct task distributions")
    if args.proposal_ckpt and args.phase != "selection":
        raise ValueError("--proposal-ckpt applies only to --phase selection")
    # Input conventions expected by the released INViT constructors.
    args.dim_input_nodes = 2
    args.batchnorm = False
    args.if_agg_whole_graph = False
    args.if_use_local_mask = False
    return args


def build_policy(args, phase=None):
    phase = phase or args.phase
    if args.backbone == "pomo":
        if args.problem == "tsp":
            from pomo_tsp_policy_asap import POMOTSPStage1Policy, POMOTSPStage2Policy
            cls = POMOTSPStage1Policy if phase == "proposal" else POMOTSPStage2Policy
        else:
            from pomo_cvrp_policy_asap import POMOVRPStage1Policy, POMOVRPStage2Policy
            cls = POMOVRPStage1Policy if phase == "proposal" else POMOVRPStage2Policy
        return cls(embedding_dim=args.embedding_dim, encoder_layer_num=args.encoder_layer_num,
                   qkv_dim=args.qkv_dim, head_num=args.head_num,
                   ff_hidden_dim=args.ff_hidden_dim, logit_clipping=args.logit_clipping,
                   sqrt_embedding_dim=args.embedding_dim ** 0.5,
                   eval_type="sampling")
    if args.problem == "tsp":
        from tsp_policy_asap import TSPStage1Policy, TSPStage2Policy
        return (TSPStage1Policy if phase == "proposal" else TSPStage2Policy)(args)
    from cvrp_policy_asap import VRPStage1Policy, VRPStage2Policy
    return (VRPStage1Policy if phase == "proposal" else VRPStage2Policy)(args)


def generate_batch(args, distribution, size, device):
    """Draw fresh instances from exactly one distribution (one MAML task)."""
    from generator import generate_instances
    generator = getattr(generate_instances, f"generate_{distribution}_{args.problem}_instance")
    # The paper's Clustered task includes the two settings (3,10) and (7,50).
    # Balance these settings within each fresh support/query batch.
    settings = [(3, 10), (7, 50)]
    def draw(index):
        options = {}
        if distribution == "clustered":
            options["cluster_num"], options["cluster_diversity"] = settings[index % 2]
        if args.problem == "cvrp":
            options["capacity"] = args.capacity
        return generator(args.nb_nodes, **options)
    if args.problem == "tsp":
        return torch.stack([draw(index) for index in range(size)]).to(device)
    instances = [draw(index) for index in range(size)]
    return {
        "depot": torch.stack([item[0] for item in instances]).to(device),
        "loc": torch.stack([item[1] for item in instances]).to(device),
        "demand": torch.stack([item[2] for item in instances]).long().to(device),
    }


def sample_tasks(args, device):
    """No support/query tensor is reused; all tasks reset to the shared weights."""
    names = random.sample(args.task_distributions, args.tasks_per_step)
    return [DistributionTask(
        name,
        tuple(generate_batch(args, name, args.support_size, device)
              for _ in range(args.inner_steps)),
        generate_batch(args, name, args.query_size, device),
    ) for name in names]


class RoutingRollout(nn.Module):
    """Make method-based policies callable by PyTorch's functional_call."""

    def __init__(self, policy, args, proposal=None):
        super().__init__()
        self.policy = policy
        self.proposal = proposal
        self.args = args
        if self.proposal is not None:
            self.proposal.requires_grad_(False)
            self.proposal.eval()

    def _reset_caches(self):
        for model in (self.policy, self.proposal):
            if model is not None and hasattr(model, "reset"):
                model.reset()

    def _candidates(self, env, observation):
        if self.proposal is not None:
            proposal_input = observation if self.args.problem == "tsp" and self.args.backbone == "invit" else env
            with torch.no_grad():
                candidates, _, _ = self.proposal.select_k(
                    proposal_input, self.args.k_promising, deterministic=True)
            return candidates
        if self.args.problem == "tsp":
            distances = (env.x - env.last_visited_node).square().sum(-1)
            distances = distances.masked_fill(~env.mask_global, float("inf"))
            k = min(self.args.k_promising, int(env.mask_global.sum(1).min()))
            return distances.topk(k, largest=False).indices
        demand = env.full_demands[:, :env.nb_nodes]
        remaining = (env.true_capacity_vec - env.true_used_capacity_vec).float() / env.capacity
        feasible = (demand > 0) & (demand <= remaining + 1e-6)
        distances = (env.nodes - env.last_visited_node).square().sum(-1)
        distances = distances.masked_fill(~feasible, float("inf"))
        k = min(self.args.k_promising, env.nb_nodes)
        # Both CVRP selectors mask infeasible entries; explicitly include the depot.
        return torch.cat((distances.topk(k, largest=False).indices, env.depot_idx), 1)

    def forward(self, batch, deterministic=False):
        # Cached encodings must never survive a support/query or greedy rollout.
        self._reset_caches()
        try:
            return self._rollout(batch, deterministic)
        finally:
            self._reset_caches()

    def _rollout(self, batch, deterministic):
        from tsp_env import TSPEnvironment
        from vrp_env import VRPEnvironment
        from utils.utils_for_model import compute_tsp_tour_length, compute_vrp_tour_length
        is_tsp = self.args.problem == "tsp"
        if is_tsp:
            # A fixed start gives the sampled/greedy baseline the same initial state.
            start = torch.zeros(batch.shape[0], dtype=torch.long, device=batch.device)
            env = TSPEnvironment(batch, start_idx=start)
        else:
            env = VRPEnvironment(batch, capacity=self.args.capacity, problem="cvrp")
        log_probs, actions = [], []
        limit = env.nb_nodes - 1 if is_tsp else 2 * env.nb_nodes + 2
        for _ in range(limit):
            observation = env.observation() if is_tsp else None
            policy_input = observation if is_tsp and self.args.backbone == "invit" else env
            if self.args.phase == "selection":
                candidates = self._candidates(env, observation)
                action, logp, _ = self.policy.select_action(
                    policy_input, selected_global_idx=candidates, deterministic=deterministic)
            else:
                action, logp, _ = self.policy.select_action(policy_input, deterministic=deterministic)
            log_probs.append(logp)
            actions.append(action)
            if is_tsp:
                _, done = env.step(action)
            else:
                env.step(action)
                done = env.is_finished()
            if done:
                break
        else:
            if not is_tsp:
                raise RuntimeError("CVRP rollout did not serve all demands within 2*n+2 steps")
        if is_tsp:
            lengths = compute_tsp_tour_length(batch, env.get_tour_tensor())
        else:
            coordinates = torch.cat((batch["loc"], batch["depot"].unsqueeze(1)), 1)
            lengths = compute_vrp_tour_length(coordinates, env.get_tour_tensor(actions))
        return lengths, torch.stack(log_probs, 1).sum(1)


def reinforce_loss(module, parameters, buffers, batch):
    """REINFORCE with an action-independent, detached greedy rollout baseline."""
    lengths, log_probs = call_with_parameters(module, parameters, buffers, batch)
    with torch.no_grad():
        baseline, _ = call_with_parameters(module, parameters, buffers, batch, deterministic=True)
    advantage = (lengths - baseline).detach()
    return (advantage * log_probs).mean()


def load_initialization(policy, path, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("policy_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    policy.load_state_dict(state, strict=True)


def save_initialization(policy, optimizer, args, step, path):
    """Use the existing pretrain schema so ASAP's loaders accept this file."""
    options = {key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items()}
    torch.save({
        "stage": "pretrain", "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "args": options,
        "meta_step": step, "meta_method": "fomaml" if args.first_order else "maml",
        "meta_phase": args.phase, "task_distributions": args.task_distributions,
    }, path)


def train(args):
    args = validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    policy = build_policy(args).to(device)
    if args.initial_checkpoint:
        load_initialization(policy, args.initial_checkpoint, device)
    proposal = None
    if args.proposal_ckpt:
        proposal = build_policy(args, "proposal").to(device)
        load_initialization(proposal, args.proposal_ckpt, device)
    model = RoutingRollout(policy, args, proposal=proposal)
    model.policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.outer_lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.meta_steps + 1):
        tasks = sample_tasks(args, device)
        optimizer.zero_grad()
        losses = []
        # Backpropagate one task at a time to release its second-order graph.
        for task in tasks:
            loss = maml_task_loss(model, task, reinforce_loss, args.inner_lr, args.first_order)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite MAML loss for {task.distribution}")
            (loss / len(tasks)).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        print(f"[MAML {step}/{args.meta_steps}] " + " ".join(
            f"{task.distribution}={loss:.6f}" for task, loss in zip(tasks, losses)))
        if step % args.save_every == 0 or step == args.meta_steps:
            save_initialization(policy, optimizer, args, step, args.output_dir / "maml_init.ckpt")
    return args.output_dir / "maml_init.ckpt"


if __name__ == "__main__":
    train(build_parser().parse_args())
