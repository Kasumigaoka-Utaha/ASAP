import torch
from typing import Any, Dict, Tuple

from pomo_vrp_model import POMOVRPModel, _get_encoding
from vrp_env import VRPEnvironment


class POMOVRPStage1Policy(POMOVRPModel):
    """Stage 1 POMO VRP policy that can return top-k candidate actions."""

    def select_k(
        self,
        env: VRPEnvironment,
        k_promising: int,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.encoded_nodes is None or self._env_cache_key != id(env):
            self._prepare_from_env(env)

        ninf_mask, load = self._build_masks_and_load(env)

        last_idx = env.last_visited_idx
        gather_idx = last_idx + 1  # depot -> 0, node i -> i+1
        encoded_last = _get_encoding(self.encoded_nodes, gather_idx)

        probs = self.decoder(encoded_last, load=load, ninf_mask=ninf_mask)  # (bsz, 1, problem+1)
        probs = probs[:, 0, :]

        available_mask = torch.isfinite(ninf_mask.squeeze(1))
        max_available = int(available_mask.sum(dim=1).min().clamp(min=1).item())
        k = max(1, min(k_promising, max_available))

        if deterministic or self.model_params.get("eval_type") == "argmax":
            selected_probs, selected = probs.topk(k, dim=1)
            method = "topk"
        else:
            selected = probs.multinomial(num_samples=k, replacement=False)
            selected_probs = probs.gather(1, selected)
            method = "sample"

        action_idx = torch.where(selected == 0, torch.full_like(selected, -1), selected - 1)
        info: Dict[str, Any] = {"probs": probs, "ninf_mask": ninf_mask.squeeze(1), "method": method}
        return action_idx, selected_probs, info


class POMOVRPStage2Policy(POMOVRPModel):
    """Stage 2 POMO VRP policy that restricts decoding to Stage 1 candidates."""

    def _candidate_ninf_mask(
        self, base_ninf_mask: torch.Tensor, selected_global_idx: torch.Tensor, nb_nodes: int
    ) -> torch.Tensor:
        """Build -inf mask allowing only selected candidates that are also feasible."""
        device = base_ninf_mask.device
        bsz = base_ninf_mask.size(0)
        allowed = torch.zeros((bsz, nb_nodes + 1), device=device, dtype=torch.bool)
        if selected_global_idx.numel() > 0:
            idx_for_mask = torch.where(
                selected_global_idx < 0, torch.zeros_like(selected_global_idx), selected_global_idx + 1
            )
            idx_for_mask = idx_for_mask.clamp(min=0, max=nb_nodes)
            allowed.scatter_(1, idx_for_mask.long(), True)

        feasible = torch.isfinite(base_ninf_mask.squeeze(1))
        allowed = allowed & feasible
        no_candidate = allowed.sum(dim=1) == 0
        if no_candidate.any():
            allowed[no_candidate] = feasible[no_candidate]

        ninf_mask = base_ninf_mask.clone()
        ninf_mask.masked_fill_(~allowed.unsqueeze(1), float("-inf"))
        return ninf_mask

    def select_action(
        self,
        env: VRPEnvironment,
        selected_global_idx: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.encoded_nodes is None or self._env_cache_key != id(env):
            self._prepare_from_env(env)

        selected_global_idx = selected_global_idx.to(env.nodes.device)
        base_ninf_mask, load = self._build_masks_and_load(env)
        ninf_mask = self._candidate_ninf_mask(base_ninf_mask, selected_global_idx, env.nb_nodes)

        last_idx = env.last_visited_idx
        gather_idx = last_idx + 1  # depot -> 0, node i -> i+1
        encoded_last = _get_encoding(self.encoded_nodes, gather_idx)

        probs = self.decoder(encoded_last, load=load, ninf_mask=ninf_mask)  # (bsz, 1, problem+1)
        probs = probs[:, 0, :]

        if deterministic or self.model_params.get("eval_type") == "argmax":
            selected = probs.argmax(dim=1)
        else:
            while True:
                with torch.no_grad():
                    sampled = probs.multinomial(1).squeeze(1)
                    prob = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
                if (prob != 0).all():
                    selected = sampled
                    break

        prob = probs.gather(1, selected.unsqueeze(1)).clamp_min(1e-12).squeeze(1)
        log_prob = prob.log()
        action_idx = torch.where(selected == 0, torch.full_like(selected, -1), selected - 1)
        info: Dict[str, Any] = {
            "probs": probs,
            "ninf_mask": ninf_mask.squeeze(1),
            "candidate_mask": torch.isfinite(ninf_mask.squeeze(1)),
        }
        return action_idx, log_prob, info


class POMOVRPTwoStagePolicy:
    """Composed two-stage policy using independent Stage 1 and Stage 2 POMO VRP models."""

    def __init__(self, **model_params: Any):
        self.stage1 = POMOVRPStage1Policy(**model_params)
        self.stage2 = POMOVRPStage2Policy(**model_params)

    def reset(self) -> None:
        self.stage1.reset()
        self.stage2.reset()

    @torch.no_grad()
    def select_action(
        self,
        env: VRPEnvironment,
        k_promising: int,
        deterministic_stage1: bool = True,
        deterministic_stage2: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        selected_idx, selected_probs, info1 = self.stage1.select_k(
            env, k_promising=k_promising, deterministic=deterministic_stage1
        )
        chosen, logp, info2 = self.stage2.select_action(
            env, selected_global_idx=selected_idx, deterministic=deterministic_stage2
        )
        info = {
            "stage1_probs": info1.get("probs"),
            "stage1_ninf_mask": info1.get("ninf_mask"),
            "stage1_method": info1.get("method"),
            "stage2_probs": info2.get("probs"),
            "stage2_ninf_mask": info2.get("ninf_mask"),
            "stage2_candidate_mask": info2.get("candidate_mask"),
            "selected_global_idx": selected_idx,
            "selected_probs": selected_probs,
        }
        return chosen, logp, info
