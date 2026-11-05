import torch
from typing import Any, Dict, Tuple

from pomo_tsp_policy import POMOTSPModel, _get_encoding
from tsp_env import TSPEnvironment


class POMOTSPStage1Policy(POMOTSPModel):
    """Stage 1 POMO policy that can return top-k candidate actions."""

    def select_k(
        self,
        env: TSPEnvironment,
        k_promising: int,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.encoded_nodes is None or self._env_cache_key != id(env):
            self._prepare_from_env(env)

        obs = env.observation()
        device = obs["x"].device

        last_idx = env.tours[-1].to(device)
        encoded_last = _get_encoding(self.encoded_nodes, last_idx.unsqueeze(1))
        ninf_mask = self._ninf_mask_from_env(obs["mask_global"])

        probs = self.decoder(encoded_last, ninf_mask=ninf_mask)  # (bsz, 1, problem)
        probs = probs[:, 0, :]

        # Limit k to the number of available nodes so we never select a visited node.
        max_k = int(obs["mask_global"].sum(dim=1).min().item())
        k = max(1, min(k_promising, max_k))
        if deterministic or self.model_params.get("eval_type") == "argmax":
            selected_probs, selected = probs.topk(k, dim=1)
            method = "topk"
        else:
            selected = probs.multinomial(num_samples=k, replacement=False)
            selected_probs = probs.gather(1, selected)
            method = "sample"

        info: Dict[str, Any] = {
            "probs": probs,
            "ninf_mask": ninf_mask.squeeze(1),
            "method": method,
        }
        return selected, selected_probs, info


class POMOTSPStage2Policy(POMOTSPModel):
    """Stage 2 POMO policy that restricts decoding to a provided candidate set."""

    def _ninf_mask_from_candidates(self, mask_global: torch.Tensor, selected_global_idx: torch.Tensor) -> torch.Tensor:
        """Build -inf mask that blocks visited nodes and those outside selected_global_idx."""
        bsz, problem = mask_global.shape
        device = mask_global.device
        allowed = torch.zeros((bsz, problem), device=device, dtype=torch.bool)
        if selected_global_idx.numel() > 0:
            allowed.scatter_(1, selected_global_idx.long().clamp(min=0, max=problem - 1), True)
        allowed = allowed & mask_global  # only allow unvisited nodes that are in the candidate set

        ninf_mask = torch.zeros((bsz, 1, problem), device=device, dtype=self.encoded_nodes.dtype)
        ninf_mask.masked_fill_(~allowed.unsqueeze(1), float("-inf"))
        return ninf_mask

    def select_action(
        self,
        env: TSPEnvironment,
        selected_global_idx: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.encoded_nodes is None or self._env_cache_key != id(env):
            self._prepare_from_env(env)

        obs = env.observation()
        device = obs["x"].device
        selected_global_idx = selected_global_idx.to(device)

        last_idx = env.tours[-1].to(device)
        encoded_last = _get_encoding(self.encoded_nodes, last_idx.unsqueeze(1))
        ninf_mask = self._ninf_mask_from_candidates(obs["mask_global"], selected_global_idx)

        probs = self.decoder(encoded_last, ninf_mask=ninf_mask)  # (bsz, 1, problem)
        probs = probs[:, 0, :]

        if deterministic or self.model_params.get("eval_type") == "argmax":
            chosen = probs.argmax(dim=1)
        else:
            chosen = probs.multinomial(num_samples=1).squeeze(1)

        prob = probs.gather(1, chosen.unsqueeze(1)).clamp_min(1e-12).squeeze(1)
        log_prob = prob.log()
        info: Dict[str, Any] = {
            "probs": probs,
            "ninf_mask": ninf_mask.squeeze(1),
            "candidate_mask": ninf_mask.squeeze(1).isfinite(),
        }
        return chosen, log_prob, info


class POMOTSPTwoStagePolicy:
    """Composed two-stage policy using independent Stage 1 and Stage 2 POMO models."""

    def __init__(self, **model_params: Any):
        self.stage1 = POMOTSPStage1Policy(**model_params)
        self.stage2 = POMOTSPStage2Policy(**model_params)

    def reset(self) -> None:
        self.stage1.reset()
        self.stage2.reset()

    @torch.no_grad()
    def select_action(
        self,
        env: TSPEnvironment,
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
