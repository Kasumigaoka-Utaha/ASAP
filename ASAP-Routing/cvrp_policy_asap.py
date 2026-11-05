import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from typing import Dict, Any, Tuple

from encoder import action_encoder_vrp, state_encoder_vrp
from decoder import Transformer_decoder_net
from vrp_env import VRPEnvironment


class _VRPBaseStage(nn.Module):
    """Shared components for VRP two-stage policies (action/state encoders + decoder over full nodes)."""

    def __init__(self, args):
        super().__init__()
        dim_emb = args.dim_emb  # alias
        self.dim_input = args.dim_input_nodes
        self.k_nearest = getattr(args, "knn_k", 15)
        self.num_state_encoder = getattr(args, "num_state_encoder", 0)
        self.if_agg_whole_graph = getattr(args, "if_agg_whole_graph", False)

        self.action_encoder = action_encoder_vrp(
            args.dim_input_nodes, dim_emb, args.dim_ff, args.nb_layers_action_encoder, args.nb_heads, batchnorm=args.batchnorm
        )
        self.state_encoders = nn.ModuleList(
            [
                state_encoder_vrp(
                    args.dim_input_nodes,
                    dim_emb,
                    args.dim_ff,
                    args.nb_layers_state_encoder,
                    args.nb_heads,
                    args.batchnorm,
                    if_agg_whole_graph=self.if_agg_whole_graph,
                )
                for _ in range(self.num_state_encoder)
            ]
        )
        self.decoder = Transformer_decoder_net(dim_emb, args.nb_heads, args.nb_layers_decoder)
        combined_dim = (self.num_state_encoder + 1) * dim_emb
        self.WK_att_decoder = nn.Linear(combined_dim, args.nb_layers_decoder * dim_emb)
        self.WV_att_decoder = nn.Linear(combined_dim, args.nb_layers_decoder * dim_emb)
        self.query_mlp = nn.Linear(2 * combined_dim, dim_emb)

    def _build_embeddings(
        self,
        env: VRPEnvironment,
    ) -> Dict[str, torch.Tensor]:
        """Prepare normalized inputs, masks, and demand/capacity info for a VRP step."""
        nodes = env.nodes
        bsz, nb_nodes, _ = nodes.shape

        # Demand/capacity-derived availability
        demands = env.full_demands[:, :nb_nodes]
        remain_capacity_vec = (env.true_capacity_vec - env.true_used_capacity_vec) / env.capacity
        if env.problem == 'cvrp':
            available = (demands > 0) & (demands <= remain_capacity_vec + 1e-6)
        elif env.problem == 'sdvrp':
            available = demands > 0
        else:
            raise ValueError(f"Unsupported problem type: {env.problem}")
        action_mask = ~available  # True = infeasible
        # Do not allow staying at depot if already there and there is remaining demand.
        at_depot = (env.last_visited_idx.squeeze() == -1) & (demands.sum(dim=1) != 0)

        # action_encoder_vrp handles normalization internally; use raw coords here
        nodes_for_enc = nodes
        graph_ext = torch.cat((nodes_for_enc, env.depot), dim=1)
        demands_ext = torch.cat((demands, torch.zeros((bsz, 1), device=nodes.device, dtype=demands.dtype)), dim=1)
        action_mask_ext = torch.cat((action_mask, at_depot.unsqueeze(1)), dim=1)

        candidate_global_idx = torch.cat(
            (
                torch.arange(nb_nodes, device=nodes.device).unsqueeze(0).repeat(bsz, 1),
                env.depot_idx,
            ),
            dim=1,
        )

        mask_for_decoder = action_mask_ext
        finished_mask = ~(demands > 0)
        finished_mask_ext = ~(demands_ext > 0)

        return {
            "bsz": bsz,
            "nb_nodes": nb_nodes,
            "nodes_for_enc": nodes_for_enc,
            "graph_ext": graph_ext,
            "depot_for_enc": env.depot,
            "last_for_enc": env.last_visited_node,
            "demands": demands,
            "demands_ext": demands_ext,
            "remain_capacity_vec": remain_capacity_vec,
            "action_mask": action_mask,
            "action_mask_ext": action_mask_ext,
            "candidate_global_idx": candidate_global_idx,
            "mask_for_decoder": mask_for_decoder,
            "at_depot": at_depot,
            "finished_mask": finished_mask,
            "finished_mask_ext": finished_mask_ext,
        }


class VRPStage1Policy(_VRPBaseStage):
    """Stage 1 policy: propose k promising actions over current VRP candidates."""

    def select_k(
        self,
        env: VRPEnvironment,
        k_promising: int,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        base = self._build_embeddings(env)

        # Restrict to at most k_nearest feasible nodes w.r.t last visited
        bsz, nb_nodes = base["bsz"], base["nb_nodes"]
        nodes = base["nodes_for_enc"]
        last = base["last_for_enc"]
        feasible_mask = ~base["action_mask"]  # True where feasible
        num_select = min(self.k_nearest, nb_nodes)
        dist = torch.norm(nodes - last, dim=2)  # (bsz, nb_nodes)
        dist_masked = dist.masked_fill(~feasible_mask, float('inf'))
        _, nearest_idx = torch.topk(-dist_masked, k=num_select, dim=1)  # negative for smallest distance
        action_idx = nearest_idx

        # Encode selected subset (padding handled by encoder mask)
        enc_mask_subset = ~feasible_mask.gather(1, action_idx)
        emb_action = self.action_encoder(
            base["nodes_for_enc"],
            action_idx,
            base["last_for_enc"],
            base["depot_for_enc"],
            base["demands"],
            base["remain_capacity_vec"],
            encoder_mask=enc_mask_subset,
        )

        emb_last = emb_action[:, num_select:(num_select + 1), :]
        emb_depot = emb_action[:, (num_select + 1):(num_select + 2), :]
        emb_q_parts = [emb_last, emb_depot]
        # Append depot embedding as an action option; mask controls feasibility.
        emb_other_parts = [torch.cat((emb_action[:, :num_select, :], emb_depot), dim=1)]

        finished_mask = base["finished_mask"]
        for state_enc in self.state_encoders:
            emb_state = state_enc(
                base["nodes_for_enc"],
                action_idx,
                base["last_for_enc"],
                base["depot_for_enc"],
                base["demands"],
                base["remain_capacity_vec"],
                finished_mask=finished_mask,
                encoder_mask=enc_mask_subset,
            )
            state_last = emb_state[:, num_select:(num_select + 1), :]
            state_depot = emb_state[:, (num_select + 1):(num_select + 2), :]
            emb_q_parts.extend([state_last, state_depot])
            emb_other_parts.append(torch.cat((emb_state[:, :num_select, :], state_depot), dim=1))

        emb_q = torch.cat(emb_q_parts, dim=2)
        emb_other = torch.cat(emb_other_parts, dim=2)

        h_q = self.query_mlp(emb_q)
        K_att_decoder = self.WK_att_decoder(emb_other)
        V_att_decoder = self.WV_att_decoder(emb_other)
        depot_mask = base["at_depot"].unsqueeze(1)
        decoder_mask = torch.cat((~feasible_mask.gather(1, action_idx), depot_mask), dim=1)
        prob_next_node = self.decoder(h_q, K_att_decoder, V_att_decoder, decoder_mask)
        # Explicitly zero out masked positions (e.g., depot when already at depot) and renormalize.
        prob_next_node = prob_next_node.masked_fill(decoder_mask, 0.0)
        valid_mask = ~decoder_mask
        row_sum = (prob_next_node * valid_mask).sum(dim=1, keepdim=True)
        # If all probabilities got masked/zeroed, fall back to uniform over valid positions (or all if none).
        fallback = valid_mask.float()
        fallback_sum = fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
        prob_next_node = torch.where(
            row_sum > 0,
            prob_next_node / row_sum,
            fallback / fallback_sum,
        )

        k = max(1, min(k_promising, prob_next_node.size(1)))
        candidate_subset = base["candidate_global_idx"].gather(1, action_idx)
        candidate_subset_with_depot = torch.cat((candidate_subset, env.depot_idx), dim=1)
        if deterministic:
            topk_vals, topk_pos = torch.topk(prob_next_node, k, dim=1)
            selected_global_idx = candidate_subset_with_depot.gather(1, topk_pos)
            selected_probs = topk_vals
            pos = topk_pos
            method = 'topk'
        else:
            sampled_pos = torch.multinomial(prob_next_node, num_samples=k, replacement=False)
            selected_global_idx = candidate_subset_with_depot.gather(1, sampled_pos)
            selected_probs = prob_next_node.gather(1, sampled_pos)
            pos = sampled_pos
            method = 'sample'

        out_info: Dict[str, Any] = {
            "candidate_global_idx": candidate_subset_with_depot,
            "candidate_probs": prob_next_node,
            "selected_pos_in_cand": pos,
            "method": method,
            "sampled_pos_in_cand": sampled_pos if not deterministic else None,
            "topk_pos_in_cand": topk_pos if deterministic else None,
        }
        return selected_global_idx, selected_probs, out_info

    def select_action(
        self,
        env: VRPEnvironment,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Directly pick one action (without Stage 2)."""
        selected_idx, selected_probs, info = self.select_k(env, k_promising=1, deterministic=deterministic)
        chosen = selected_idx.squeeze(1)
        log_prob = torch.log(selected_probs.squeeze(1).clamp_min(1e-12))
        return chosen, log_prob, info


class VRPStage2Policy(_VRPBaseStage):
    """Stage 2 policy: choose final action among Stage 1 candidates (action/state encoders)."""

    def select_action(
        self,
        env: VRPEnvironment,
        selected_global_idx: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        base = self._build_embeddings(env)
        device = env.nodes.device

        bsz = base["bsz"]
        nb_nodes = base["nb_nodes"]
        nb_ext = nb_nodes + 1  # include depot

        candidate_global_idx = selected_global_idx.long()
        candidate_with_depot = torch.cat((candidate_global_idx, env.depot_idx), dim=1)

        # Map Stage 1 selections to local indices over nodes+depot
        action_idx_local = candidate_global_idx.clone()
        action_idx_local[action_idx_local < 0] = nb_nodes
        k_action = action_idx_local.size(1)

        encoder_mask_actions = base["action_mask_ext"].gather(1, action_idx_local.clamp(max=nb_ext - 1))
        # Decoder mask: candidates + depot slot. Action encoder mask covers only candidates.
        decoder_mask = torch.cat((encoder_mask_actions, base["at_depot"].unsqueeze(1)), dim=1)

        # Action encoder over restricted candidate set (normalized graph + depot appended)
        emb_action = self.action_encoder(
            base["graph_ext"],
            action_idx_local,
            base["last_for_enc"],
            base["depot_for_enc"],
            base["demands_ext"],
            base["remain_capacity_vec"],
            encoder_mask=encoder_mask_actions,
        )

        emb_last = emb_action[:, k_action:(k_action + 1), :]
        emb_depot = emb_action[:, (k_action + 1):(k_action + 2), :]
        emb_q_parts = [emb_last, emb_depot]
        # Append depot embedding as selectable action; mask will block when at depot.
        emb_other_parts = [torch.cat((emb_action[:, :k_action, :], emb_depot), dim=1)]

        finished_mask_ext = base["finished_mask_ext"]
        for state_enc in self.state_encoders:
            emb_state = state_enc(
                base["graph_ext"],
                action_idx_local,
                base["last_for_enc"],
                base["depot_for_enc"],
                base["demands_ext"],
                base["remain_capacity_vec"],
                finished_mask=finished_mask_ext,
                encoder_mask=encoder_mask_actions,
            )
            state_last = emb_state[:, k_action:(k_action + 1), :]
            state_depot = emb_state[:, (k_action + 1):(k_action + 2), :]
            emb_q_parts.extend([state_last, state_depot])
            emb_other_parts.append(torch.cat((emb_state[:, :k_action, :], state_depot), dim=1))

        emb_q = torch.cat(emb_q_parts, dim=2)
        emb_other = torch.cat(emb_other_parts, dim=2)

        h_q = self.query_mlp(emb_q)
        K_att_decoder = self.WK_att_decoder(emb_other)
        V_att_decoder = self.WV_att_decoder(emb_other)
        prob_next_node = self.decoder(h_q, K_att_decoder, V_att_decoder, decoder_mask)
        # Zero out masked actions (including depot when currently at depot) and renormalize.
        prob_next_node = prob_next_node.masked_fill(decoder_mask, 0.0)
        valid_mask = ~decoder_mask
        row_sum = (prob_next_node * valid_mask).sum(dim=1, keepdim=True)
        fallback = valid_mask.float()
        fallback_sum = fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
        prob_next_node = torch.where(
            row_sum > 0,
            prob_next_node / row_sum,
            fallback / fallback_sum,
        )

        if deterministic:
            idx = torch.argmax(prob_next_node, dim=1)
        else:
            idx = Categorical(prob_next_node).sample()

        zero_to_bsz = torch.arange(bsz, device=device)
        chosen_global_idx = candidate_with_depot[zero_to_bsz, idx]
        log_prob = torch.log(prob_next_node[zero_to_bsz, idx].clamp_min(1e-12))
        info: Dict[str, Any] = {
            "candidate_global_idx": candidate_with_depot,
            "candidate_probs": prob_next_node,
            "select_idx": idx,
        }
        return chosen_global_idx, log_prob, info


class VRPTwoStagePolicy:
    """Composed two-stage VRP policy using independent Stage 1 and Stage 2 networks."""

    def __init__(self, args):
        self.stage1 = VRPStage1Policy(args)
        self.stage2 = VRPStage2Policy(args)

    @torch.no_grad()
    def select_action(
        self,
        env: VRPEnvironment,
        k_promising: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        selected_idx, selected_probs, info1 = self.stage1.select_k(
            env, k_promising=k_promising, deterministic=True
        )
        chosen, logp, info2 = self.stage2.select_action(
            env, selected_global_idx=selected_idx, deterministic=True
        )
        info = {**info1, **info2, "selected_global_idx": selected_idx, "selected_probs": selected_probs}
        return chosen, logp, info
