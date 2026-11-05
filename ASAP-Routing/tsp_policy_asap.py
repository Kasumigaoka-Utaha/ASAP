import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from utils.knn import knn
from typing import Any, Dict, Tuple

from encoder import action_encoder_tsp, state_encoder_tsp
from decoder import Transformer_decoder_net


class TSPStage1Policy(nn.Module):
    """Stage 1 policy: sample k promising actions from probabilities over local unvisited nodes.

    Encodes up to k nearest unvisited nodes with an independent action encoder (no state encoder),
    projects to the decoder space, obtains probabilities, and samples k without replacement.
    """

    def __init__(self, args):
        super().__init__()
        # Save dims
        self.dim_input = args.dim_input_nodes
        dim_emb = args.dim_emb
        self.k_nearest = getattr(args, "knn_k", 25)
        # Build independent action encoder and decoder from args (always normalize before encoding)
        self.action_encoder = action_encoder_tsp(
            args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_action_encoder, args.nb_heads, args.batchnorm, use_normalization_layer=True
        )
        self.decoder = Transformer_decoder_net(args.dim_emb, args.nb_heads, args.nb_layers_decoder)
        nb_layers_decoder = args.nb_layers_decoder
        self.WK_att = nn.Linear(dim_emb, nb_layers_decoder * dim_emb)
        self.WV_att = nn.Linear(dim_emb, nb_layers_decoder * dim_emb)
        self.WQ = nn.Linear(dim_emb, dim_emb)

    def select_k(
        self,
        obs: Dict[str, torch.Tensor],
        k_promising: int,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Return k candidate actions and their probs using only action encoder.

        Outputs
        - selected_global_idx: (bsz, k)
        - selected_probs:      (bsz, k)
        - info: full candidate set, positions, and encodings of the selected nodes
        """
        x: torch.Tensor = obs["x"]
        last_visited_node: torch.Tensor = obs["last_visited_node"]
        mask_global: torch.Tensor = obs["mask_global"]


        bsz, nb_nodes, _ = x.shape
        zero_to_bsz = torch.arange(bsz, device=x.device)

        # Build unvisited matrix and full unvisited graph
        all_idx = torch.arange(0, nb_nodes, device=x.device).repeat((bsz, 1))
        unvisited_matrix = torch.reshape(all_idx[mask_global], (bsz, -1))
        num_nodes = unvisited_matrix.size(1)

        b_graph = torch.arange(0, bsz, device=x.device).repeat(num_nodes).sort()[0]
        unvisited_matrix_idx = unvisited_matrix.view((-1,))
        graph = x[b_graph, unvisited_matrix_idx].view((bsz, -1, self.dim_input))

        # Restrict to k nearest neighbors of the last visited node
        k_all = min(self.k_nearest, num_nodes)
        graph_for_knn = graph.view((-1, self.dim_input))
        last_for_knn = last_visited_node.view((-1, self.dim_input))
        knn_output = knn(graph_for_knn, last_for_knn, k_all, b_graph, zero_to_bsz)
        action_idx = (knn_output[1, :] % num_nodes).view(bsz, k_all).contiguous()

        # Encode nearest unvisited nodes with action encoder
        emb_action = self.action_encoder(graph, action_idx, last_visited_node, mask=None)
        emb_q = emb_action[:, k_all:(k_all + 1), :]
        emb_other = emb_action[:, :k_all, :]

        # Decoder over the full unvisited set using only action features
        h_q = self.WQ(emb_q)
        K_att_decoder = self.WK_att(emb_other)
        V_att_decoder = self.WV_att(emb_other)
        prob_next_node = self.decoder(h_q, K_att_decoder, V_att_decoder, mask=None)

        # Candidates map directly to unvisited_matrix order
        candidate_global_idx = unvisited_matrix.gather(1, action_idx)

        # Select k promising actions: top-k (deterministic) or sample without replacement
        k = max(1, min(k_promising, prob_next_node.size(1)))
        if deterministic:
            topk_vals, topk_pos = torch.topk(prob_next_node, k, dim=1)
            selected_global_idx = candidate_global_idx.gather(1, topk_pos)
            selected_probs = topk_vals
            pos = topk_pos
            method = 'topk'
        else:
            sampled_pos = torch.multinomial(prob_next_node, num_samples=k, replacement=False)
            selected_global_idx = candidate_global_idx.gather(1, sampled_pos)
            selected_probs = prob_next_node.gather(1, sampled_pos)
            pos = sampled_pos
            method = 'sample'

        # Gather the action-encoder embeddings corresponding to the selected candidates
        gather_idx = pos.unsqueeze(-1).expand(-1, -1, emb_other.size(-1))
        selected_action_encoding = emb_other.gather(1, gather_idx)
        selected_action_encoding = torch.cat((selected_action_encoding, emb_q), dim=1)
        out_info: Dict[str, Any] = {
            "candidate_global_idx": candidate_global_idx,
            "candidate_probs": prob_next_node,
            "selected_pos_in_cand": pos,
            "method": method,
            "sampled_pos_in_cand": sampled_pos if not deterministic else None,
            "topk_pos_in_cand": topk_pos if deterministic else None,
            "selected_action_encoding": selected_action_encoding,
        }
        return selected_global_idx, selected_probs, out_info

    def select_action(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Directly select a single final action over ALL unvisited nodes.

        Returns
        - chosen_global_idx: (bsz,)
        - log_prob:          (bsz,)
        - info:              dict with full candidate indices and probabilities
        """
        x: torch.Tensor = obs["x"]
        last_visited_node: torch.Tensor = obs["last_visited_node"]
        mask_global: torch.Tensor = obs["mask_global"]

        bsz, nb_nodes, _ = x.shape
        zero_to_bsz = torch.arange(bsz, device=x.device)

        # Unvisited matrix and graph
        all_idx = torch.arange(0, nb_nodes, device=x.device).repeat((bsz, 1))
        unvisited_matrix = torch.reshape(all_idx[mask_global], (bsz, -1))
        num_nodes = unvisited_matrix.size(1)

        b_graph = torch.arange(0, bsz, device=x.device).repeat(num_nodes).sort()[0]
        unvisited_matrix_idx = unvisited_matrix.view((-1,))
        graph = x[b_graph, unvisited_matrix_idx].view((bsz, -1, self.dim_input))

        # Encode nearest unvisited
        k_all = min(self.k_nearest, num_nodes)
        graph_for_knn = graph.view((-1, self.dim_input))
        last_for_knn = last_visited_node.view((-1, self.dim_input))
        knn_output = knn(graph_for_knn, last_for_knn, k_all, b_graph, zero_to_bsz)
        action_idx = (knn_output[1, :] % num_nodes).view(bsz, k_all).contiguous()
        emb_action = self.action_encoder(graph, action_idx, last_visited_node, mask=None)
        emb_q = emb_action[:, k_all:(k_all + 1), :]
        emb_other = emb_action[:, :k_all, :]

        # Decode probs
        h_q = self.WQ(emb_q)
        K_att_decoder = self.WK_att(emb_other)
        V_att_decoder = self.WV_att(emb_other)
        prob_next_node = self.decoder(h_q, K_att_decoder, V_att_decoder, mask=None)

        # Choose action
        if deterministic:
            idx = torch.argmax(prob_next_node, dim=1)
        else:
            idx = Categorical(prob_next_node).sample()

        candidate_global_idx = unvisited_matrix.gather(1, action_idx)
        chosen_global_idx = candidate_global_idx[zero_to_bsz, idx]
        log_prob = torch.log(prob_next_node[zero_to_bsz, idx].clamp_min(1e-12))
        info: Dict[str, Any] = {
            "candidate_global_idx": candidate_global_idx,
            "candidate_probs": prob_next_node,
            "idx": idx,
        }
        return chosen_global_idx, log_prob, info


class TSPStage2Policy(nn.Module):
    """Stage 2 policy: independent network to predict final action among Stage 1 candidates.

    Re-runs the one-step computation restricted to the Stage 1 candidate subset using
    its own action encoder (over Stage 1 candidates) and a state encoder over k nearest neighbors.
    """

    def __init__(self, args):
        super().__init__()
        # Save dims
        self.dim_input = args.dim_input_nodes
        dim_emb = args.dim_emb
        self.k_nearest = getattr(args, "knn_k", 25)
        # Independent encoders and decoder from args (normalize inputs before each encoder)
        self.action_encoder = action_encoder_tsp(
            args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_action_encoder, args.nb_heads, args.batchnorm, use_normalization_layer=True
        )
        self.state_encoder = state_encoder_tsp(
            args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_state_encoder, args.nb_heads, args.batchnorm, if_agg_whole_graph=False
        )
        self.decoder = Transformer_decoder_net(args.dim_emb, args.nb_heads, args.nb_layers_decoder)
        nb_layers_decoder = args.nb_layers_decoder
        # action emb + state emb -> concat, plus last/first state context for query
        self.WK_att = nn.Linear(2 * dim_emb, nb_layers_decoder * dim_emb)
        self.WV_att = nn.Linear(2 * dim_emb, nb_layers_decoder * dim_emb)
        self.WQ = nn.Linear(3 * dim_emb, dim_emb)

    def select_action(
        self,
        obs: Dict[str, torch.Tensor],
        selected_global_idx: torch.Tensor,  # (bsz, k)
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x: torch.Tensor = obs["x"]
        last_visited_node: torch.Tensor = obs["last_visited_node"]
        first_visited_node: torch.Tensor = obs["first_visited_node"]
        mask_global: torch.Tensor = obs["mask_global"]

        # Ensure projections on device

        bsz = x.shape[0]
        nb_nodes = x.shape[1]
        zero_to_bsz = torch.arange(bsz, device=x.device)
        k_action = selected_global_idx.size(1)

        # Build unvisited matrix and local subgraph
        all_idx = torch.arange(0, nb_nodes, device=x.device).repeat((bsz, 1))
        unvisited_matrix = torch.reshape(all_idx[mask_global], (bsz, -1))
        num_nodes = unvisited_matrix.size(1)

        b_graph = torch.arange(0, bsz, device=x.device).repeat(num_nodes).sort()[0]
        unvisited_matrix_idx = unvisited_matrix.view((-1,))
        graph = x[b_graph, unvisited_matrix_idx].view((bsz, -1, self.dim_input))

        # Map Stage1 candidate globals -> local indices within unvisited_matrix
        positions = torch.arange(num_nodes, device=x.device).repeat(bsz, 1)
        idx_map = torch.full((bsz, nb_nodes), -1, dtype=torch.long, device=x.device)
        idx_map.scatter_(1, unvisited_matrix, positions)
        action_idx = idx_map.gather(1, selected_global_idx).contiguous()  # (bsz, k)

        # action encoder over restricted candidate set (no mask)
        emb_action = self.action_encoder(graph, action_idx, last_visited_node, mask=None)
        emb_q = emb_action[:, k_action:(k_action + 1), :]
        emb_other = emb_action[:, :k_action, :]
        # State encoder over k nearest neighbors
        k_state = min(self.k_nearest, num_nodes)
        graph_for_knn = graph.view((-1, self.dim_input))
        last_for_knn = last_visited_node.view((-1, self.dim_input))
        knn_output = knn(graph_for_knn, last_for_knn, k_state, b_graph, zero_to_bsz)
        knn_idx = (knn_output[1, :] % num_nodes).view(bsz, k_state).contiguous()
        emb_state = self.state_encoder(graph, knn_idx, last_visited_node, first_visited_node)

        # Gather state embeddings aligned with candidate actions
        pos_map = torch.full((bsz, num_nodes), -1, device=x.device, dtype=torch.long)
        pos_map.scatter_(1, knn_idx, torch.arange(k_state, device=x.device).unsqueeze(0).expand(bsz, k_state))
        state_pos_for_actions = pos_map.gather(1, action_idx).clamp(min=0)
        missing_mask = pos_map.gather(1, action_idx) < 0
        gather_idx = state_pos_for_actions.unsqueeze(-1).expand(-1, -1, emb_state.size(-1))
        state_for_actions = emb_state.gather(1, gather_idx)
        if missing_mask.any():
            state_for_actions = state_for_actions.masked_fill(missing_mask.unsqueeze(-1), 0.0)
        state_last = emb_state[:, k_state:(k_state + 1), :]
        state_first = emb_state[:, (k_state + 1):(k_state + 2), :]

        # decoder
        emb_q_cat = torch.cat((emb_q, state_last, state_first), dim=2)
        emb_other_cat = torch.cat((emb_other, state_for_actions), dim=2)
        h_q = self.WQ(emb_q_cat)
        K_att_decoder = self.WK_att(emb_other_cat)
        V_att_decoder = self.WV_att(emb_other_cat)
        prob_next_node = self.decoder(h_q, K_att_decoder, V_att_decoder, mask=None)

        # select final action among restricted candidates
        if deterministic:
            idx = torch.argmax(prob_next_node, dim=1)
        else:
            idx = Categorical(prob_next_node).sample()

        chosen_global_idx = selected_global_idx[zero_to_bsz, idx]
        log_prob = torch.log(prob_next_node[zero_to_bsz, idx].clamp_min(1e-12))
        info: Dict[str, Any] = {"select_idx":idx,"prob_restrict": prob_next_node}
        return chosen_global_idx, log_prob, info


class TSPTwoStagePolicy:
    """Composed two-stage policy using independent Stage 1 and Stage 2 networks."""

    def __init__(self, args):
        self.stage1 = TSPStage1Policy(args)
        self.stage2 = TSPStage2Policy(args)

    @torch.no_grad()
    def select_action(
        self,
        obs: Dict[str, torch.Tensor],
        k_promising: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        selected_idx, selected_probs, info1 = self.stage1.select_k(obs, k_promising=k_promising)
        chosen, logp, info2 = self.stage2.select_action(
            obs,
            selected_global_idx=selected_idx,
            deterministic=True,
        )
        info = {**info1, **info2, "selected_global_idx": selected_idx, "selected_probs": selected_probs}
        return chosen, logp, info
