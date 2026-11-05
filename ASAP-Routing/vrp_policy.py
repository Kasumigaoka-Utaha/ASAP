import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from typing import List

from encoder import state_encoder_vrp, action_encoder_vrp
from decoder import Transformer_decoder_net
from vrp_env import VRPEnvironment


class VRPPolicy(nn.Module):
    """VRP policy network that rolls out using VRPEnvironment."""

    def __init__(
        self,
        dim_input_nodes: int,
        dim_emb: int,
        dim_ff: int,
        num_state_encoder: int,
        nb_layers_state_encoder: int,
        nb_layers_action_encoder: int,
        nb_layers_decoder: int,
        nb_heads: int,
        batchnorm: bool = True,
        if_agg_whole_graph: bool = False,
    ):
        super().__init__()
        self.dim_input = dim_input_nodes
        self.dim_emb = dim_emb
        self.if_agg_whole_graph = if_agg_whole_graph
        self.num_state_encoder = num_state_encoder

        self.state_encoders = nn.ModuleList(
            [
                state_encoder_vrp(
                    dim_input_nodes,
                    dim_emb,
                    dim_ff,
                    nb_layers_state_encoder,
                    nb_heads,
                    batchnorm=batchnorm,
                    if_agg_whole_graph=if_agg_whole_graph,
                )
                for _ in range(num_state_encoder)
            ]
        )

        self.action_encoder = action_encoder_vrp(
            dim_input_nodes, dim_emb, dim_ff, nb_layers_action_encoder, nb_heads, batchnorm=batchnorm
        )

        self.decoder = Transformer_decoder_net(dim_emb, nb_heads, nb_layers_decoder)
        self.WK_att_decoder = nn.Linear((num_state_encoder + 1) * dim_emb, nb_layers_decoder * dim_emb)
        self.WV_att_decoder = nn.Linear((num_state_encoder + 1) * dim_emb, nb_layers_decoder * dim_emb)
        self.query_mlp = nn.Linear(2 * (num_state_encoder + 1) * dim_emb, dim_emb)

    def load_pretrained_state_encoder(self, model, i: int):
        if i >= self.num_state_encoder:
            return
        self.state_encoders[i].load_state_dict(model.state_encoders[0].state_dict())
        for _, parameter in self.state_encoders[i].named_parameters():
            parameter.requires_grad = False

    def forward(
        self,
        x: dict,
        action_k: int,
        state_k: List[int],
        capacity: float,
        problem: str = 'cvrp',
        choice_deterministic: bool = False,
        if_use_local_mask: bool = False,
    ):
        assert isinstance(state_k, list)
        assert isinstance(action_k, int)
        assert self.num_state_encoder == len(state_k)

        env = VRPEnvironment(x, capacity=capacity, problem=problem)
        tours: List[torch.Tensor] = []
        sum_log_prob_actions: List[torch.Tensor] = []

        nodes = env.nodes
        bsz = nodes.shape[0]
        zero_to_bsz = torch.arange(bsz, device=nodes.device)

        while not env.is_finished():
            step_ctx = env.build_step_context(action_k, state_k, if_use_local_mask)
            action_idx = step_ctx["action_idx"]
            action_mask = step_ctx["action_mask"]
            state_idx = step_ctx["state_idx"]
            state_mask = step_ctx["state_mask"]
            action_idx_for_choice = step_ctx["action_idx_for_choice"]
            depot_bsz = step_ctx["depot_bsz"]
            demands = step_ctx["demands"]
            remain_capacity_vec = step_ctx["remain_capacity_vec"]
            finished_mask = step_ctx["finished_mask"]

            # action encoder
            emb_action = self.action_encoder(
                nodes, action_idx, env.last_visited_node, env.depot, demands, remain_capacity_vec, encoder_mask=action_mask
            )
            emb_q = emb_action[:, action_k:(action_k + 1), :]
            emb_q = torch.cat((emb_q, emb_action[:, (action_k + 1):(action_k + 2), :]), dim=2)
            emb_other = torch.cat((emb_action[:, :action_k, :], emb_action[:, (action_k + 1):(action_k + 2), :]), dim=1)

            # state encoders
            for i in range(self.num_state_encoder):
                temp_k = state_k[i]
                temp_idx = state_idx[:, :temp_k].contiguous()
                temp_mask = state_mask[:, :temp_k]
                emb_state = self.state_encoders[i](
                    nodes, temp_idx, env.last_visited_node, env.depot, demands, remain_capacity_vec,
                    finished_mask=finished_mask, encoder_mask=temp_mask
                )
                emb_q = torch.cat((emb_q, emb_state[:, temp_k:(temp_k + 1), :]), dim=2)
                emb_q = torch.cat((emb_q, emb_state[:, (temp_k + 1):(temp_k + 2), :]), dim=2)
                temp_other = torch.cat((emb_state[:, :action_k, :], emb_state[:, (temp_k + 1):(temp_k + 2), :]), dim=1)
                emb_other = torch.cat((emb_other, temp_other), dim=2)

            # decoder
            mask_for_decoder = torch.cat((action_mask, torch.zeros((bsz, 1), device=nodes.device)), dim=1).bool()
            mask_for_decoder[depot_bsz, -1] = True
            h_q = self.query_mlp(emb_q)
            K_att_decoder = self.WK_att_decoder(emb_other)
            V_att_decoder = self.WV_att_decoder(emb_other)
            prob_next_node = self.decoder(h_q, K_att_decoder, V_att_decoder, mask_for_decoder)

            if choice_deterministic:
                idx = torch.argmax(prob_next_node, dim=1)
            else:
                idx = Categorical(prob_next_node).sample()

            next_node_idx = action_idx_for_choice[zero_to_bsz, idx]
            env.step(next_node_idx)

            prob_choice = prob_next_node[zero_to_bsz, idx]
            sum_log_prob_actions.append(torch.log(prob_choice))
            tours.append(next_node_idx)

        sum_log_prob_actions = torch.stack(sum_log_prob_actions, dim=1).sum(dim=1)
        tours = env.get_tour_tensor(tours)
        return tours, sum_log_prob_actions
