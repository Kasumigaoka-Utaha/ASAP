import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from vrp_env import VRPEnvironment


DEFAULT_MODEL_PARAMS = {
    "embedding_dim": 128,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10.0,
    "ff_hidden_dim": 512,
    "sqrt_embedding_dim": 128 ** 0.5,
    "eval_type": "argmax",
}


class POMOVRPModel(nn.Module):
    """POMO-style VRP policy compatible with VRPEnvironment step-wise decoding."""

    def __init__(self, **model_params: Any):
        super().__init__()
        params = {**DEFAULT_MODEL_PARAMS, **model_params}
        params["sqrt_embedding_dim"] = model_params.get("sqrt_embedding_dim", params["embedding_dim"] ** 0.5)
        self.model_params = params

        self.encoder = CVRP_Encoder(**self.model_params)
        self.decoder = CVRP_Decoder(**self.model_params)
        self.encoded_nodes: Optional[torch.Tensor] = None
        self._env_cache_key: Optional[int] = None

    def reset(self) -> None:
        self.encoded_nodes = None
        self._env_cache_key = None

    def _prepare_from_env(self, env: VRPEnvironment) -> None:
        depot_xy = env.depot
        node_xy = env.nodes
        node_demand = env.full_demands[:, :env.nb_nodes]
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        self.decoder.set_kv(self.encoded_nodes)
        self._env_cache_key = id(env)

    def _build_masks_and_load(self, env: VRPEnvironment) -> Tuple[torch.Tensor, torch.Tensor]:
        device = env.nodes.device
        demands = env.full_demands[:, :env.nb_nodes]
        remain_capacity_vec = (
            (env.true_capacity_vec - env.true_used_capacity_vec).float() / env.capacity
        ).clamp(min=0.0, max=1.0)

        if env.problem == "cvrp":
            feasible = (demands > 0) & (demands <= remain_capacity_vec + 1e-6)
        elif env.problem == "sdvrp":
            feasible = demands > 0
        else:
            raise ValueError(f"Unsupported problem type: {env.problem}")

        node_mask = ~feasible
        depot_forbidden = (env.last_visited_idx.squeeze() == -1) & (demands.sum(dim=1) > 0)

        ninf_mask = torch.zeros((env.bsz, 1, env.nb_nodes + 1), device=device)
        ninf_mask[:, :, 0].masked_fill_(depot_forbidden.view(env.bsz, 1), float("-inf"))
        ninf_mask[:, :, 1:].masked_fill_(node_mask.unsqueeze(1), float("-inf"))

        load = remain_capacity_vec.view(env.bsz, 1)
        return ninf_mask, load

    def select_action(
        self, env: VRPEnvironment, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.encoded_nodes is None or self._env_cache_key != id(env):
            self._prepare_from_env(env)

        ninf_mask, load = self._build_masks_and_load(env)

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
        action_idx = torch.where(selected == 0, torch.full_like(selected, -1), selected - 1)
        log_prob = prob.log()
        info: Dict[str, torch.Tensor] = {
            "probs": probs,
            "ninf_mask": ninf_mask.squeeze(1),
            "load": load.squeeze(1),
        }
        return action_idx, log_prob, info

    def rollout(self, env: VRPEnvironment, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset()
        self._prepare_from_env(env)

        tours = []
        log_probs = []
        while not env.is_finished():
            action, logp, _ = self.select_action(env, deterministic=deterministic)
            env.step(action)
            tours.append(action)
            log_probs.append(logp)

        tour_tensor = env.get_tour_tensor(tours)
        if log_probs:
            sum_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)
        else:
            sum_log_prob = torch.zeros(env.bsz, device=env.nodes.device)
        return tour_tensor, sum_log_prob


def _get_encoding(encoded_nodes: torch.Tensor, node_index_to_pick: torch.Tensor) -> torch.Tensor:
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = self.model_params["encoder_layer_num"]

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy: torch.Tensor, node_xy_demand: torch.Tensor) -> torch.Tensor:
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1: torch.Tensor) -> torch.Tensor:
        head_num = self.model_params["head_num"]

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3


class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.single_head_key: Optional[torch.Tensor] = None

    def set_kv(self, encoded_nodes: torch.Tensor) -> None:
        head_num = self.model_params["head_num"]

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last_node: torch.Tensor, load: torch.Tensor, ninf_mask: torch.Tensor) -> torch.Tensor:
        head_num = self.model_params["head_num"]

        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)

        out_concat = multi_head_attention(q_last, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        score = torch.matmul(mh_atten_out, self.single_head_key)

        sqrt_embedding_dim = self.model_params["sqrt_embedding_dim"]
        logit_clipping = self.model_params["logit_clipping"]

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)

        return probs


def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


def multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rank2_ninf_mask: Optional[torch.Tensor] = None,
    rank3_ninf_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float, device=q.device))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)

    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        added = input1 + input2

        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1: torch.Tensor) -> torch.Tensor:
        return self.W2(F.relu(self.W1(input1)))
