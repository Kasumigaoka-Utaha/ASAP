import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from tsp_env import TSPEnvironment


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


class POMOTSPModel(nn.Module):
    """POMO TSP model adapted to the step-wise TSPEnvironment."""

    def __init__(self, **model_params: Any):
        super().__init__()
        params = {**DEFAULT_MODEL_PARAMS, **model_params}
        params["sqrt_embedding_dim"] = model_params.get("sqrt_embedding_dim", params["embedding_dim"] ** 0.5)
        self.model_params = params

        self.encoder = TSP_Encoder(**self.model_params)
        self.decoder = TSP_Decoder(**self.model_params)
        self.encoded_nodes: Optional[torch.Tensor] = None
        self._env_cache_key: Optional[int] = None

    def reset(self) -> None:
        self.encoded_nodes = None
        self._env_cache_key = None

    def _prepare_from_env(self, env: TSPEnvironment) -> None:
        obs = env.observation()
        x = obs["x"]
        self.encoded_nodes = self.encoder(x)
        self.decoder.set_kv(self.encoded_nodes)

        start_idx = env.tours[0].to(x.device)
        encoded_start = _get_encoding(self.encoded_nodes, start_idx.unsqueeze(1))
        self.decoder.set_q1(encoded_start)
        self._env_cache_key = id(env)

    def _ninf_mask_from_env(self, mask_global: torch.Tensor) -> torch.Tensor:
        visited = (~mask_global).unsqueeze(1)
        ninf_mask = torch.zeros(visited.size(), device=mask_global.device, dtype=self.encoded_nodes.dtype)
        ninf_mask.masked_fill_(visited, float("-inf"))
        return ninf_mask

    def select_action(
        self, env: TSPEnvironment, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.encoded_nodes is None or self._env_cache_key != id(env):
            self._prepare_from_env(env)

        obs = env.observation()
        last_idx = env.tours[-1].to(obs["x"].device)
        encoded_last = _get_encoding(self.encoded_nodes, last_idx.unsqueeze(1))
        ninf_mask = self._ninf_mask_from_env(obs["mask_global"])

        probs = self.decoder(encoded_last, ninf_mask=ninf_mask)  # (bsz, 1, problem)
        probs = probs[:, 0, :]

        if deterministic or self.model_params.get("eval_type") == "argmax":
            selected = probs.argmax(dim=1)
        else:
            selected = probs.multinomial(1).squeeze(1)

        prob = probs.gather(1, selected.unsqueeze(1)).clamp_min(1e-12).squeeze(1)
        log_prob = prob.log()
        info: Dict[str, torch.Tensor] = {"probs": probs, "ninf_mask": ninf_mask.squeeze(1)}
        return selected, log_prob, info

    def rollout(self, env: TSPEnvironment, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a full tour on the provided environment."""
        self.reset()
        self._prepare_from_env(env)

        log_probs = []
        done = False
        while not done:
            action, logp, _ = self.select_action(env, deterministic=deterministic)
            _, done = env.step(action)
            log_probs.append(logp)

        tours = env.get_tour_tensor()
        sum_log_prob = torch.stack(log_probs, dim=1).sum(dim=1) if log_probs else torch.zeros(env.bsz, device=env.device)
        return tours, sum_log_prob


def _get_encoding(encoded_nodes: torch.Tensor, node_index_to_pick: torch.Tensor) -> torch.Tensor:
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


class TSP_Encoder(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = self.model_params["encoder_layer_num"]

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        embedded_input = self.embedding(data)
        out = embedded_input
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

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1: torch.Tensor) -> torch.Tensor:
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)
        return out3


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.single_head_key: Optional[torch.Tensor] = None
        self.q_first: Optional[torch.Tensor] = None

    def set_kv(self, encoded_nodes: torch.Tensor) -> None:
        head_num = self.model_params["head_num"]
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_q1(self, encoded_q1: torch.Tensor) -> None:
        head_num = self.model_params["head_num"]
        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)

    def forward(self, encoded_last_node: torch.Tensor, ninf_mask: torch.Tensor) -> torch.Tensor:
        head_num = self.model_params["head_num"]
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        q = self.q_first + q_last

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
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
    sqrt_key_dim = torch.sqrt(torch.tensor(key_dim, dtype=q.dtype, device=q.device))
    score_scaled = score / sqrt_key_dim
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)

    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        added = input1 + input2
        normalized = self.norm(added.transpose(1, 2))
        back_trans = normalized.transpose(1, 2)
        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1: torch.Tensor) -> torch.Tensor:
        return self.W2(F.relu(self.W1(input1)))
