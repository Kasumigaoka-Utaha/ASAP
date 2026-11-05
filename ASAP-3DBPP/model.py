import torch.nn as nn
import torch.nn.functional as F
import torch
from tools import init
from numpy import sqrt
from attention_model import AttentionModel
from tools import observation_decode_leaf_node_actor
from graph_encoder import GraphAttentionEncoder
import numpy as np





class DRL_GAT(nn.Module):
    def __init__(self, args, search_model = False):
        super(DRL_GAT, self).__init__()
        
        if search_model:
            leaf_node_holder = args.search_num
        else:
            leaf_node_holder = args.leaf_node_holder

        self.actor = AttentionModel(args.embedding_size,
                                    args.hidden_size,
                                    n_encode_layers = args.gat_layer_num,
                                    allow_dist_input = args.allow_dist_input,
                                    n_heads = 1,
                                    num_seen_box = args.num_seen_box,
                                    internal_node_holder = args.internal_node_holder,
                                    internal_node_length = args.internal_node_length,
                                    leaf_node_holder = leaf_node_holder,
                                    )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.critic = init_(nn.Linear(args.embedding_size, 1))

    def forward(self, items, distribution=None, deterministic = False, normFactor = 1, evaluate = False):
        action_log_probs, action, dist_entropy, hidden, dist, node_embeddings = self.actor(items, deterministic=deterministic, normFactor=normFactor, evaluate=evaluate)
        values = self.critic(hidden)
        return action_log_probs, action, dist_entropy, values, dist, node_embeddings

    def action_log_probs(self, input, distribution = None, normFactor = 1):
        _, _, _, _, dist, _ = self.actor(input, evaluate_action=True, normFactor=normFactor)
        return dist.probs.log()

    def evaluate(self, items, actions, distribution = None, deterministic = False, normFactor = 1):
        _, action, dist_entropy, hidden, dist, _ = self.actor(items, deterministic=deterministic, evaluate_action=True, normFactor=normFactor)
        action_log_probs = dist.log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
        values =  self.critic(hidden)
        return values, action_log_probs, dist_entropy.mean(), dist





