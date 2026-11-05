import os
import torch
import torch.nn as nn
import argparse
import packing_config
import numpy as np
from gym.envs.registration import register

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        elif x.dim() == 1:
            bias = self._bias.t().view(1, -1)
        elif x.dim() == 3:
            bias = self._bias.t().view(1, 1, -1)
        else:
            assert False

        return x + bias



# Parsing PCT node from state returned in environment
def get_leaf_nodes(observation, internal_node_holder, leaf_node_holder):
    unify_obs = observation.reshape((observation.shape[0], -1, 9))
    leaf_nodes = unify_obs[:, internal_node_holder:internal_node_holder + leaf_node_holder, :]
    return unify_obs, leaf_nodes


'''
Parsing the raw state returned in environment:

internal_nodes    : A packed item vector, [x1, y1, z1, x2, y2, z2, density(optional) ]
                    x1, y1, z1 are coordinates of a packed item
                    x2 = x1 + x, y2 = y1 + y, z2 = z1 + z
                    x, y, z are sizes of a packed item (a little different from the original paper,
                    these two description have similar performance.).
leaf_nodes        : A placement vector, [x1, y1, z1, x2, y2, z2]
                    x1, y1, z1 are coordinates of a placement.
                    x2 = x1 + x, y2 = y1 + y, z2 = z1 + z
                    x, y, z are  sizes of the current item after an axis-aligned orientation (a little different from the original paper,
                    these two description have similar performance.).
next_item         : The next item to be packed [density(optional), 0, 0,x, y, z]
                    x, y, z are  sizes of the current item.
invalid_leaf_nodes: The mask which indicates whether this placement is feasible.
full_mask         : The mask which indicates whether this node should be encode by GAT.
'''
def observation_decode_leaf_node_actor(observation, internal_node_holder, internal_node_length, leaf_node_holder,num_seen_box=None):
    internal_nodes = observation[:, 0:internal_node_holder, 0:internal_node_length]
    leaf_nodes = observation[:, internal_node_holder:internal_node_holder + leaf_node_holder, 0:8]
    valid_flag = observation[:,internal_node_holder: internal_node_holder + leaf_node_holder, 8]
    if num_seen_box is not None:
        current_box = observation[:,internal_node_holder + leaf_node_holder:internal_node_holder + leaf_node_holder + num_seen_box, 0:6]
        full_mask = observation[:, 0:internal_node_holder + leaf_node_holder + num_seen_box, -1]
    else:
        current_box = observation[:,internal_node_holder + leaf_node_holder:, 0:6]
        full_mask = observation[:, :, -1]
    return internal_nodes, leaf_nodes, current_box, valid_flag, full_mask




def process_search_space(observation, internal_node_holder, leaf_node_holder, indices, sol_action = None):

    internal_nodes = observation[:, 0:internal_node_holder, :]
    leaf_nodes = observation[:, internal_node_holder:internal_node_holder + leaf_node_holder, :]
    current_box = observation[:,internal_node_holder + leaf_node_holder:, :]
    bsz = indices.size(0)
    search_num = indices.size(1)
    idx = torch.arange(0,bsz).repeat(search_num).sort()[0].to(observation.device)
    processed_leaf_nodes = leaf_nodes[idx,indices.view((-1,)),:].view((bsz,search_num,-1))
    if sol_action is not None:
        processed_leaf_nodes = torch.cat((processed_leaf_nodes,sol_action.unsqueeze(dim=1)),dim=1)
    return torch.cat((internal_nodes,processed_leaf_nodes,current_box),dim=1), processed_leaf_nodes




def load_policy(load_path, upper_policy):
    
    assert os.path.exists(load_path), 'File does not exist'
    pretrained_state_dict = torch.load(load_path, map_location='cpu')
    if len(pretrained_state_dict) == 2:
        pretrained_state_dict, ob_rms = pretrained_state_dict

    load_dict = {}
    for k, v in pretrained_state_dict.items():
        if 'actor.embedder.layers' in k:
            load_dict[k.replace('module.weight', 'weight')] = v
        else:
            load_dict[k.replace('module.', '')] = v

    load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
    load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
    for k, v in load_dict.items():
        if len(v.size()) <= 3:
            load_dict[k] = v.squeeze(dim=-1)
    upper_policy.load_state_dict(load_dict, strict=True)
    print('Loading pre-train upper model', load_path)
    return upper_policy


def get_args_heuristic():
    parser = argparse.ArgumentParser(description='Heuristic baseline arguments')

    parser.add_argument('--continuous', action='store_true', help='Use continuous enviroment, otherwise the enviroment is discrete')
    parser.add_argument('--setting', type=int, default=2, help='Experiment setting, please see our paper for details')
    # parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of episodes evaluated')
    parser.add_argument('--load-dataset', action='store_true', help='Load an existing dataset, otherwise the data is generated on the fly')
    parser.add_argument('--dataset-path', type=str, help='The path to load dataset')
    parser.add_argument('--heuristic', type=str, default='LSAH', help='Options: LSAH DBL MACS OnlineBPH HM BR RANDOM')


    args = parser.parse_args()
    args.container_size = packing_config.container_size
    args.item_size_set  = packing_config.item_size_set
    args.evaluate = True

    if args.continuous:
        assert args.heuristic == 'LSAH' or args.heuristic == 'OnlineBPH' or args.heuristic == 'BR', 'only LSAH, OnlineBPH, and BR allowed for continuous environment'

    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    if args.evaluate:
        args.num_processes = 1
    args.normFactor = 1.0 / np.max(args.container_size)

    return args

def registration_envs():
    import gym
    for name, target in [
        ('PctDiscrete-v0', 'pct_envs.PctDiscrete0:PackingDiscrete'),
        ('PctDiscrete-v1', 'pct_envs.PctDiscrete0:PackingDiscreteV1'),
        ('PctContinuous-v0', 'pct_envs.PctContinuous0:PackingContinuous'),
        ('PctContinuous-v1', 'pct_envs.PctContinuous0:PackingContinuousV1'),
    ]:
        try:
            gym.spec(name)
        except gym.error.Error:
            register(id=name, entry_point=target)


def load_dataset(save_path, if_sol = False):

    idx_place = 3 if if_sol else 1

    def find_idx(name):
        return int(name.split('-')[idx_place].split('.')[0])

    if if_sol:
        sols = []
    else:
        distributions = []
        instances = []

    data_path = [name for name in os.listdir(save_path) if name.endswith('.pt')
                 and (if_sol or name.startswith('instance-'))]
    data_path.sort(key=find_idx)
        
    for i in range(len(data_path)):

        data = torch.load(os.path.join(save_path,data_path[i]), map_location='cpu')
        if if_sol:
            sols.append(data)
        else:
            distributions.append(data['dist'])
            instances.append(data['ins'])            

    if if_sol:
        sols = torch.stack(sols,dim=0)
        return sols
    else:
        distributions = torch.stack(distributions,dim=0)
        instances = torch.stack(instances,dim=0)
        return distributions, instances
    
def load_multiple_datasets(save_path, seq_len_set, sol_tag = None):

    if_sol = sol_tag is not None
    tag = 'num-box' if not if_sol else '{}-num-box'.format(sol_tag)
        

    if if_sol:
        sols = {}
    else:
        distributions = {}
        instances = {}

    for i in range(len(seq_len_set)):

        seq_len = seq_len_set[i]
        cur_save_path = os.path.join(save_path,'{}-{}'.format(tag, seq_len))

        if if_sol:
            cur_sols = load_dataset(cur_save_path, if_sol)
            sols[seq_len] = cur_sols
        else:
            cur_distributions, cur_instances = load_dataset(cur_save_path, if_sol)
            distributions[seq_len] = cur_distributions
            instances[seq_len] = cur_instances

    if if_sol:
        return sols
    else:
        return distributions, instances, torch.load(os.path.join(save_path,'box-set.pt'), map_location='cpu')

def save_sol(save_path, sol, seq_len, idx, tag = 'MCTS-sol'):

    save_path = os.path.join(save_path, '{}-num-box-{}'.format(tag, seq_len))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(
        sol,
        os.path.join(save_path, '{}-instance-{}.pt'.format(tag, idx))
    )


def load_model_mcts(policy,model_save_path, tag, sub_time_str):

    policy_model_state_dict = torch.load(os.path.join(model_save_path, '{}-{}.pt'.format(tag, sub_time_str)))
    load_dict = {}
    for k, v in policy_model_state_dict.items():
        if 'actor.embedder.layers' in k:
            load_dict[k.replace('module.weight', 'weight')] = v
        else:
            load_dict[k.replace('module.', '')] = v

    load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
    load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
    for k, v in load_dict.items():
        if len(v.size()) <= 3:
            load_dict[k] = v.squeeze(dim=-1)
    policy.load_state_dict(load_dict, strict=True)