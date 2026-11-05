import os
import torch.optim as optim
import numpy as np
import tools
import time
from collections import deque
from storage import PCTRolloutStorage
from kfac import KFACOptimizer
import random
import torch
import torch.nn as nn
#np.set_printoptions(threshold=np.inf)

class ASAPTrainer(object):

    def __init__(self, writer, timeStr, selection_policy, proposal_head, ins_policy, args):

        self.writer = writer
        self.timeStr = timeStr
        self.step_counter = 0
        self.selection_policy = selection_policy
        self.ins_policy = ins_policy
        self.proposal_head = proposal_head

        # self.proposal_head_optim = optim.Adam(self.proposal_head.parameters(), lr=args.learning_rate)
        self.proposal_head_optim = KFACOptimizer(self.proposal_head)
        self.selection_policy_optim = KFACOptimizer(self.selection_policy)

        self.factor = args.normFactor

        self.proposal_head_save_que = []
        self.selection_policy_save_que = []
        self.train_steps = 0
        self.num_processes = args.num_processes
        self.embed_dim = args.embedding_size

        self.proposal_rollout = PCTRolloutStorage(args.num_steps,
                                        args.num_processes,
                                        obs_shape=(args.internal_node_holder+args.leaf_node_holder+args.next_holder, 9),
                                        gamma = args.gamma)

        self.selection_rollout = PCTRolloutStorage(args.num_steps,
                                        args.num_processes,
                                        obs_shape=(args.internal_node_holder+args.search_num+args.next_holder, 9),
                                        gamma = args.gamma)
        

    def train_n_steps(self, envs, args, train_steps, num_episode, device, log_dir, dataset, test_steps, seq_len_list, box_set_list, change_box_set=False):
        
        self.start = time.time()
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
        with open(os.path.join(log_dir,'log.txt'),'a') as file:
            for i in range(num_episode):

                if change_box_set:
                    self.train_steps += 2*train_steps
                    self.train_proposal(envs, args, 2*train_steps, device, args.num_steps, box_set_list[0], change_box_set)
                    self.train_steps += 2*train_steps
                    self.train_selection_policy(envs, args, 2*train_steps, device, args.num_steps, box_set_list[0], change_box_set)
                else:
                    for j in range(len(box_set_list)):
                        self.train_steps += train_steps
                        self.train_proposal(envs, args, train_steps, device, args.num_steps, box_set_list[j])
                        # self.train_selection_policy(envs, args, train_steps, device, args.num_steps, box_set_list[j])

                    for j in range(len(box_set_list)):
                        self.train_steps += train_steps
                        # self.train_proposal(envs, args, train_steps, device, args.num_steps, box_set_list[j])
                        self.train_selection_policy(envs, args, train_steps, device, args.num_steps, box_set_list[j])

                if i % 2 == 1: # evaluate after 5 iterations
                    for j in range(len(box_set_list)):
                        ratio, counter = self.evaluate(envs, args, test_steps, device, seq_len_list[j], box_set = box_set_list[j], dataset = dataset[j]) 

                        episodes_eval_results = \
                            "\nEvaluation on training epoch {}\n"\
                            "Test BPP policy on dataset{} with {} instances containing {} boxes\n" \
                            "Mean/Median Ratio {:.3f}/{:.3f}, Min/Max Ratio {:.3f}/{:.3f}\n" \
                            "Mean/Median Counter {:.1f}/{:.1f}, Min/Max Counter {:.1f}/{:.1f}\n" \
                                .format(i, j, test_steps, seq_len_list[j],
                                        np.mean(ratio), np.median(ratio),
                                        np.min(ratio), np.max(ratio),
                                        np.mean(counter), np.median(counter),
                                        np.min(counter), np.max(counter),
                                        )
                        file.write(episodes_eval_results)
                        print(episodes_eval_results)
                        self.writer.add_scalar("BPP/Ratio/dataset_{}/Mean".format(i), np.mean(ratio), i)
                        self.writer.add_scalar("BPP/Ratio/dataset_{}/Max".format(i), np.max(ratio), i)
                        self.writer.add_scalar("BPP/Ratio/dataset_{}/Min".format(i), np.min(ratio), i)
                        self.writer.add_scalar('BPP/Counter/dataset_{}/Mean'.format(i), np.mean(counter), i)
                        self.writer.add_scalar('BPP/Counter/dataset_{}/Max'.format(i), np.max(counter), i)
                        self.writer.add_scalar('BPP/Counter/dataset_{}/Min'.format(i), np.min(counter), i)

                    self.save_model(
                        model_save_path, self.proposal_head, sub_time_str,
                        self.proposal_head_save_que, args.max_model_num,
                        tag='proposal'
                    )

                    self.save_model(
                        model_save_path, self.selection_policy, sub_time_str,
                        self.selection_policy_save_que, args.max_model_num,
                        tag='selection'
                    )

    def train_n_steps_proposal(self, envs, args, train_steps, num_episode, device, log_dir, dataset, test_steps, seq_len_list, box_set_list, change_box_set=False):
        
        self.start = time.time()
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
        with open(os.path.join(log_dir,'log.txt'),'a') as file:
            for i in range(num_episode):

                for j in range(len(box_set_list)):
                    box_set = box_set_list[0] if change_box_set else box_set_list[j]
                    self.train_steps += train_steps
                    self.train_proposal(envs, args, train_steps, device, args.num_steps, box_set, change_box_set)

                if i % 2 == 1: # evaluate after 5 iterations
                    for j in range(len(box_set_list)):
                        ratio, counter = self.evaluate(envs, args, test_steps, device, seq_len_list[j], box_set = box_set_list[j], dataset = dataset[j]) 

                        episodes_eval_results = \
                            "\nEvaluation on training epoch {}\n"\
                            "Test BPP policy on dataset{} with {} instances containing {} boxes\n" \
                            "Mean/Median Ratio {:.3f}/{:.3f}, Min/Max Ratio {:.3f}/{:.3f}\n" \
                            "Mean/Median Counter {:.1f}/{:.1f}, Min/Max Counter {:.1f}/{:.1f}\n" \
                                .format(i, j, test_steps, seq_len_list[j],
                                        np.mean(ratio), np.median(ratio),
                                        np.min(ratio), np.max(ratio),
                                        np.mean(counter), np.median(counter),
                                        np.min(counter), np.max(counter),
                                        )
                        file.write(episodes_eval_results)
                        print(episodes_eval_results)
                        self.writer.add_scalar("BPP/Ratio/dataset_{}/Mean".format(i), np.mean(ratio), i)
                        self.writer.add_scalar("BPP/Ratio/dataset_{}/Max".format(i), np.max(ratio), i)
                        self.writer.add_scalar("BPP/Ratio/dataset_{}/Min".format(i), np.min(ratio), i)
                        self.writer.add_scalar('BPP/Counter/dataset_{}/Mean'.format(i), np.mean(counter), i)
                        self.writer.add_scalar('BPP/Counter/dataset_{}/Max'.format(i), np.max(counter), i)
                        self.writer.add_scalar('BPP/Counter/dataset_{}/Min'.format(i), np.min(counter), i)

                    self.save_model(
                        model_save_path, self.proposal_head, sub_time_str,
                        self.proposal_head_save_que, args.max_model_num,
                        tag='proposal'
                    )
    
    def train_n_steps_selection(self, envs, args, train_steps, num_episode, device, log_dir, dataset, test_steps, seq_len_list, box_set_list, change_box_set=False):
        
        self.start = time.time()
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
        with open(os.path.join(log_dir,'log.txt'),'a') as file:
            for i in range(num_episode):

                for j in range(len(box_set_list)):
                    box_set = box_set_list[0] if change_box_set else box_set_list[j]
                    self.train_steps += train_steps
                    self.train_selection_policy(envs, args, train_steps, device, args.num_steps, box_set, change_box_set)
                

                if i % 2 == 1: # evaluate after 2 iterations
                    for j in range(len(box_set_list)):
                        ratio, counter = self.evaluate(envs, args, test_steps, device, seq_len_list[j], box_set = box_set_list[j], dataset = dataset[j]) 

                        episodes_eval_results = \
                            "\nEvaluation on training epoch {}\n"\
                            "Test BPP policy on dataset{} with {} instances containing {} boxes\n" \
                            "Mean/Median Ratio {:.3f}/{:.3f}, Min/Max Ratio {:.3f}/{:.3f}\n" \
                            "Mean/Median Counter {:.1f}/{:.1f}, Min/Max Counter {:.1f}/{:.1f}\n" \
                                .format(i, j, test_steps, seq_len_list[j],
                                        np.mean(ratio), np.median(ratio),
                                        np.min(ratio), np.max(ratio),
                                        np.mean(counter), np.median(counter),
                                        np.min(counter), np.max(counter),
                                        )
                        file.write(episodes_eval_results)
                        print(episodes_eval_results)
                        self.writer.add_scalar("BPP/Ratio/dataset_{}/Mean".format(i), np.mean(ratio), i)
                        self.writer.add_scalar("BPP/Ratio/dataset_{}/Max".format(i), np.max(ratio), i)
                        self.writer.add_scalar("BPP/Ratio/dataset_{}/Min".format(i), np.min(ratio), i)
                        self.writer.add_scalar('BPP/Counter/dataset_{}/Mean'.format(i), np.mean(counter), i)
                        self.writer.add_scalar('BPP/Counter/dataset_{}/Max'.format(i), np.max(counter), i)
                        self.writer.add_scalar('BPP/Counter/dataset_{}/Min'.format(i), np.min(counter), i)

                    self.save_model(
                        model_save_path, self.selection_policy, sub_time_str,
                        self.selection_policy_save_que, args.max_model_num,
                        tag='selection'
                    )

    def load_proposal_model(self,model_save_path, tag, sub_time_str):
        print('load proposal model...')

        policy_model_state_dict = torch.load(os.path.join(model_save_path, '{}-{}.pt'.format(tag, sub_time_str)))
        if self.proposal_head_optim is not None:
            load_dict = policy_model_state_dict
        else:
            load_dict = {}
            for k, v in policy_model_state_dict.items():
                if 'actor.embedder.layers' in k:
                    load_dict[k.replace('module.weight', 'weight')] = v
                else:
                    load_dict[k.replace('module.', '')] = v

            load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
            load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
            for k, v in load_dict.items():
                if len(v.size()) <= 3 and len(v.size()) > 1:
                    load_dict[k] = v.squeeze(dim=-1)
        self.proposal_head.load_state_dict(load_dict, strict=True)


    def train_proposal(self, envs, args, train_steps, device, seq_len, box_set, change_box_set = False):

        self.proposal_head.train()

        self.selection_policy.eval()
        self.ins_policy.eval()

        num_steps, num_processes = seq_len, args.num_processes

        batchX = torch.arange(args.num_processes).to(device)

        # sample box set every epoch

        current_box_set = box_set

        if change_box_set:
            current_box_set = list(box_set)
            random.shuffle(current_box_set)
            current_box_set = current_box_set[0:np.random.randint(15,35)]
            current_box_set = torch.stack(current_box_set,dim=0)
            num_steps = seq_len
        
        for i in range(train_steps): 

            # sample from random distribution
            distribution = torch.rand((current_box_set.size(0),)).repeat((args.num_processes,)).view((args.num_processes,current_box_set.size(0))).to(device)
            distribution = distribution/torch.sum(distribution,dim=1).view((-1,1))
            distribution_set = []

            # env reset
            done_mask = torch.zeros(args.num_processes).to(device).bool()
            reward_array = torch.zeros(args.num_processes).to(device)
            step_counter_array = torch.zeros(args.num_processes).to(device)
            with torch.no_grad():
                ins, _ = self.ins_policy(current_box_set,num_steps,args.num_processes,distribution,deterministic=False, random_mode=True, continuous = args.continuous)
            # print(ins.size())
            obs = envs.reset(ins)
            all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
            all_nodes = all_nodes.to(device)
            leaf_node_mask = torch.zeros(args.num_processes).to(device).bool()
            prev_all_nodes = all_nodes.clone()
            prev_action = None
            prev_action_log_probs = None
            prev_value = None
            self.proposal_rollout.obs[0].copy_(all_nodes)


            for step in range(num_steps):
                # print(step)
                input_box_set = current_box_set.repeat((args.num_processes,1,1)).view((args.num_processes,current_box_set.size(0),3)).to(device)
                input_distribution = torch.cat((input_box_set,distribution.unsqueeze(dim=2)),dim=2)
                distribution_set.append(input_distribution)
                with torch.no_grad():
                    action_log_probs, _, _, value, dist, _ = self.proposal_head(all_nodes, normFactor = self.factor)

                _, indices = torch.topk(dist.probs, args.search_num, dim=1)
                all_nodes, leaf_nodes = tools.process_search_space(all_nodes,args.internal_node_holder,args.leaf_node_holder,indices)
                
                with torch.no_grad():
                    if args.allow_dist_input:
                        _, action , _, _ , _, _ = self.selection_policy(all_nodes, input_distribution, normFactor = self.factor)
                    else:
                        _, action , _, _ , _, _ = self.selection_policy(all_nodes, normFactor = self.factor)

                # Map the selection index back to the proposal action space.
                proposal_action = indices[batchX, action.squeeze(-1)].unsqueeze(-1)
                # process next step
                selected_leaf_node = leaf_nodes[batchX,action.squeeze()]
                obs, reward, done, _ = envs.step(selected_leaf_node.cpu().numpy())
                all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
                all_nodes = all_nodes.to(device)
                done = torch.tensor(done).to(device)

                # recorder
                normal_idx = (done_mask == False) & (done == False)
                reward_array = reward_array.clone()
                reward_array[normal_idx] = reward_array[normal_idx] + reward.reshape(-1).to(device)[normal_idx]
                step_counter_array[normal_idx] += 1
                done_mask[done_mask==False] = done[done_mask==False]

                # update step info
                if prev_action is not None:
                    prev_action = prev_action.clone()
                    prev_action[leaf_node_mask==False] = proposal_action[leaf_node_mask==False]
                else:
                    prev_action = proposal_action
                if prev_action_log_probs is not None:
                    prev_action_log_probs = prev_action_log_probs.clone()
                    prev_action_log_probs[leaf_node_mask==False] = action_log_probs[leaf_node_mask==False]
                else:
                    prev_action_log_probs = action_log_probs
                if prev_value is not None:
                    prev_value = prev_value.clone()
                    prev_value[leaf_node_mask==False] = value[leaf_node_mask==False]
                    prev_value[leaf_node_mask] = 0 # value comes to 0 if there's no action to perform
                else:
                    prev_value = value
                reward = reward.clone()
                reward[leaf_node_mask] = 0

                # update step+1 info
                temp_mask = (torch.sum(leaf_nodes[:,:,-1],dim=1)==0)
                leaf_node_mask = leaf_node_mask.clone()
                leaf_node_mask[leaf_node_mask==False] = temp_mask[leaf_node_mask==False]
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in leaf_node_mask])
                prev_all_nodes = prev_all_nodes.clone()
                prev_all_nodes[leaf_node_mask==False] = all_nodes[leaf_node_mask==False]
                
                self.proposal_rollout.insert(prev_all_nodes, prev_action, prev_action_log_probs, reward, masks)


            with torch.no_grad():
                input_box_set = current_box_set.repeat((args.num_processes,1,1)).view((args.num_processes,current_box_set.size(0),3)).to(device)
                input_distribution = torch.cat((input_box_set,distribution.unsqueeze(dim=2)),dim=2)
                distribution_set.append(input_distribution)
                _,_,_,next_value,_,_ = self.proposal_head(self.proposal_rollout.obs[-1].to(device), normFactor = self.factor)

            self.proposal_rollout.compute_returns(next_value)

            obs_shape = self.proposal_rollout.obs.size()[2:]
            action_shape = self.proposal_rollout.actions.size()[-1]

            leaf_node_value, selectedlogProb, _, dist = self.proposal_head.evaluate(self.proposal_rollout.obs[:num_steps].view(-1, *obs_shape).to(device),
                                                                                        self.proposal_rollout.actions[:num_steps].view(-1, action_shape).to(device),
                                                                                        normFactor=self.factor)                                                                          
            
            leaf_node_value = leaf_node_value.view(num_steps, num_processes, 1)
            selectedlogProb = selectedlogProb.view(num_steps, num_processes, 1)

            advantages = self.proposal_rollout.returns[:num_steps].to(device) - leaf_node_value
            critic_loss = advantages.pow(2).mean()
            actor_loss  = -(advantages.detach() * selectedlogProb).mean()

            if self.proposal_head_optim.steps % self.proposal_head_optim.Ts == 0:
                # Sampled fisher, see Martens 2014d
                self.proposal_head.zero_grad()
                pg_fisher_loss = - selectedlogProb.mean()

                value_noise = torch.randn(leaf_node_value.size())
                if leaf_node_value.is_cuda:
                    value_noise = value_noise.to(device)

                sample_values = leaf_node_value + value_noise
                vf_fisher_loss = -(leaf_node_value - sample_values.detach()).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                self.proposal_head_optim.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.proposal_head_optim.acc_stats = False

            self.proposal_head_optim.zero_grad()
            (args.actor_loss_coef * actor_loss
             + args.critic_loss_coef  * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.proposal_head.parameters(), args.max_grad_norm)
            self.proposal_head_optim.step()

            self.proposal_rollout.after_update()

    def train_selection_policy(self, envs, args, train_steps, device, seq_len, box_set, change_box_set = False):

        self.proposal_head.eval()

        self.selection_policy.train()
        self.ins_policy.eval()

        num_steps, num_processes = seq_len, args.num_processes

        batchX = torch.arange(args.num_processes).to(device)

        # sample box set every epoch

        current_box_set = box_set
        if change_box_set:
            current_box_set = list(box_set)
            random.shuffle(current_box_set)
            current_box_set = current_box_set[0:np.random.randint(15,35)]
            current_box_set = torch.stack(current_box_set,dim=0)
            num_steps = seq_len
        
        for i in range(train_steps): 

            # sample from random distribution
            distribution = torch.rand((current_box_set.size(0),)).repeat((args.num_processes,)).view((args.num_processes,current_box_set.size(0))).to(device)
            distribution = distribution/torch.sum(distribution,dim=1).view((-1,1))
            distribution_set = []

            # env reset
            done_mask = torch.zeros(args.num_processes).to(device).bool()
            reward_array = torch.zeros(args.num_processes).to(device)
            step_counter_array = torch.zeros(args.num_processes).to(device)
            with torch.no_grad():
                ins, _ = self.ins_policy(current_box_set,num_steps,args.num_processes,distribution,deterministic=False, random_mode=True, continuous = args.continuous)
            obs = envs.reset(ins)
            all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
            all_nodes = all_nodes.to(device)
            with torch.no_grad():
                _, _, _, _, dist, _ = self.proposal_head(all_nodes, normFactor = self.factor)

            _, indices = torch.topk(dist.probs, args.search_num, dim=1)
            all_nodes, leaf_nodes = tools.process_search_space(all_nodes,args.internal_node_holder,args.leaf_node_holder,indices)

            leaf_node_mask = torch.zeros(args.num_processes).to(device).bool()
            prev_all_nodes = all_nodes.clone()
            prev_action = None
            prev_action_log_probs = None
            prev_value = None
            self.selection_rollout.obs[0].copy_(all_nodes)


            for step in range(num_steps):
                # print(step)
                input_box_set = current_box_set.repeat((args.num_processes,1,1)).view((args.num_processes,current_box_set.size(0),3)).to(device)
                input_distribution = torch.cat((input_box_set,distribution.unsqueeze(dim=2)),dim=2)
                distribution_set.append(input_distribution)
                
                with torch.no_grad():
                    if args.allow_dist_input:
                        action_log_probs, action, entropy, value, _, _ = self.selection_policy(all_nodes, input_distribution, normFactor = self.factor)
                    else:
                        action_log_probs, action, entropy, value, _, _ = self.selection_policy(all_nodes, normFactor = self.factor)

                # process next step
                selected_leaf_node = leaf_nodes[batchX,action.squeeze()]
                obs, reward, done, infos = envs.step(selected_leaf_node.cpu().numpy())
                all_nodes, _ = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
                all_nodes = all_nodes.to(device)
                with torch.no_grad():
                    _, _, _, _, dist, _ = self.proposal_head(all_nodes, normFactor = self.factor)

                _, indices = torch.topk(dist.probs, args.search_num, dim=1)
                all_nodes, leaf_nodes = tools.process_search_space(all_nodes,args.internal_node_holder,args.leaf_node_holder,indices)
                done = torch.tensor(done).to(device)

                # recorder
                normal_idx = (done_mask == False) & (done == False)
                reward_array = reward_array.clone()
                reward_array[normal_idx] = reward_array[normal_idx] + reward.reshape(-1).to(device)[normal_idx]
                step_counter_array[normal_idx] += 1
                done_mask[done_mask==False] = done[done_mask==False]

                # update step info
                if prev_action is not None:
                    prev_action = prev_action.clone()
                    prev_action[leaf_node_mask==False] = action[leaf_node_mask==False]
                else:
                    prev_action = action
                if prev_action_log_probs is not None:
                    prev_action_log_probs = prev_action_log_probs.clone()
                    prev_action_log_probs[leaf_node_mask==False] = action_log_probs[leaf_node_mask==False]
                else:
                    prev_action_log_probs = action_log_probs
                if prev_value is not None:
                    prev_value = prev_value.clone()
                    prev_value[leaf_node_mask==False] = value[leaf_node_mask==False]
                    prev_value[leaf_node_mask] = 0 # value comes to 0 if there's no action to perform
                else:
                    prev_value = value
                reward = reward.clone()
                reward[leaf_node_mask] = 0

                # update step+1 info
                temp_mask = (torch.sum(leaf_nodes[:,:,-1],dim=1)==0)
                leaf_node_mask = leaf_node_mask.clone()
                leaf_node_mask[leaf_node_mask==False] = temp_mask[leaf_node_mask==False]
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in leaf_node_mask])
                prev_all_nodes = prev_all_nodes.clone()
                prev_all_nodes[leaf_node_mask==False] = all_nodes[leaf_node_mask==False]
                
                # if self.use_meta:
                #     self.pct_rollout.insert(prev_all_nodes, prev_action, prev_action_log_probs, reward, masks, self.context_vec)
                # else:
                #     self.pct_rollout.insert(prev_all_nodes, prev_action, prev_action_log_probs, reward, masks)
                self.selection_rollout.insert(prev_all_nodes, prev_action, prev_action_log_probs, reward, masks)


            with torch.no_grad():
                input_box_set = current_box_set.repeat((args.num_processes,1,1)).view((args.num_processes,current_box_set.size(0),3)).to(device)
                input_distribution = torch.cat((input_box_set,distribution.unsqueeze(dim=2)),dim=2)
                distribution_set.append(input_distribution)
                if args.allow_dist_input:
                    _, _, _, next_value, _, _ = self.selection_policy(self.selection_rollout.obs[-1].to(device), input_distribution, normFactor = self.factor)
                else:
                    _, _, _, next_value, _, _ = self.selection_policy(self.selection_rollout.obs[-1].to(device), normFactor = self.factor)

            self.selection_rollout.compute_returns(next_value)

            obs_shape = self.selection_rollout.obs.size()[2:]
            action_shape = self.selection_rollout.actions.size()[-1]
            distribution_set = torch.stack(distribution_set,dim=0).view((-1,current_box_set.size(0),4)) 

            if args.allow_dist_input:
                leaf_node_value, selectedlogProb, dist_entropy, dist = self.selection_policy.evaluate(self.selection_rollout.obs[:num_steps].view(-1, *obs_shape).to(device),
                                                                                            self.selection_rollout.actions[:num_steps].view(-1, action_shape).to(device),
                                                                                            distribution_set[:num_steps].to(device),
                                                                                            normFactor=self.factor)  
            else:
                leaf_node_value, selectedlogProb, dist_entropy, dist = self.selection_policy.evaluate(self.selection_rollout.obs[:num_steps].view(-1, *obs_shape).to(device),
                                                                                            self.selection_rollout.actions[:num_steps].view(-1, action_shape).to(device),
                                                                                            normFactor=self.factor)                                                                          
            
            leaf_node_value = leaf_node_value.view(num_steps, num_processes, 1)
            selectedlogProb = selectedlogProb.view(num_steps, num_processes, 1)

            advantages = self.selection_rollout.returns[:num_steps].to(device) - leaf_node_value
            critic_loss = advantages.pow(2).mean()
            actor_loss  = -(advantages.detach() * selectedlogProb).mean()

            if self.selection_policy_optim.steps % self.selection_policy_optim.Ts == 0:
                # Sampled fisher, see Martens 2014d
                self.selection_policy.zero_grad()
                pg_fisher_loss = - selectedlogProb.mean()

                value_noise = torch.randn(leaf_node_value.size())
                if leaf_node_value.is_cuda:
                    value_noise = value_noise.to(device)

                sample_values = leaf_node_value + value_noise
                vf_fisher_loss = -(leaf_node_value - sample_values.detach()).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                self.selection_policy_optim.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.selection_policy_optim.acc_stats = False

            self.selection_policy_optim.zero_grad()
            (args.actor_loss_coef * actor_loss
            + args.critic_loss_coef  * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.selection_policy.parameters(), args.max_grad_norm)
            self.selection_policy_optim.step()

            self.selection_rollout.after_update()


    def save_model(self, model_save_path, model, sub_time_str, model_save_que, max_model_num, tag):

        if model_save_path != "":
            sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

            if sub_time_str not in model_save_que:
                model_save_que.append(sub_time_str)

            if len(model_save_que) > max_model_num:
                rm_model = model_save_que.pop(0)
                os.remove(os.path.join(model_save_path, '{}-{}.pt'.format(tag, rm_model)))

            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            torch.save(
                model.state_dict(),
                os.path.join(model_save_path, '{}-{}.pt'.format(tag, sub_time_str))
            )
        return      

    def load_selection_model(self,model_save_path, tag, sub_time_str):
        print('load action model...')

        policy_model_state_dict = torch.load(os.path.join(model_save_path, '{}-{}.pt'.format(tag, sub_time_str)))
        if self.selection_policy_optim is not None:
            load_dict = policy_model_state_dict
        else:
            load_dict = {}
            for k, v in policy_model_state_dict.items():
                if 'actor.embedder.layers' in k:
                    load_dict[k.replace('module.weight', 'weight')] = v
                else:
                    load_dict[k.replace('module.', '')] = v

            load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
            load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
            for k, v in load_dict.items():
                if len(v.size()) <= 3 and len(v.size()) > 1:
                    load_dict[k] = v.squeeze(dim=-1)
        self.selection_policy.load_state_dict(load_dict, strict=True)

    def evaluate(self, envs, args, test_steps, device, seq_len, box_set = None, dataset = None):

        self.selection_policy.eval()
        self.ins_policy.eval()
        self.proposal_head.eval()

        if dataset is not None:
            num_steps, num_processes = seq_len, args.num_processes
        else:
            num_steps, num_processes = args.num_steps, args.num_processes

        box_set = self.default_box_set if box_set is None else box_set
        batchX = torch.arange(num_processes).to(device)

        ratio_set = []
        counter_set = []
        
        for i in range(test_steps):
            
            if dataset is not None:

                distribution = dataset['distributions'][i]
                ins = dataset['instances'][i]

            else:
                # sample from random distribution
                distribution = torch.rand((box_set.size(0),)).repeat((num_processes,)).view((num_processes,box_set.size(0))).to(device)
                distribution = distribution/torch.sum(distribution,dim=1).view((-1,1))
                with torch.no_grad():
                    ins, _ = self.ins_policy(box_set,num_steps,num_processes,distribution,deterministic=False, random_mode=True, continuous = args.continuous)

            # env reset
            done_mask = torch.zeros(args.num_processes).to(device).bool()
            reward_array = torch.zeros(args.num_processes).to(device)
            step_counter_array = torch.zeros(args.num_processes).to(device)
            out_path = []

            obs = envs.reset(ins)
            all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
            all_nodes = all_nodes.to(device)

            with torch.no_grad():
                _, _, _, _, dist, _ = self.proposal_head(all_nodes, normFactor = self.factor)
            _, indices = torch.topk(dist.probs, args.search_num, dim=1)
            all_nodes, leaf_nodes = tools.process_search_space(all_nodes,args.internal_node_holder,args.leaf_node_holder,indices)
            leaf_node_mask = torch.zeros(args.num_processes).to(device).bool()

            for step in range(num_steps):
                # print(step)
                with torch.no_grad():
                    if args.allow_dist_input:
                        input_box_set = box_set.repeat((args.num_processes,1,1)).view((args.num_processes,box_set.size(0),3)).to(device)
                        input_distribution = torch.cat((input_box_set,distribution.unsqueeze(dim=2)),dim=2)
                        action_log_probs, action, entropy, value,_,_ = self.selection_policy(all_nodes, input_distribution, normFactor = self.factor)
                    else:
                        action_log_probs, action, entropy, value,_,_ = self.selection_policy(all_nodes, normFactor = self.factor)

                # process next step
                selected_leaf_node = leaf_nodes[batchX,action.squeeze()]
                out_path.append(selected_leaf_node)
                obs, reward, done, infos = envs.step(selected_leaf_node.cpu().numpy())
                all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
                all_nodes = all_nodes.to(device)
                with torch.no_grad():
                    _, _, _, _, dist, _ = self.proposal_head(all_nodes, normFactor = self.factor)
                _, indices = torch.topk(dist.probs, args.search_num, dim=1)
                all_nodes, leaf_nodes = tools.process_search_space(all_nodes,args.internal_node_holder,args.leaf_node_holder,indices)

                done = torch.tensor(done).to(device)

                # recorder
                normal_idx = (done_mask == False) & (done == False)
                reward_array = reward_array.clone()
                reward_array[normal_idx] = reward_array[normal_idx] + reward.reshape(-1).to(device)[normal_idx]
                step_counter_array[normal_idx] += 1
                done_mask[done_mask==False] = done[done_mask==False]

                # update step info
                reward = reward.clone()
                reward[leaf_node_mask] = 0

                # update step+1 info
                temp_mask = (torch.sum(leaf_nodes[:,:,-1],dim=1)==0)
                leaf_node_mask = leaf_node_mask.clone()
                leaf_node_mask[leaf_node_mask==False] = temp_mask[leaf_node_mask==False]
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in leaf_node_mask])

            reward_array = np.array(reward_array.cpu())
            step_counter_array = np.array(step_counter_array.cpu())
            ratio_array = reward_array/10


            total_num_steps = self.train_steps
            end = time.time()


            ratio_set.append(ratio_array)
            counter_set.append(step_counter_array)
        

        ratio_set = np.array(ratio_set)
        counter_set = np.array(counter_set)
        return ratio_set, counter_set

