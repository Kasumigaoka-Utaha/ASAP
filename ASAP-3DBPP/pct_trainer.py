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




class PCTTrainer(object):
    # baseline training methods
    def __init__(self, writer, timeStr, PCT_policy, ins_policy, args, teacher_PCT_policy = None):

        self.writer = writer
        self.timeStr = timeStr
        self.step_counter = 0
        self.PCT_policy = PCT_policy
        self.ins_policy = ins_policy
        self.pct_rollout = PCTRolloutStorage(args.num_steps,
                                        args.num_processes,
                                        obs_shape=(args.internal_node_holder+args.leaf_node_holder+args.next_holder, 9),
                                        gamma = args.gamma)
        if teacher_PCT_policy is not None:
            self.bc_training = True
            self.teacher_PCT_policy = teacher_PCT_policy
        else:
            self.bc_training = False
        
        self.use_acktr = args.use_acktr
        self.default_box_set = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.default_box_set.append((1+i, 1+j, 1+k))
        self.default_box_set = torch.tensor(self.default_box_set)
        self.factor = args.normFactor

        self.policy_optim = KFACOptimizer(self.PCT_policy)

        self.pct_model_save_que = []
        self.train_steps = 0


    def train_n_steps_with_eval(self, envs, args, train_steps, num_episode, device, log_dir, dataset, test_steps, seq_len_list, box_set_list, default_mod=False):
        # print(log_dir)
        self.start = time.time()
        model_save_path = os.path.join(args.model_save_path, self.timeStr)

        with open(os.path.join(log_dir,'log.txt'),'a') as file:
            for i in range(num_episode):
                sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                for j in range(len(box_set_list)):
                    box_set = box_set_list[0] if default_mod else box_set_list[j]
                    self.train_steps += train_steps
                    self.train_PCT_policy(envs, args, train_steps, device, box_set,seq_len_list[j],default_mod=default_mod)

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
                                        np.min(ratio), np.max( ratio),
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
                        self.train_steps, args.model_save_interval, args.model_update_interval,
                        model_save_path, self.PCT_policy, sub_time_str,
                        self.pct_model_save_que, args.max_model_num,
                        tag='pct'
                    )


    def train_PCT_policy(self, envs, args, train_steps, device, box_set, seq_len, default_mod=False):


        self.PCT_policy.train()
        self.ins_policy.eval()

        num_steps, num_processes = seq_len, args.num_processes

        batchX = torch.arange(args.num_processes).to(device)

        if default_mod:
            current_box_set = list(box_set)
            random.shuffle(current_box_set)
            current_box_set = current_box_set[0:np.random.randint(15,35)]
            box_set = torch.stack(current_box_set,dim=0)
            num_steps = seq_len
        
        for _ in range(train_steps):
            
            # sample from random distribution
            distribution = torch.rand((box_set.size(0),)).repeat((args.num_processes,)).view((args.num_processes,box_set.size(0))).to(device)
            distribution = distribution/torch.sum(distribution,dim=1).view((-1,1))
            distribution_set = []

            # env reset
            done_mask = torch.zeros(args.num_processes).to(device).bool()
            reward_array = torch.zeros(args.num_processes).to(device)
            step_counter_array = torch.zeros(args.num_processes).to(device)
            with torch.no_grad():
                # ins, _ = self.ins_policy(box_set,num_steps,args.num_processes,distribution,deterministic=False, random_mode=True, continuous=args.continuous)
                ins, _ = self.ins_policy(box_set,num_steps,args.num_processes,distribution,deterministic=True, random_mode=False, continuous=args.continuous)
            obs = envs.reset(ins)
            all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
            all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)
            leaf_node_mask = torch.zeros(args.num_processes).to(device).bool()
            prev_all_nodes = all_nodes.clone()
            prev_action = None
            prev_action_log_probs = None
            prev_value = None
            self.pct_rollout.obs[0].copy_(all_nodes)

            for step in range(num_steps):
                with torch.no_grad():
                    action_log_probs, action, entropy, value,_,_ = self.PCT_policy(all_nodes, normFactor = self.factor)

                # process next step
                selected_leaf_node = leaf_nodes[batchX,action.squeeze()]
                obs, reward, done, infos = envs.step(selected_leaf_node.cpu().numpy())
                all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
                all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)
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
                

                self.pct_rollout.insert(prev_all_nodes, prev_action, prev_action_log_probs, reward, masks)


            with torch.no_grad():
                # next_value = self.PCT_policy.forward_critic(self.pct_rollout.obs[-1].to(device), normFactor=self.factor)
                if args.allow_dist_input:
                    input_box_set = box_set.repeat((args.num_processes,1,1)).view((args.num_processes,box_set.size(0),3)).to(device)
                    input_distribution = torch.cat((input_box_set,distribution.unsqueeze(dim=2)),dim=2)
                    _,_,_,next_value,_,_ = self.PCT_policy(self.pct_rollout.obs[-1].to(device), input_distribution, normFactor = self.factor)
                else:
                    _,_,_,next_value,_,_ = self.PCT_policy(self.pct_rollout.obs[-1].to(device),  normFactor = self.factor)

            self.pct_rollout.compute_returns(next_value)

            if args.allow_dist_input:
                distribution_set = torch.stack(distribution_set,dim=0).view((-1,box_set.size(0),4)) 
            obs_shape = self.pct_rollout.obs.size()[2:]
            action_shape = self.pct_rollout.actions.size()[-1]
            if args.allow_dist_input:
                leaf_node_value, selectedlogProb, dist_entropy, dist = self.PCT_policy.evaluate(self.pct_rollout.obs[:num_steps].view(-1, *obs_shape).to(device),
                                                                                            self.pct_rollout.actions[:num_steps].view(-1, action_shape).to(device),
                                                                                            distribution_set.to(device),
                                                                                            normFactor=self.factor)                                                                        
            else:
                leaf_node_value, selectedlogProb, dist_entropy, dist = self.PCT_policy.evaluate(self.pct_rollout.obs[:num_steps].view(-1, *obs_shape).to(device),
                                                                                            self.pct_rollout.actions[:num_steps].view(-1, action_shape).to(device),
                                                                                            normFactor=self.factor)
            
            leaf_node_value = leaf_node_value.view(num_steps, num_processes, 1)
            selectedlogProb = selectedlogProb.view(num_steps, num_processes, 1)
            # print(selectedlogProb)

            advantages = self.pct_rollout.returns[:num_steps].to(device) - leaf_node_value
            critic_loss = advantages.pow(2).mean()
            actor_loss  = -(advantages.detach() * selectedlogProb).mean()

            if self.policy_optim.steps % self.policy_optim.Ts == 0:
                # Sampled fisher, see Martens 2014d
                self.PCT_policy.zero_grad()
                pg_fisher_loss = - selectedlogProb.mean()

                value_noise = torch.randn(leaf_node_value.size())
                if leaf_node_value.is_cuda:
                    value_noise = value_noise.to(device)

                sample_values = leaf_node_value + value_noise
                vf_fisher_loss = -(leaf_node_value - sample_values.detach()).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                self.policy_optim.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.policy_optim.acc_stats = False

            self.policy_optim.zero_grad()
            (args.actor_loss_coef * actor_loss
             + args.critic_loss_coef  * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.PCT_policy.parameters(), args.max_grad_norm)
            self.policy_optim.step()


            self.pct_rollout.after_update()

        
    


    def save_model(self, step_counter, save_interval, update_interval, model_save_path,
                   model, sub_time_str, model_save_que, max_model_num, tag):

        if step_counter % save_interval == 0 and model_save_path != "":

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


    def load_model(self,model_save_path, tag, sub_time_str):

        policy_model_state_dict = torch.load(os.path.join(model_save_path, '{}-{}.pt'.format(tag, sub_time_str)))
        self.PCT_policy.load_state_dict(policy_model_state_dict)
        # print(self.PCT_policy.state_dict())

        # encoder_model_state_dict = torch.load(os.path.join(model_save_path, '{}-{}-encoder.pt'.format(tag, sub_time_str)))
        # self.hist_encoder.load_state_dict(encoder_model_state_dict)

    def evaluate(self, envs, args, test_steps, device, seq_len, box_set = None, dataset = None):

        self.PCT_policy.eval()
        self.ins_policy.eval()

        if dataset is not None:
            num_steps, num_processes = seq_len, args.num_processes
        else:
            num_steps, num_processes = args.num_steps, args.num_processes

        box_set = self.default_box_set if box_set is None else box_set
        batchX = torch.arange(num_processes).to(device)

        # test_steps = dataset['distributions'].size(0) if dataset is not None else test_steps

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
                    ins, _ = self.ins_policy(box_set,num_steps,num_processes,distribution,deterministic=False, random_mode=True, continuous=args.continuous)

            # env reset
            done_mask = torch.zeros(num_processes).to(device).bool()
            reward_array = torch.zeros(num_processes).to(device)
            step_counter_array = torch.zeros(num_processes).to(device)

            # print(ins)
            obs = envs.reset(ins)
            all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
            all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)
            # all_nodes_for_net = all_nodes.clone()
            # all_nodes_for_net[:,:,0:6] = all_nodes[:,:,0:6]/2
            leaf_node_mask = torch.zeros(args.num_processes).to(device).bool()
            prev_all_nodes = all_nodes.clone()
            prev_action = None
            prev_action_log_probs = None
            prev_value = None
            self.pct_rollout.obs[0].copy_(all_nodes)

            for step in range(num_steps):
                # print(step)
                # choose actions
                with torch.no_grad():
                    action_log_probs, action, entropy, value, _,_ = self.PCT_policy(all_nodes, deterministic = True, normFactor = self.factor)

                # print(action)
                # process next step
                selected_leaf_node = leaf_nodes[batchX,action.squeeze()]
                # print(action)
                obs, reward, done, infos = envs.step(selected_leaf_node.cpu().numpy())
                all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
                all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)
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


            reward_array = np.array(reward_array.cpu())
            step_counter_array = np.array(step_counter_array.cpu())
            ratio_array = reward_array/10


            total_num_steps = self.train_steps
            end = time.time()

            ratio_set.append(ratio_array)
            counter_set.append(step_counter_array)
        
            # print(episodes_training_results)

        ratio_set = np.array(ratio_set)
        counter_set = np.array(counter_set)
        return ratio_set, counter_set




