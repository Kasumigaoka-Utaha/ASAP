import gym
import numpy as np
import torch

import os
import multiprocessing as mp
from wrapper.monitor import Monitor
from wrapper.vec_env import VecEnvWrapper
from wrapper.shmem_vec_env import ShmemVecEnv
from wrapper.dummy_vec_env import DummyVecEnv


def make_env(env_id, seed, rank, log_dir, allow_early_resets, args):
    def _thunk():
        from tools import registration_envs
        registration_envs()
        env = gym.make(
                       env_id,
                       setting = args.setting,
                       container_size = args.container_size,
                       item_set = args.item_size_set,
                       data_name = args.dataset_path,
                       load_test_data = args.load_dataset,
                       internal_node_holder = args.internal_node_holder,
                       leaf_node_holder = args.leaf_node_holder,
                       next_holder = args.next_holder,
                       LNES = args.lnes,
                       shuffle = args.shuffle,
                       keep_prev = args.keep_prev,
                       sample_from_distribution = args.sample_from_distribution,
                       sample_left_bound = args.sample_left_bound,
                       sample_right_bound = args.sample_right_bound
                       )
        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)
        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)
        return env
    return _thunk

def make_vec_envs(args,
                  log_dir,
                  allow_early_resets):

    env_name = args.id
    seed = args.seed
    num_processes = args.num_processes
    device = args.device
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
    print(args.id)
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, args)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        # fork is unavailable on Windows; CloudpickleWrapper also supports spawn.
        context = 'fork' if 'fork' in mp.get_all_start_methods() else 'spawn'
        envs = ShmemVecEnv(envs, context=context)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)


    return envs

class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self,ins_pack):
        obs = self.venv.reset(ins_pack)
        obs = torch.from_numpy(np.array(obs)).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)

        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(np.array(obs)).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
