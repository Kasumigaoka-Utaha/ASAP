"""Dataset loading and experiment dispatch for PCT and ASAP."""

import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from envs import make_vec_envs
from instance_generator import InstanceGenerator
from model import DRL_GAT
from tools import load_multiple_datasets, registration_envs


def run(args, mode):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu') if args.no_cuda else torch.device('cuda', args.device)
    if not args.no_cuda:
        torch.cuda.set_device(args.device)
    args.device = device
    datasets, box_sets = [], []
    for directory, length in zip(args.datasets, args.sequence_lengths):
        distributions, instances, box_set = load_multiple_datasets(directory, [length])
        if instances[length].shape[1] != args.num_processes:
            raise ValueError(f'{directory}: --num-processes must equal {instances[length].shape[1]}')
        if distributions[length].shape[0] < args.evaluation_steps:
            raise ValueError(f'{directory}: not enough distributions for --evaluation-steps')
        datasets.append({'instances': instances[length], 'distributions': distributions[length]})
        box_sets.append(box_set)
    timestamp = time.strftime('%Y.%m.%d-%H-%M-%S')
    log_dir = Path('logs/runs') / f'{mode}-{timestamp}'
    log_dir.mkdir(parents=True, exist_ok=True)
    registration_envs()
    envs = None
    with SummaryWriter(logdir=str(log_dir)) as writer:
        try:
            envs = make_vec_envs(args, './logs/runinfo', True)
            sampler = InstanceGenerator()
            policy = DRL_GAT(args).to(device)
            if mode == 'asap':
                from asap_trainer import ASAPTrainer
                selection = DRL_GAT(args, search_model=True).to(device)
                trainer = ASAPTrainer(writer, timestamp, selection, policy, sampler, args)
                for model, checkpoint in [
                    (policy, args.proposal_checkpoint or args.initial_checkpoint),
                    (selection, args.selection_checkpoint or args.initial_checkpoint),
                ]:
                    if not checkpoint:
                        raise ValueError('ASAP requires --initial-checkpoint or both policy checkpoints')
                    model.load_state_dict(torch.load(checkpoint, map_location=device))
                epochs = args.epochs or (30 if args.continuous else 50)
                steps = args.training_steps or (10 if args.continuous else 20)
            elif mode == 'meta':
                from meta_trainer import MetaTrainer
                trainer = MetaTrainer(writer, timestamp, policy, DRL_GAT(args).to(device), sampler, args)
                epochs = args.epochs or (70 if args.continuous else 150)
                steps = args.training_steps or (8 if args.continuous else 16)
            else:
                from pct_trainer import PCTTrainer
                trainer = PCTTrainer(writer, timestamp, policy, sampler, args)
                epochs = args.epochs or (70 if args.continuous else 200)
                steps = args.training_steps or (8 if args.continuous else 16)
            if mode != 'asap' and args.initial_checkpoint:
                policy.load_state_dict(torch.load(args.initial_checkpoint, map_location=device))
            args.epochs = epochs
            common = (envs, args, steps, epochs, device, str(log_dir), datasets,
                      args.evaluation_steps, args.sequence_lengths, box_sets)
            if args.evaluate:
                for dataset, length, box_set in zip(datasets, args.sequence_lengths, box_sets):
                    ratio, count = trainer.evaluate(envs, args, args.evaluation_steps, device,
                                                    length, box_set=box_set, dataset=dataset)
                    print(f'utilization={np.mean(ratio):.6f}, packed_items={np.mean(count):.2f}')
            elif mode == 'asap':
                if args.adapt_policy in ('proposal', 'both'):
                    trainer.train_n_steps_proposal(*common, change_box_set=args.sample_box_subsets)
                if args.adapt_policy in ('selection', 'both'):
                    trainer.train_n_steps_selection(*common, change_box_set=args.sample_box_subsets)
            elif mode == 'meta':
                trainer.train_n_steps(*common, default_mod=args.sample_box_subsets)
            else:
                trainer.train_n_steps_with_eval(*common, default_mod=args.sample_box_subsets)
            if not args.evaluate:
                # Always emit final state_dicts, including short smoke runs.
                destination = Path(args.model_save_path) / f'{mode}-{timestamp}'
                destination.mkdir(parents=True, exist_ok=True)
                if mode == 'asap':
                    torch.save(policy.state_dict(), destination / 'proposal-final.pt')
                    torch.save(selection.state_dict(), destination / 'selection-final.pt')
                else:
                    torch.save(policy.state_dict(), destination / f'{mode}-final.pt')
        finally:
            if envs is not None:
                envs.close()
