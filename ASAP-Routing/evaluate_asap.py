"""Evaluate ASAP proposal + selection policies for INViT/POMO on TSP/CVRP."""

import importlib
import torch
from utils.utils_for_model import (
    create_parser, load_stage_ckpt, compute_tsp_tour_length, compute_vrp_tour_length,
)


def build_args():
    config = {
        'problem': 'tsp',
        'backbone': 'invit',
        'bsz': 64,
        'nb_nodes': 50,
        'dim_input_nodes': 2,
        'dim_emb': 128,
        'dim_ff': 512,
        'nb_heads': 8,
        'nb_layers_action_encoder': 2,
        'nb_layers_state_encoder': 2,
        'nb_layers_decoder': 3,
        'num_state_encoder': 1,
        'batchnorm': False,
        'knn_k': 25,
        'state_k': '35,50,65',
        'if_agg_whole_graph': False,
        'use_stage1_action_encoding': True,
        'embedding_dim': 128,
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'ff_hidden_dim': 512,
        'logit_clipping': 10.0,
        'eval_type': 'sampling',
        'stage1_ckpt': '',
        'stage2_ckpt': '',
        'k_promising': 8,
        'capacity': 50.0,
        'aug': 'mix',
        'test_aug_num': 16,
        'data_path': './data/',
        'eval_mode': 'dataset',
        'eval_sizes': '100',
        'eval_distributions': 'uniform',
        'eval_num_instances': -1,
        'nb_batch_eval': 100,
        'use_best_over_aug': True,
        'measure_eval_time': False,
    }
    parser, args = create_parser(config)
    args = parser.parse_args(namespace=args)
    args.problem = args.problem.lower()
    args.backbone = args.backbone.lower()
    args.eval_mode = args.eval_mode.lower()
    if args.problem not in ('tsp', 'cvrp') or args.backbone not in ('invit', 'pomo'):
        parser.error('Use --problem tsp|cvrp and --backbone invit|pomo')
    if args.eval_mode not in ('dataset', 'synthetic', 'tsplib', 'cvrplib'):
        parser.error('eval_mode must be dataset, synthetic, tsplib, or cvrplib')
    if args.eval_mode == 'tsplib' and args.problem != 'tsp':
        parser.error('TSPLIB requires --problem tsp')
    if args.eval_mode == 'cvrplib' and args.problem != 'cvrp':
        parser.error('CVRPLIB requires --problem cvrp')
    if not args.stage1_ckpt or not args.stage2_ckpt:
        parser.error('Both --proposal_ckpt and --selection_ckpt are required')
    if args.test_aug_num < 1 or args.bsz < 1 or args.bsz % args.test_aug_num:
        parser.error('bsz must be a positive multiple of test_aug_num')
    args.eval_sizes = [int(s) for s in args.eval_sizes.split(',') if s.strip()]
    args.eval_distributions = [s.strip().lower() for s in args.eval_distributions.split(',') if s.strip()]
    args.state_k = [int(s) for s in args.state_k.split(',') if s.strip()][:args.num_state_encoder]
    return args


def _make_policies(args):
    suffix = ('pomo_' if args.backbone == 'pomo' else '') + args.problem
    evaluator = importlib.import_module('adapt_selection_' + suffix)
    if args.backbone == 'pomo':
        prefix = 'POMOTSP' if args.problem == 'tsp' else 'POMOVRP'
        kwargs = evaluator._model_kwargs(args)
        proposal = getattr(evaluator, prefix + 'Stage1Policy')(**kwargs)
        selection = getattr(evaluator, prefix + 'Stage2Policy')(**kwargs)
    else:
        prefix = 'TSP' if args.problem == 'tsp' else 'VRP'
        proposal = getattr(evaluator, prefix + 'Stage1Policy')(args)
        selection = getattr(evaluator, prefix + 'Stage2Policy')(args)
    return evaluator, proposal, selection


@torch.no_grad()
def evaluate_asap(args, device):
    evaluator, proposal, selection = _make_policies(args)
    proposal.to(device)
    selection.to(device)
    load_stage_ckpt(proposal, args.stage1_ckpt, device, expected_stage='stage1')
    load_stage_ckpt(selection, args.stage2_ckpt, device, expected_stage='stage2')
    proposal.eval()
    selection.eval()
    if args.eval_mode != 'synthetic':
        return evaluator._evaluate_model(args, device, proposal, selection)

    results = {}
    for distribution in args.eval_distributions:
        lengths = []
        for _ in range(args.nb_batch_eval):
            batch = evaluator._build_batch(args, device, distribution)
            if args.problem == 'tsp':
                coords = batch
                tours = evaluator._rollout_lengths(args, coords, proposal, selection, deterministic=True)
                length = compute_tsp_tour_length(coords, tours)
            else:
                inputs, coords = batch
                tours = evaluator._rollout_lengths(args, inputs, proposal, selection, args.capacity, deterministic=True)
                length = compute_vrp_tour_length(coords, tours)
            lengths.append(length.mean().item())
        results[distribution] = sum(lengths) / len(lengths)
        print(f'[ASAP][Synthetic][{distribution}] mean tour length={results[distribution]:.6f}')
    return results


def main():
    args = build_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_asap(args, device)


if __name__ == '__main__':
    main()
