"""Command-line options shared by the 3D-BPP experiment entry points."""

import argparse
from packing_config import container_size, item_size_set


def get_args(mode):
    parser = argparse.ArgumentParser(description=f"ASAP 3D-BPP: {mode}")
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Dataset directories containing box-set.pt and num-box-N/')
    parser.add_argument('--sequence-lengths', nargs='+', type=int, required=True,
                        help='One sequence length per dataset')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--num-processes', type=int, default=64,
                        help='Must match the number of instances per distribution in each dataset')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--training-steps', type=int, default=None)
    parser.add_argument('--evaluation-steps', type=int, default=20)
    parser.add_argument('--initial-checkpoint', help='Phase-1 PCT state_dict checkpoint')
    parser.add_argument('--proposal-checkpoint', help='ASAP proposal state_dict checkpoint')
    parser.add_argument('--selection-checkpoint', help='ASAP selection state_dict checkpoint')
    parser.add_argument('--adapt-policy', choices=['selection', 'proposal', 'both'], default='selection',
                        help='selection freezes proposal; proposal is an ablation; both performs cooperative tuning')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the loaded policy without updates')
    parser.add_argument('--sample-box-subsets', action='store_true',
                        help='Use the original randomized box-subset training schedule')
    parser.add_argument('--model-save-path', default='./logs/experiment')
    parser.add_argument('--max-model-num', type=int, default=20)
    parser.add_argument('--model-save-interval', type=int, default=200)
    parser.add_argument('--model-update-interval', type=int, default=20000)
    parser.add_argument('--setting', type=int, choices=[1, 2], default=2,
                        help='PCT rotation setting: 1 permits two orientations, 2 permits six')
    parser.add_argument('--lnes', choices=['EMS', 'EV', 'EP', 'CP', 'FC'], default='EMS')
    parser.add_argument('--internal-node-holder', type=int, default=80)
    parser.add_argument('--leaf-node-holder', type=int, default=50)
    parser.add_argument('--num-seen-box', type=int, default=1)
    parser.add_argument('--proposal-size', '--search-num', dest='search_num', type=int, default=5,
                        help='Top-k proposal set size (k in the paper)')
    parser.add_argument('--embedding-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--gat-layer-num', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--actor-loss-coef', type=float, default=1.0)
    parser.add_argument('--critic-loss-coef', type=float, default=1.0)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--no-shuffle', action='store_true', help='Disable candidate placement shuffling')
    args = parser.parse_args()
    if mode == 'asap' and not (args.initial_checkpoint or
                               (args.proposal_checkpoint and args.selection_checkpoint)):
        parser.error('ASAP requires --initial-checkpoint or both policy checkpoints')
    if mode != 'asap' and args.evaluate and not args.initial_checkpoint:
        parser.error('--evaluate requires --initial-checkpoint')
    if len(args.datasets) != len(args.sequence_lengths):
        parser.error('--datasets and --sequence-lengths must have the same number of entries')
    if len(set(args.sequence_lengths)) != 1:
        parser.error('all datasets in a run must use the same sequence length; run different lengths separately')
    if any(length <= 0 for length in args.sequence_lengths):
        parser.error('sequence lengths must be positive')
    if args.num_processes <= 0 or args.evaluation_steps <= 0:
        parser.error('process and evaluation counts must be positive')
    if args.max_model_num <= 0 or args.model_save_interval <= 0:
        parser.error('checkpoint counts and intervals must be positive')
    if args.epochs is not None and args.epochs <= 0:
        parser.error('--epochs must be positive')
    if args.training_steps is not None and args.training_steps <= 0:
        parser.error('--training-steps must be positive')
    if not 1 <= args.search_num <= args.leaf_node_holder:
        parser.error('proposal size must lie between 1 and --leaf-node-holder')
    if args.num_seen_box != 1:
        parser.error('the released online policy uses --num-seen-box 1')
    args.num_steps = max(args.sequence_lengths)
    args.internal_node_holder = max(args.internal_node_holder, args.num_steps)
    args.next_holder = args.num_steps
    args.internal_node_length = 6
    args.container_size = container_size
    args.item_size_set = item_size_set
    args.normFactor = 1.0 / max(container_size)
    args.id = 'PctContinuous-v1' if args.continuous else 'PctDiscrete-v1'
    args.device = 'cpu' if args.no_cuda else args.device
    args.shuffle = not args.no_shuffle
    args.allow_dist_input = False
    args.use_acktr = True
    args.dataset_path = ''
    args.load_dataset = False
    args.keep_prev = False
    args.output_mask = False
    args.sample_from_distribution = False
    args.sample_left_bound = None
    args.sample_right_bound = None
    return args
