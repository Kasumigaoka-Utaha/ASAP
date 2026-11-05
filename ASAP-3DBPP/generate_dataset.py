"""Generate box sets and item sequences for the 3D-BPP experiments."""
import argparse
from itertools import product
from pathlib import Path
from packing_config import PAPER_ITEM_SIZES


def get_generation_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset', choices=list(PAPER_ITEM_SIZES), default='Default',
                        help='Appendix E.1 dimension set; overridden by --item-sizes or --least-size')
    parser.add_argument('--item-sizes', nargs='+', type=int, help='Explicit custom item dimensions')
    parser.add_argument('--least-size', type=int, default=None)
    parser.add_argument('--cross-range', type=int, default=6)
    parser.add_argument('--sequence-lengths', nargs='+', type=int, default=[70, 80])
    parser.add_argument('--num-distributions', type=int, default=100)
    parser.add_argument('--num-processes', type=int, default=64,
                        help='Number of instances per distribution')
    parser.add_argument('--continuous', action='store_true',
                        help='Add independent U(-0.5, 0.5) perturbations to item dimensions')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()
    if args.item_sizes and args.least_size is not None:
        parser.error('use either --item-sizes or --least-size')
    values = [args.cross_range, args.num_distributions,
              args.num_processes] + args.sequence_lengths
    values += args.item_sizes or []
    if args.least_size is not None:
        values.append(args.least_size)
    if any(value <= 0 for value in values):
        parser.error('sizes, lengths and sample counts must be positive')
    return args


def main(args):
    import torch
    from instance_generator import InstanceGenerator

    torch.manual_seed(args.seed)
    device = torch.device('cpu') if args.no_cuda else torch.device('cuda', args.device)
    sizes = args.item_sizes or (list(range(args.least_size, args.least_size + args.cross_range))
                                if args.least_size is not None else PAPER_ITEM_SIZES[args.dataset])
    box_set = torch.tensor(sorted({tuple(sorted(box)) for box in product(sizes, repeat=3)}),
                           device=device)
    # Reject existing datasets before writing the box-set metadata.
    for length in args.sequence_lengths:
        directory = args.output_dir / f'num-box-{length}'
        if any(directory.glob('instance-*.pt')):
            raise FileExistsError(f'{directory} already contains generated instances; choose a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(box_set.cpu(), args.output_dir / 'box-set.pt')
    sampler = InstanceGenerator()
    for length in args.sequence_lengths:
        directory = args.output_dir / f'num-box-{length}'
        directory.mkdir(exist_ok=True)
        for index in range(args.num_distributions):
            distribution = torch.rand(args.num_processes, len(box_set), device=device)
            distribution /= distribution.sum(dim=1, keepdim=True)
            with torch.no_grad():
                instances, _ = sampler(box_set, length, args.num_processes, distribution,
                                       random_mode=True, continuous=args.continuous)
            torch.save({'dist': distribution.cpu(), 'ins': instances.cpu()},
                       directory / f'instance-{index}.pt')
        print(f'Saved {args.num_distributions} distributions to {directory}')


if __name__ == '__main__':
    main(get_generation_args())
