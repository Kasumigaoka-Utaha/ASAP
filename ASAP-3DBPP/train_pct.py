"""Phase 1: pretrain the PCT base policy without meta-learning."""
from cli import get_args


if __name__ == '__main__':
    args = get_args('pct')
    from experiment import run
    run(args, 'pct')
