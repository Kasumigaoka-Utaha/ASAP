"""Phase-2 cooperative tuning and policy-specific online adaptation."""
from cli import get_args


if __name__ == '__main__':
    args = get_args('asap')
    from experiment import run
    run(args, 'asap')
