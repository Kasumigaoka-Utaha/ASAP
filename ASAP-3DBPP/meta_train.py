"""Run the released task-wise PCT meta-initialization experiment."""
from cli import get_args


if __name__ == '__main__':
    args = get_args('meta')
    from experiment import run
    run(args, 'meta')
