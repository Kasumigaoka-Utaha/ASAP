"""Run a routing entry point with a flat YAML configuration and CLI overrides."""

import argparse
from pathlib import Path
import subprocess
import sys


ENTRYPOINTS = (
    'pretrain_policy', 'pretrain_pomo_tsp_policy', 'pretrain_pomo_cvrp_policy',
    'train_asap', 'train_asap_pomo_tsp', 'train_asap_pomo_cvrp',
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('entrypoint', choices=ENTRYPOINTS)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Print the command without training')
    args, overrides = parser.parse_known_args(argv)
    try:
        import yaml
    except ImportError:
        parser.error('PyYAML is required: python -m pip install PyYAML')
    with args.config.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        parser.error('Configuration must be a mapping of CLI option names to scalar values')
    root = Path(__file__).resolve().parent
    command = [sys.executable, '-u', str(root / (args.entrypoint + '.py'))]
    for key, value in config.items():
        if not isinstance(key, str) or not key or key.startswith('-'):
            parser.error('Configuration keys must be CLI option names without leading dashes')
        if value is None or value == '':
            continue
        if isinstance(value, (dict, list)):
            parser.error(f'{key} must have a scalar value')
        command.extend(['--' + key, str(value)])
    # argparse uses the last occurrence of an option, so explicit overrides win.
    command.extend(overrides)
    if args.dry_run:
        print(subprocess.list2cmdline(command))
        return 0
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == '__main__':
    sys.exit(main())
