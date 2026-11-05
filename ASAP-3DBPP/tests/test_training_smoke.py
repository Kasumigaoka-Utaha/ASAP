"""Short real CPU regression for both packing environments and frozen proposal."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ['torch', 'gym', 'numpy', 'tensorboardX', 'cloudpickle', 'pandas']
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in DEPENDENCIES)


@unittest.skipUnless(AVAILABLE, 'requires the BPP training dependencies')
class TrainingSmokeTests(unittest.TestCase):
    def test_selection_updates_with_frozen_proposal(self):
        import torch
        for continuous in (False, True):
            with self.subTest(continuous=continuous), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                environment = dict(os.environ, PYTHONWARNINGS='ignore::FutureWarning')

                def run(script, *arguments):
                    result = subprocess.run([sys.executable, str(ROOT / script), *arguments],
                                            cwd=directory, env=environment, capture_output=True,
                                            text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    return result.stdout

                continuous_flag = ['--continuous'] if continuous else []
                run('generate_dataset.py', '--dataset', 'Default', '--output-dir', 'data',
                    '--sequence-lengths', '4', '--num-distributions', '1', '--num-processes', '1',
                    '--no-cuda', *continuous_flag)
                common = ['--datasets', 'data', '--sequence-lengths', '4', '--num-processes', '1',
                          '--epochs', '1', '--training-steps', '1', '--evaluation-steps', '1',
                          '--no-cuda', '--embedding-size', '16', '--hidden-size', '32',
                          '--leaf-node-holder', '8', '--model-save-path', 'models', *continuous_flag]
                run('train_pct.py', *common)
                initial_path = next(directory.glob('models/pct-*/pct-final.pt'))
                run('adapt_asap.py', *common, '--proposal-size', '3',
                    '--initial-checkpoint', str(initial_path), '--adapt-policy', 'selection')
                proposal_path = next(directory.glob('models/asap-*/proposal-final.pt'))
                selection_path = proposal_path.with_name('selection-final.pt')
                # All artifacts are state_dict tensors created by this test. Use the
                # default loader for compatibility with the suggested PyTorch 1.13.
                initial = torch.load(initial_path, map_location='cpu')
                proposal = torch.load(proposal_path, map_location='cpu')
                selection = torch.load(selection_path, map_location='cpu')
                self.assertEqual(initial.keys(), proposal.keys())
                self.assertTrue(all(torch.equal(initial[key], proposal[key]) for key in initial))
                self.assertTrue(any(not torch.equal(initial[key], selection[key]) for key in initial))
                self.assertTrue(all(torch.isfinite(value).all() for value in selection.values()))
                output = run('adapt_asap.py', *common, '--proposal-size', '3',
                             '--proposal-checkpoint', str(proposal_path),
                             '--selection-checkpoint', str(selection_path), '--evaluate')
                self.assertIn('utilization=', output)
                self.assertIn('packed_items=4.00', output)


if __name__ == '__main__':
    unittest.main()
