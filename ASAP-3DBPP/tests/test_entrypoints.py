"""Public CLI contracts; these checks do not require PyTorch or Gym."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def invoke(self, script, *arguments):
        return subprocess.run([sys.executable, str(ROOT / script), *arguments],
                              capture_output=True, text=True, cwd=ROOT)

    def test_help_without_training_dependencies(self):
        for script in ['train_pct.py', 'meta_train.py', 'adapt_asap.py', 'generate_dataset.py']:
            with self.subTest(script=script):
                result = self.invoke(script, '--help')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_invalid_dataset_pairing_is_reported_before_loading(self):
        result = self.invoke('train_pct.py', '--datasets', 'first', 'second',
                             '--sequence-lengths', '70')
        self.assertEqual(result.returncode, 2)
        self.assertIn('same number of entries', result.stderr)

    def test_asap_requires_pretrained_weights(self):
        result = self.invoke('adapt_asap.py', '--datasets', 'example',
                             '--sequence-lengths', '70')
        self.assertEqual(result.returncode, 2)
        self.assertIn('both policy checkpoints', result.stderr)

    def test_evaluation_requires_loaded_policy(self):
        result = self.invoke('train_pct.py', '--datasets', 'example',
                             '--sequence-lengths', '70', '--evaluate')
        self.assertEqual(result.returncode, 2)
        self.assertIn('--initial-checkpoint', result.stderr)

    def test_paper_proposal_size_and_frozen_proposal_default(self):
        code = ('import json; from cli import get_args; a = get_args("asap"); '
                'print(json.dumps([a.search_num, a.adapt_policy, str(a.device)]))')
        result = subprocess.run([sys.executable, '-c', code, '--datasets', 'example',
                                 '--sequence-lengths', '70', '--initial-checkpoint', 'model.pt',
                                 '--proposal-size', '3', '--no-cuda'],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [3, 'selection', 'cpu'])


if __name__ == '__main__':
    unittest.main()
