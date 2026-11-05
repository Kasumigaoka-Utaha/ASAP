"""Small routing integration checks requiring no datasets or trained models."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
ROUTING = next(p for p in ROOT.glob('ASAP-*') if (p / 'train_asap.py').is_file())
sys.path.insert(0, str(ROUTING))


class RoutingEntrypointTests(unittest.TestCase):
    def command(self, script, *args):
        result = subprocess.run(
            [sys.executable, str(ROUTING / script), *args], cwd=ROUTING,
            capture_output=True, text=True, timeout=60,
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_entrypoints_print_real_help(self):
        scripts = ['pretrain_policy.py', 'pretrain_pomo_tsp_policy.py',
                   'pretrain_pomo_cvrp_policy.py', 'train_asap.py',
                   'train_asap_pomo_tsp.py', 'train_asap_pomo_cvrp.py',
                   'evaluate_asap.py']
        scripts += [p.name for p in ROUTING.glob('adapt_*.py')]
        for script in scripts:
            with self.subTest(script=script):
                self.assertIn('usage:', self.command(script, '--help').lower())

    def test_all_backbones_train_save_and_evaluate(self):
        with tempfile.TemporaryDirectory(prefix='asap-routing-test-') as temporary:
            for backbone in ('invit', 'pomo'):
                for problem in ('tsp', 'cvrp'):
                    with self.subTest(backbone=backbone, problem=problem):
                        output = Path(temporary) / f'{backbone}_{problem}'
                        shared = ['--bsz', '4', '--nb_nodes', '5',
                                  '--test_aug_num', '1', '--nb_batch_eval', '1',
                                  '--k_promising', '2']
                        if backbone == 'invit':
                            architecture = ['--dim_emb', '16', '--dim_ff', '32',
                                            '--nb_heads', '2', '--nb_layers_action_encoder', '1',
                                            '--nb_layers_state_encoder', '1', '--nb_layers_decoder', '1',
                                            '--num_state_encoder', '1', '--knn_k', '4', '--state_k', '4']
                            trainer = 'train_asap.py'
                            training_problem = ['--problem', problem]
                        else:
                            architecture = ['--embedding_dim', '16', '--encoder_layer_num', '1',
                                            '--qkv_dim', '8', '--head_num', '2', '--ff_hidden_dim', '32']
                            trainer = f'train_asap_pomo_{problem}.py'
                            training_problem = []
                        self.command(trainer, *shared, *architecture, *training_problem,
                                     '--stage', 'alternating', '--aug_num', '1',
                                     '--nb_batch_per_epoch', '1', '--nb_epochs_alt', '1',
                                     '--save_dir', str(output), '--use_swanlab', 'false')
                        proposals = list(output.rglob('stage1.ckpt'))
                        selections = list(output.rglob('stage2.ckpt'))
                        self.assertEqual(len(proposals), 1)
                        self.assertEqual(len(selections), 1)
                        result = self.command('evaluate_asap.py', *shared, *architecture,
                                              '--problem', problem, '--backbone', backbone,
                                              '--proposal_ckpt', str(proposals[0]),
                                              '--selection_ckpt', str(selections[0]),
                                              '--eval_mode', 'synthetic')
                        self.assertIn('mean tour length=', result)

    def test_batched_knn_respects_graph_boundaries(self):
        from utils.knn import knn
        x = torch.tensor([[0., 0.], [2., 0.], [10., 0.], [11., 0.]])
        y = torch.tensor([[1.8, 0.], [10.9, 0.]])
        edges = knn(x, y, 2, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1]))
        self.assertEqual(edges[0].tolist(), [0, 0, 1, 1])
        self.assertEqual(set(edges[1, :2].tolist()), {0, 1})
        self.assertEqual(set(edges[1, 2:].tolist()), {2, 3})

    def test_exact_capacity_is_feasible(self):
        from cvrp_policy_asap import VRPStage1Policy
        from vrp_env import VRPEnvironment
        args = SimpleNamespace(dim_emb=16, dim_input_nodes=2, dim_ff=32,
                               nb_layers_action_encoder=1, nb_layers_state_encoder=1,
                               nb_layers_decoder=1, nb_heads=2, batchnorm=False)
        env = VRPEnvironment({'loc': torch.rand(2, 1, 2),
                              'depot': torch.rand(2, 2),
                              'demand': torch.tensor([[10], [10]])}, capacity=10)
        policy = VRPStage1Policy(args)
        self.assertFalse(policy._build_embeddings(env)['action_mask'].any())

    def test_custom_pomo_embedding_sets_correct_scaling(self):
        from pomo_tsp_policy import POMOTSPModel
        from pomo_vrp_model import POMOVRPModel
        for cls in (POMOTSPModel, POMOVRPModel):
            policy = cls(embedding_dim=16, encoder_layer_num=1, head_num=2,
                         qkv_dim=8, ff_hidden_dim=32)
            self.assertEqual(policy.model_params['sqrt_embedding_dim'], 4.0)

    def test_benchmark_gaps_use_original_coordinate_scale(self):
        import importlib
        args = SimpleNamespace(eval_mode='tsplib', measure_eval_time=False,
                               bsz=1, test_aug_num=1, use_best_over_aug=True,
                               data_path='unused', aug='mix')
        coords = torch.tensor([[0., 0.], [3., 0.], [0., 4.]])
        for name in ('adapt_selection_tsp', 'adapt_selection_pomo_tsp'):
            module = importlib.import_module(name)
            with self.subTest(module=name), \
                    patch.object(module, 'tsplib_collections', {'sample3': 12.0}), \
                    patch.object(module, 'parse_tsplib_name', return_value=('sample', 3)), \
                    patch.object(module, 'load_tsplib_file', return_value=(coords, None)), \
                    patch.object(module, 'choose_bsz', return_value=1), \
                    patch.object(module, '_rollout_lengths', return_value=torch.tensor([[0, 1, 2]])):
                logs = module._evaluate_model(args, torch.device('cpu'), torch.nn.Module(), torch.nn.Module())
                self.assertAlmostEqual(logs['tsplib-sample3']['best_len'], 12.0)
                self.assertAlmostEqual(logs['tsplib-sample3']['gap'], 0.0)
        module = importlib.import_module('adapt_selection_cvrp')
        args.eval_mode = 'cvrplib'
        data = (coords[0], coords[1:], torch.ones(2), torch.tensor(50), None)
        with patch.object(module, 'cvrplib_collections', {'sample2': 12.0}), \
                patch.object(module, 'parse_cvrplib_name', return_value=('sample', 2)), \
                patch.object(module, 'load_cvrplib_file', return_value=data), \
                patch.object(module, 'choose_bsz', return_value=1), \
                patch.object(module, '_rollout_lengths', return_value=torch.tensor([[0, 1]])):
            logs = module._evaluate_model(args, torch.device('cpu'), torch.nn.Module(), torch.nn.Module())
            self.assertAlmostEqual(logs['cvrplib-sample2']['best_len'], 12.0)
            self.assertAlmostEqual(logs['cvrplib-sample2']['gap'], 0.0)


if __name__ == '__main__':
    unittest.main()
