"""Gradient and tiny real-policy checks for distribution-task routing MAML.

Run: python -m unittest discover -s tests -p test_maml.py -v
Requires the routing requirements; does not need data or trained checkpoints.
"""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
ROUTING = ROOT / "ASAP-Routing"
if not ROUTING.is_dir():
    ROUTING = ROOT / "ASAP-RP"
sys.path.insert(0, str(ROUTING))

from maml import (DistributionTask, adapt_parameters, call_with_parameters,
                  maml_objective, maml_task_loss)
from meta_train import (RoutingRollout, build_parser, build_policy, generate_batch,
                        load_initialization, reinforce_loss, sample_tasks,
                        save_initialization, validate_args)


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.2, dtype=torch.float64))

    def forward(self, x):
        return self.weight * x


def squared_loss(module, parameters, buffers, batch):
    x, target = batch
    predicted = call_with_parameters(module, parameters, buffers, x)
    return 0.5 * (predicted - target).square().mean()


class MAMLGradientTests(unittest.TestCase):
    def setUp(self):
        self.model = ScalarModel()
        self.task = DistributionTask("a", [(torch.tensor(2.), torch.tensor(1.))],
                                     (torch.tensor(3.), torch.tensor(2.)))

    def test_second_order_matches_analytic_gradient(self):
        loss = maml_task_loss(self.model, self.task, squared_loss, inner_lr=0.1)
        gradient, = torch.autograd.grad(loss, self.model.weight)
        adapted = 1.2 - 0.1 * (1.2 * 2.0 - 1.0) * 2.0
        expected = (adapted * 3.0 - 2.0) * 3.0 * (1.0 - 0.1 * 4.0)
        self.assertAlmostEqual(float(gradient), expected, places=10)
        self.assertEqual(float(self.model.weight), 1.2)

    def test_first_order_is_distinct_and_keeps_identity_path(self):
        full = maml_task_loss(self.model, self.task, squared_loss, inner_lr=0.1)
        full_grad, = torch.autograd.grad(full, self.model.weight)
        first = maml_task_loss(self.model, self.task, squared_loss, inner_lr=0.1,
                               first_order=True)
        first_grad, = torch.autograd.grad(first, self.model.weight)
        self.assertAlmostEqual(float(full_grad / first_grad), 0.6, places=10)
        self.assertNotEqual(float(first_grad), 0.)

    def test_task_adaptation_never_mutates_common_initialization(self):
        base = self.model.weight.detach().clone()
        other = DistributionTask("b", [(torch.tensor(1.), torch.tensor(-1.))],
                                 (torch.tensor(2.), torch.tensor(0.)))
        average = maml_objective(self.model, [self.task, other], squared_loss, 0.1)
        separate = (maml_task_loss(self.model, self.task, squared_loss, 0.1)
                    + maml_task_loss(self.model, other, squared_loss, 0.1)) / 2
        self.assertTrue(torch.equal(self.model.weight, base))
        self.assertTrue(torch.allclose(average, separate))
        average.backward()
        self.assertTrue(torch.isfinite(self.model.weight.grad))

    def test_buffers_are_task_local(self):
        class Stateful(ScalarModel):
            def __init__(self):
                super().__init__()
                self.register_buffer("counter", torch.zeros(()))

            def forward(self, x):
                self.counter.add_(1)
                return super().forward(x)

        model = Stateful()
        _, buffers = adapt_parameters(model, self.task.support, squared_loss, 0.1)
        self.assertEqual(float(model.counter), 0.)
        self.assertEqual(float(buffers["counter"]), 1.)


class RoutingMAMLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def tiny_args(backbone="pomo", problem="tsp", phase="proposal"):
        args = build_parser().parse_args([
            "--backbone", backbone, "--problem", problem, "--phase", phase,
            "--nb-nodes", "5", "--support-size", "2", "--query-size", "2",
            "--tasks-per-step", "2", "--inner-steps", "1", "--device", "cpu",
            "--dim-emb", "8", "--dim-ff", "16", "--nb-heads", "2",
            "--nb-layers-action-encoder", "1", "--nb-layers-state-encoder", "1",
            "--nb-layers-decoder", "1", "--embedding-dim", "8",
            "--encoder-layer-num", "1", "--head-num", "2", "--qkv-dim", "4",
            "--ff-hidden-dim", "16", "--k-promising", "3", "--knn-k", "4",
        ])
        return validate_args(args)

    def test_distributions_are_separate_tasks_and_query_is_fresh(self):
        args = self.tiny_args()
        args.tasks_per_step = 4
        tasks = sample_tasks(args, torch.device("cpu"))
        self.assertEqual({task.distribution for task in tasks},
                         {"uniform", "clustered", "explosion", "implosion"})
        for task in tasks:
            self.assertNotEqual(task.support[0].data_ptr(), task.query.data_ptr())
            self.assertFalse(torch.equal(task.support[0], task.query))

    def test_real_policy_second_order_for_both_backbones_and_problems(self):
        for backbone in ("invit", "pomo"):
            for problem in ("tsp", "cvrp"):
                for phase in ("proposal", "selection"):
                    with self.subTest(backbone=backbone, problem=problem, phase=phase):
                        torch.manual_seed(0)
                        args = self.tiny_args(backbone, problem, phase)
                        policy = build_policy(args)
                        model = RoutingRollout(policy, args)
                        before = {key: value.detach().clone() for key, value in policy.state_dict().items()}
                        support = generate_batch(args, "clustered", 4, torch.device("cpu"))
                        query = generate_batch(args, "explosion", 4, torch.device("cpu"))
                        task = DistributionTask("clustered", [support], query)
                        loss = maml_task_loss(model, task, reinforce_loss, inner_lr=1e-3)
                        loss.backward()
                        self.assertTrue(torch.isfinite(loss))
                        gradients = [p.grad for p in policy.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
                        self.assertTrue(all(torch.equal(before[key], value)
                                            for key, value in policy.state_dict().items()))

    def test_selection_can_meta_train_with_frozen_proposal(self):
        for backbone in ("invit", "pomo"):
            for problem in ("tsp", "cvrp"):
                with self.subTest(backbone=backbone, problem=problem):
                    torch.manual_seed(0)
                    args = self.tiny_args(backbone, problem, "selection")
                    proposal = build_policy(args, "proposal")
                    model = RoutingRollout(build_policy(args), args, proposal=proposal)
                    support = generate_batch(args, "uniform", 4, torch.device("cpu"))
                    query = generate_batch(args, "uniform", 4, torch.device("cpu"))
                    task = DistributionTask("uniform", [support], query)
                    loss = maml_task_loss(model, task, reinforce_loss, 1e-3)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(p.grad is None for p in proposal.parameters()))
                    self.assertTrue(any(p.grad is not None for p in model.policy.parameters()))

    def test_checkpoint_loads_into_existing_policy_schema(self):
        args = self.tiny_args("invit", "tsp", "selection")
        policy = build_policy(args)
        optimizer = torch.optim.Adam(policy.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "init.ckpt"
            save_initialization(policy, optimizer, args, 3, path)
            loaded = build_policy(args)
            load_initialization(loaded, path, torch.device("cpu"))
            self.assertTrue(all(torch.equal(v, loaded.state_dict()[key])
                                for key, v in policy.state_dict().items()))
            from utils.utils_for_model import load_stage_ckpt
            existing_loader_policy = build_policy(args)
            load_stage_ckpt(existing_loader_policy, str(path), torch.device("cpu"),
                            expected_stage="pretrain")
            self.assertTrue(all(torch.equal(v, existing_loader_policy.state_dict()[key])
                                for key, v in policy.state_dict().items()))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["stage"], "pretrain")
            self.assertEqual(checkpoint["meta_phase"], "selection")


if __name__ == "__main__":
    unittest.main()
