# Paper-to-code correspondence

Reference: [ASAP, ICML 2026](https://proceedings.mlr.press/v306/fang26t.html),
Sections 4-5 and Appendices C-E. The accompanying
[arXiv HTML](https://arxiv.org/html/2501.17377v4) is also available for navigation.
This document separates the method described in the paper from the implementation
actually present in this source release.

## Terminology

| Paper concept | 3D-BPP | Routing |
| --- | --- | --- |
| Adaptive Selection After Proposal (ASAP) | `adapt_asap.py`, `asap_trainer.py` | `train_asap*.py`, `*_policy_asap.py` |
| Proposal policy, decision Stage 1 | `ASAPTrainer.proposal_head` | `*Stage1Policy.select_k` |
| Selection policy, decision Stage 2 | `ASAPTrainer.selection_policy` | `*Stage2Policy.select_action` |
| Candidate action set of size k | `--proposal-size` (alias `--search-num`) | `--k_promising` in training; `--k-promising` in MAML |
| Phase 1: base initialization | `train_pct.py` | `pretrain_policy.py`, `pretrain_pomo_*_policy.py`, `meta_train.py` |
| Phase 2: cooperative tuning | `adapt_asap.py --adapt-policy both` | `train_asap*.py --stage alternating` |
| Online selection adaptation with frozen proposal | `adapt_asap.py --adapt-policy selection` | `adapt_selection*.py` |
| Proposal adaptation ablation | `adapt_asap.py --adapt-policy proposal` | `adapt_proposal*.py` |
| MAML, distributions as tasks | Legacy surrogate in `meta_trainer.py`; see below | Newly added `maml.py`, `meta_train.py` |

Training phases describe when the models are trained. Decision stages describe
which policy acts. Routing accepts `proposal`, `selection`, and `alternating` as
CLI stage names; historical checkpoint tags `stage1` and `stage2` remain intact.
Model class names and parameter keys are retained for checkpoint compatibility.

## 3D-BPP

The implementation uses PCT graph representations and discrete/continuous packing
environments. `packing_config.py` names the Appendix E.1 item sets Default,
ID-Large, ID-Medium, ID-Small, OOD, OOD-Large, and OOD-Small. The generator removes
rotation-equivalent shapes and samples a distribution for each batch; continuous
instances add dimension noise in [-0.5, 0.5]. The objective is space utilization.

The supplied actor/critic uses ACKTR/KFAC. The cooperative path alternates a
proposal training block with a selection training block; it is not a single joint
differentiable optimizer step through discrete top-k. The default online path
updates only the selection policy. The CLI makes data paths, rollout horizons,
checkpoint paths, and the policy to adapt explicit.

The retained `meta_trainer.py` does **not** implement Algorithm 1's MAML objective:
it adapts an inner policy but recomputes the outer loss on the initial policy,
without differentiating through adaptation or transferring the adapted policy's
gradients. It should not be labeled full MAML or first-order MAML. It remains
available to preserve the supplied experimental routine. The new routing MAML
implementation does not repair this separate packing routine.

The paper's full preliminary MCTS solver/figure scripts and the AR2L/GOPT
implementations are absent. Retained heuristic and I/O helpers do not supply those
experiments. Model checkpoints and complete evaluation datasets are also absent.

## Routing

Routing covers TSP and CVRP with INViT and POMO policy families. The INViT baseline
trainer and evaluator, dataset generators, Gurobi/LKH/HGS solver wrappers, and
proposal-adaptation ablations are retained because they correspond to paper
experiments. The old `ASAP-RP` directory is now `ASAP-Routing`.

`configs/pretrain_{tsp,cvrp}.yaml` and `configs/asap_{tsp,cvrp}.yaml` expose the
paper's INViT dimensions and main training parameters. The YAML runner supports
explicit CLI overrides. These settings do not certify reproduction of the
published tables. The released POMO model defaults to six encoder layers, whereas
Appendix E describes two; its decoder also follows the supplied POMO architecture.
Changing these automatically would break existing checkpoints. Match architecture
arguments explicitly when loading models.

TSPLIB/CVRPLIB evaluators compute reported route lengths in the original
coordinate scale, after using normalized coordinates as model input. Gap
calculation requires matching reference solutions. The unified evaluator executes
the proposal and selection policies for both TSP and CVRP and both backbones.
An exact PyTorch KNN fallback allows running without the optional `torch-cluster`
extension; ties may be ordered differently.

## Added distribution-task MAML

`ASAP-Routing/meta_train.py` implements distribution-task initialization for both
backbones and both problems. Each meta-iteration samples distinct distribution
tasks. Each task starts from the shared initialization, updates functional weights
with support instances, and computes a query loss on independent instances from
the same distribution. The outer optimizer averages task query gradients.

The default differentiates through inner SGD updates. `--first-order` omits
support Hessian terms. The loss uses a REINFORCE surrogate with a detached greedy
baseline; sampled discrete tours are treated as fixed when differentiating the
surrogate. Task-local model buffers and rollout caches prevent cross-task state
leakage. `--phase` names the **policy role** to initialize, not the paper's two
training phases. A selection initialization can use a frozen proposal checkpoint;
without one it uses geometric candidates, as in the supplied pretraining paths.

By default, all four named distribution families participate in meta-training.
Those families are consequently seen training tasks, not held-out OOD tests for
that run. Use `--task-distributions` and `--tasks-per-step` to define a training
subset, retain the checkpoint's recorded task list, and evaluate on separate
instances or held-out task families as appropriate.

This is a new implementation added during repository preparation. It has no
associated published-result weights and has not been trained to reproduce the
paper's reported metrics. See the [Routing README](../ASAP-Routing/README.md) for
commands, tests, and compatible checkpoint arguments.

## Release and validation scope

Source consistency checks, small CPU smoke runs, and MAML gradient tests establish
software behavior at small scale. They do not establish convergence, GPU
performance, or equality with the paper's tables. Run the complete training and
evaluation protocol with the intended datasets and random seeds for those claims.

The final preparation run passed 20 tests: 14 routing gradient/integration tests
and 6 packing CLI/training tests. Validation used Python 3.10.20, PyTorch
2.5.1 CPU, NumPy 1.23.5, and Gym 0.13.0. Additional small runs verified MAML
initialization through cooperative tuning and strict evaluation loading. Source
validation checked 93 component Python files and 6 documentation files.

Original ASAP contributions use MIT, while the PCT-derived component retains its
recorded academic-use-only restriction. See
[third-party notices](../THIRD_PARTY_NOTICES.md).
