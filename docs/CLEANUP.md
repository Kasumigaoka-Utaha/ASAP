# Repository preparation changes

## Naming

| Previous path/name | Public path/name |
| --- | --- |
| `ASAP-RP/` | `ASAP-Routing/` (TSP and CVRP) |
| 3D-BPP `main_pct.py` | `train_pct.py` |
| 3D-BPP `main_meta.py` | `meta_train.py` |
| 3D-BPP `main_asap.py` | `adapt_asap.py` |
| 3D-BPP `train_tools_acktr.py`, `train_tools_meta.py`, `train_tools_asap.py` | `pct_trainer.py`, `meta_trainer.py`, `asap_trainer.py` |
| 3D-BPP `givenData.py`, `insCreator.py` | `packing_config.py`, `instance_generator.py` |
| Routing `train_two_stage.py`, `train_pomo_*_two_stage.py` | `train_asap.py`, `train_asap_pomo_*.py` |
| Routing `evaluate_two_stage.py` | `evaluate_asap.py` |
| Routing `*_policy_two_stage.py` | `*_policy_asap.py` |
| Routing `finetune_*stage1*`, `finetune_*stage2*` | `adapt_proposal*`, `adapt_selection*` |
| Routing `config*.yaml` | `configs/pretrain_{tsp,cvrp}.yaml`, `configs/asap_{tsp,cvrp}.yaml` |
| Routing `train.py`, `training_loop.py`, `test_function.py` | `train_invit_baseline.py`, `training_invit_baseline.py`, `evaluate_invit_baseline.py` |

Historical model class names, checkpoint keys, stage tags, and PCT environment IDs
remain where changing them would break serialization or upstream compatibility.
CVRP is used in public routing entrypoint names; internal `VRP*` class names remain.

## Removed clutter

- Cached bytecode and empty evaluation log directories.
- Unused Atari wrappers, normalization/benchmark helpers, and unrelated PPO
  scaffolding from the packing snapshot.
- Broken packing prototypes and stale backup helpers with no runnable caller.
- Routing `test.py`, an isolated log reader, and the orphaned `tsp_solver.py` /
  `tsp_policy.py` wrapper pair.

PCT, INViT, POMO, heuristic baselines, solver wrappers, distribution generators,
and proposal/selection ablations are retained. The pre-existing BPP Git history
and user changes have not been reset or rewritten. Source archives exclude that
nested Git metadata.

## Runtime and documentation repairs

- Replaced empty hardcoded packing dataset lists with validated CLI inputs;
  fixed missing imports, duplicate options, stale model calls, sampler
  construction, proposal-action indexing, and vector-environment reset/spawn
  issues encountered while preparing runnable entrypoints.
- Added paper-named packing dataset presets and explicit proposal/selection
  adaptation controls.
- Corrected routing CLI booleans, configurable output directories, config runner,
  local paths, and original-scale evaluation of benchmark routes. The unified
  ASAP evaluator now runs both policy stages for both problem types.
- Added optional exact PyTorch KNN in place of a mandatory compiled extension.
- Added distribution-task MAML for routing, with genuine differentiable inner
  updates and an optional first-order mode.
- Rewrote component READMEs and added the root README, citation metadata,
  implementation correspondence, scoped MIT license, and upstream notices.

Known implementation gaps and differences from paper settings are recorded in
[PAPER_MAPPING.md](PAPER_MAPPING.md); this cleanup does not claim to reproduce
published performance.
