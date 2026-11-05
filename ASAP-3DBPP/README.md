# ASAP for Online 3D Bin Packing

The 3D-BPP component of [ASAP: Exploiting the Satisficing Generalization Edge in Neural Combinatorial Optimization](https://proceedings.mlr.press/v306/fang26t.html), by Han Fang, Paul Weng, and Yutong Ban (ICML 2026).

ASAP means **Adaptive Selection After Proposal**. A proposal policy selects a top-*k* set of feasible placements; a selection policy chooses the placement from that set. The PCT backbone represents previously packed boxes, candidate placements, and the current item as graph nodes. The objective is container space utilization, with packed-item count also reported. Both discrete and continuous item dimensions are supported in a 20 x 20 x 20 container.

## Paper-to-code mapping

| Paper element | Implementation |
| --- | --- |
| Phase 1: base-policy pretraining | `train_pct.py`, `pct_trainer.py` |
| Released task-wise meta-initialization routine | `meta_train.py`, `meta_trainer.py` (see limitation below) |
| Phase 2: cooperative tuning | `adapt_asap.py --adapt-policy both`, `asap_trainer.py` |
| Stage 1: proposal policy | `ASAPTrainer.proposal_head`; ranks full placement set |
| Stage 2: selection policy | `ASAPTrainer.selection_policy`; operates on top-*k* placements |
| Online selection adaptation | `adapt_asap.py --adapt-policy selection` (default); proposal weights remain frozen |
| Proposal adaptation ablation | `adapt_asap.py --adapt-policy proposal` |
| Proposal size *k* | `--proposal-size` (`--search-num` is an alias) |
| PCT graph actor/critic | `model.py`, `attention_model.py`, `graph_encoder.py` |
| Packing environments | `pct_envs/PctDiscrete0/`, `pct_envs/PctContinuous0/` |
| Appendix E.1 dataset dimension sets | `packing_config.py`, `generate_dataset.py` |
| Handcrafted packing baselines | `heuristic.py` |

**Reproduction limitation:** the supplied meta routine updates a separate inner policy, then recomputes the outer loss on the initial policy. It neither differentiates through inner updates nor transfers adapted-model gradients to the initial policy. It therefore does not implement the paper's MAML objective, including standard first-order MAML. Its name identifies the existing experiment, not a verified MAML reproduction. The MCTS solver and scripts for the paper's preliminary MCTS figures, AR2L, and GOPT are absent from this source snapshot. Small checkpoint/solution I/O helpers have been retained, but do not constitute an MCTS implementation. No datasets, pretrained weights, or claimed paper-result reproductions are bundled.

## Environment

Run commands from this directory. Use a separate environment from the Routing component. Python 3.9 on Linux is suggested for the legacy Gym and ACKTR code:

```bash
pip install -r requirements.txt
```

The requirement pins describe a suggested compatible environment, not an end-to-end validated training environment. Select a PyTorch wheel appropriate for your CUDA installation. The packing code uses Gym's older four-value `step` API. Modern Gym/Gymnasium is not a drop-in replacement. `--no-cuda` enables CPU execution. Help and argument validation work without installing the training dependencies.

Short CPU smoke tests passed on Windows with Python 3.10.20, PyTorch 2.5.1, Gym 0.13.0, and NumPy 1.23.5. They cover data generation, one PCT update, one selection update, frozen proposal weights, and checkpoint reload/evaluation in both discrete and continuous environments. A two-worker Windows spawn run and the proposal-adaptation path also passed. These checks validate execution and policy isolation, not the paper's final training results. Run `python -m unittest discover -s tests -v`; the training smoke test skips itself when its dependencies are absent.

## Generate paper-aligned datasets

The generator removes rotation-equivalent shapes, samples 100 nonuniform categorical distributions, and draws 64 item sequences per distribution by default. Continuous generation adds independent dimension noise in [-0.5, 0.5]. These match the construction in Appendix E.1; sequence length is explicit because it must cover the intended packing horizon.

| `--dataset` | Allowed discrete dimension values |
| --- | --- |
| `Default` | 2, 4, 6, 8, 10 |
| `ID-Large` | 6, 8, 10 |
| `ID-Medium` | 4, 6, 8 |
| `ID-Small` | 2, 4, 6 |
| `OOD` | 1 through 11 |
| `OOD-Large` | 6 through 11 |
| `OOD-Small` | 1 through 6 |

```bash
python generate_dataset.py --dataset Default --output-dir dataset/discrete/Default --sequence-lengths 70 --no-cuda
python generate_dataset.py --dataset OOD --output-dir dataset/discrete/OOD --sequence-lengths 70 --no-cuda
python generate_dataset.py --dataset Default --output-dir dataset/continuous/Default --sequence-lengths 70 --continuous --no-cuda
```

Custom dimension sets use `--item-sizes 2 4 6` or a consecutive range with `--least-size 1 --cross-range 6`. Each output directory contains `box-set.pt` and `num-box-N/instance-I.pt` files. Each instance file stores `dist` and `ins` CPU tensors. Existing generated instances cause an error instead of being overwritten.

## Train, tune, and adapt

First train a base policy. Dataset paths are CLI arguments; source files no longer need to be edited:

```bash
python train_pct.py --datasets dataset/discrete/Default --sequence-lengths 70
```

The final checkpoint is written to `logs/experiment/pct-<timestamp>/pct-final.pt`. The released meta-initialization routine is selected by replacing `train_pct.py` with `meta_train.py`, subject to the MAML limitation above.

Use a Phase-1 checkpoint to initialize both policies, then tune them in their proposal/selection roles. `both` runs a proposal block followed by a selection block; each block lasts `--epochs` epochs:

```bash
python adapt_asap.py --datasets dataset/discrete/Default --sequence-lengths 70 --initial-checkpoint logs/experiment/pct-<timestamp>/pct-final.pt --adapt-policy both --proposal-size 5
```

On a new distribution, load the resulting proposal and selection checkpoints and adapt only the selection policy:

```bash
python adapt_asap.py --datasets dataset/discrete/OOD --sequence-lengths 70 --proposal-checkpoint logs/experiment/asap-<timestamp>/proposal-final.pt --selection-checkpoint logs/experiment/asap-<timestamp>/selection-final.pt --proposal-size 5 --adapt-policy selection
```

Add `--evaluate` to evaluate loaded policies without updates. With `train_pct.py` and `meta_train.py`, evaluation also requires `--initial-checkpoint`. Use `--continuous` consistently for generation and experiments with continuous dimensions. Proposal adaptation is retained as an ablation rather than used as the default deployment procedure.

`--num-processes` must equal the saved dataset batch size. Multiple `--datasets` and matching `--sequence-lengths` are supported when all sequence lengths are equal; run different horizons separately. `--evaluation-steps` cannot exceed the number of saved distributions. `--epochs`, `--training-steps`, and `--evaluation-steps` control run length. `--sample-box-subsets` enables the original randomized box-subset training schedule. Training draws fresh instances from the supplied box sets; saved instances are used for evaluation. A held-out evaluation split must be supplied separately when needed. Space utilization is computed from the environment's volume rewards; the logs divide their sum by 10.

The actor/critic parameter names and ACKTR bias wrappers are retained so existing state-dictionary checkpoints remain structurally compatible when architecture arguments match. Historical selection checkpoints may use the filename prefix `action-`; pass their full path with `--selection-checkpoint`.

## Attribution and release status

The packing environments, PCT architecture, ACKTR optimizer, and vector-environment utilities derive from [Online-3D-BPP-PCT](https://github.com/alexfrom0815/Online-3D-BPP-PCT). Keep the upstream PCT environment names to make that provenance visible. The heuristic implementation also contains its original TAP-Net attribution. See [third-party notices](../THIRD_PARTY_NOTICES.md) for upstream terms, including the PCT academic-use restriction. No new license grant is implied by this cleanup.



