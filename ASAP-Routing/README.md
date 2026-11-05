# ASAP for Routing Problems (TSP and CVRP)

Routing implementation accompanying [ASAP: Exploiting the Satisficing Generalization Edge in Neural Combinatorial Optimization](https://proceedings.mlr.press/v306/fang26t.html), ICML 2026. See the [repository README](../README.md) for the paper, citation, and release status.

ASAP means **Adaptive Selection After Proposal**. The proposal policy selects a subset of promising feasible actions; the selection policy chooses the next action from that subset. The routing backbones in this code are **INViT** and **POMO**.

## Installation

Run commands from this directory. Install the PyTorch build matching your CPU/CUDA platform, then:

```bash
python -m pip install -r requirements.txt
```

The exact batched KNN fallback uses PyTorch. `requirements-fast.txt` optionally adds `torch-cluster` acceleration; match its wheel to your PyTorch/CUDA build. Equal-distance neighbors can be ordered differently by the two implementations. Reference solvers are optional (`requirements-baselines.txt`). SwanLab logging is optional and disabled by default; install `swanlab` and pass `--use_swanlab true` to enable it.

The paper reports Python 3.9 and PyTorch 1.12. The requirements identify dependencies; they are not a recovered lockfile of the original experiments.

## Paper-to-code mapping

| Paper concept | Implementation |
| --- | --- |
| Proposal policy, pi_p | `TSPStage1Policy`, `VRPStage1Policy`, `POMOTSPStage1Policy`, `POMOVRPStage1Policy`; `select_k` |
| Selection policy, pi_s | Corresponding `Stage2Policy` classes; `select_action` with a candidate subset |
| Proposal subset size k | `--k_promising` |
| Phase 1: initialization/pretraining | `pretrain_policy.py`, `pretrain_pomo_tsp_policy.py`, `pretrain_pomo_cvrp_policy.py` |
| Phase 2: cooperative tuning | `train_asap.py`, `train_asap_pomo_tsp.py`, `train_asap_pomo_cvrp.py`; `--stage alternating` |
| Adaptation with frozen proposal | `adapt_selection_{tsp,cvrp,pomo_tsp,pomo_cvrp}.py` |
| Proposal adaptation comparisons | `adapt_proposal_{tsp,cvrp,pomo_tsp,pomo_cvrp}.py`; inspect each script's training objective before interpreting an ablation |
| INViT baseline | `train_invit_baseline.py`, `training_invit_baseline.py`, `evaluate_invit_baseline.py`, `TSP_net.py`, `vrp_policy.py` |
| Routing distributions | `generator/generate_instances.py`: Uniform, Clustered, Explosion, Implosion |
| Reference solvers | `baselines/`: Gurobi/LKH3 for TSP and HGS for CVRP |

`stage1` and `stage2` identify **policies**, not the paper's training phases. CLI accepts `proposal`/`selection`; the old names and checkpoint fields remain supported. Thus `--proposal_ckpt` aliases `--stage1_ckpt`, and `--selection_ckpt` aliases `--stage2_ckpt`. Model class names, parameter names, and checkpoint tags remain unchanged.

## INViT workflow

The example YAML files use the supported paper parameters: TSP-100/CVRP-50, batch size 64, 8 training augmentations, embedding 128, feed-forward 512, 8 heads, 2 encoder layers, 3 decoder layers, k=8, and learning rate 1e-4. Initialization is 400 epochs and the cooperative schedules are 50 epochs. These are examples for the released implementation, not a claim of numerical reproduction.

CVRP uses capacity Q=50 by default and integer demands 1 through 10, following Appendix E.2. `--capacity` is honored independently of graph size. Feasibility includes demands equal to the remaining vehicle capacity.

INViT proposal and selection modules have different parameter structures. Pretrain them separately; a proposal checkpoint cannot be strictly loaded into the selection module.

```bash
python run_config.py pretrain_policy --config configs/pretrain_tsp.yaml --stage proposal
python run_config.py pretrain_policy --config configs/pretrain_tsp.yaml --stage selection
python run_config.py train_asap --config configs/asap_tsp.yaml --stage alternating --proposal_init_ckpt PATH_TO_PROPOSAL_PRETRAIN --selection_init_ckpt PATH_TO_SELECTION_PRETRAIN
```

Use `configs/pretrain_cvrp.yaml` and `configs/asap_cvrp.yaml` for CVRP. Supply actual checkpoint paths produced by your runs. Outputs go under the requested `--save_dir`, followed by a timestamped run directory; cooperative tuning writes `stage1.ckpt` and `stage2.ckpt`. `run_pretrain.sh` and `run_asap.sh` are Bash conveniences for the same Python configuration launcher. Add `--dry-run` to `run_config.py` to inspect the resolved command.

To adapt only the selection policy to a new distribution:

```bash
python adapt_selection_tsp.py --nb_nodes 100 --distribution clustered --proposal_ckpt PATH_TO_PROPOSAL --selection_init_ckpt PATH_TO_SELECTION --data_path ./data/ --eval_sizes 100,1000 --eval_distributions clustered1,clustered2
python adapt_selection_cvrp.py --nb_nodes 50 --distribution explosion --proposal_ckpt PATH_TO_PROPOSAL --selection_init_ckpt PATH_TO_SELECTION --data_path ./data/ --eval_sizes 50,500 --eval_distributions explosion
```

These adaptation scripts evaluate a dataset before and after training, so the example commands need evaluation data. Use `explosion`, `implosion`, `uniform`, or `clustered` for training; the stored dataset names `clustered1` and `clustered2` identify the two Clustered evaluation variants.

## POMO workflow

Use `pretrain_pomo_tsp_policy.py` / `pretrain_pomo_cvrp_policy.py`, then `train_asap_pomo_tsp.py` / `train_asap_pomo_cvrp.py`, and finally `adapt_selection_pomo_tsp.py` / `adapt_selection_pomo_cvrp.py`. They accept the same proposal/selection checkpoint aliases and `--stage proposal|selection|alternating` where applicable.

POMO architecture arguments differ from INViT: `--embedding_dim`, `--encoder_layer_num`, `--qkv_dim`, `--head_num`, and `--ff_hidden_dim`. Existing POMO defaults use **6 encoder layers**; Appendix E.2 states **2**. Pass `--encoder_layer_num 2` consistently for a new run using that setting, and use the architecture that matches your checkpoint when evaluating. The current POMO decoder structure is the supplied implementation; its equivalence to the paper's decoder description has not been established. Renaming does not establish identical experimental results.

## Distribution-task MAML

`meta_train.py` provides a newly added MAML initialization path. Uniform, Clustered, Explosion, and Implosion distributions form tasks with independent support/query instances. It differentiates through inner SGD updates by default; `--first-order` selects the first-order approximation.

```bash
python meta_train.py --backbone invit --problem tsp --policy-role proposal --task-distributions uniform clustered explosion implosion --meta-steps 1000 --output-dir checkpoints/maml/invit_tsp_proposal
python meta_train.py --backbone invit --problem tsp --policy-role selection --proposal-ckpt PATH_TO_PROPOSAL --task-distributions uniform clustered explosion implosion --meta-steps 1000 --output-dir checkpoints/maml/invit_tsp_selection
```

Use `--backbone pomo` and/or `--problem cvrp` for the other combinations. Selection MAML freezes the supplied proposal; without `--proposal-ckpt` it uses KNN candidates for a warm start. Meta checkpoints expose `policy_state_dict` and the `pretrain` tag so they can initialize the corresponding policy in cooperative tuning. INViT requires separate checkpoints for its two policy architectures. MAML options use hyphens (for example `--nb-nodes`), while the existing training/adaptation scripts retain underscores. This training path has not reproduced the paper's result tables.

## Evaluation and data

`evaluate_asap.py` evaluates both policies for all four combinations of `--backbone invit|pomo` and `--problem tsp|cvrp`:

```bash
python evaluate_asap.py --backbone invit --problem tsp --proposal_ckpt PATH_TO_PROPOSAL --selection_ckpt PATH_TO_SELECTION --data_path ./data/ --eval_sizes 100,1000 --eval_distributions explosion
python evaluate_asap.py --backbone invit --problem cvrp --proposal_ckpt PATH_TO_PROPOSAL --selection_ckpt PATH_TO_SELECTION --data_path ./data/ --eval_sizes 50,500 --eval_distributions explosion
```

Use `--eval_mode tsplib` or `--eval_mode cvrplib` for benchmark files, and `--eval_mode synthetic --eval_distributions uniform,clustered,explosion,implosion` for generated instances without reference optimality gaps. `evaluate_pretrain.py` handles pretrained models. Checkpoint loading in the ASAP evaluator is strict, so all architecture options must match training.

Dataset evaluation expects `data_path/data_farm/` and `data_path/solution_farm/` in the format read by `load_data.py`; datasets, reference tours, and pretrained checkpoints are not bundled. `generator/data_farm.py` and the scripts under `baselines/` generate those files; use `--help` to inspect their arguments. Benchmark evaluation decodes normalized coordinates and measures the resulting tours on the **original coordinates** before comparing with reference lengths. The metric functions use Euclidean distances; exact benchmark distance conventions must be checked for instances using other conventions.

## Validation

From the repository root, run:

```bash
python -m unittest discover -s tests -p test_routing_entrypoints.py -v
python -m unittest discover -s tests -p test_maml.py -v
```

The routing checks exercise actual CLI parsing, tiny cooperative training, checkpoint writing/loading, synthetic evaluation for all four backbone/problem combinations, KNN batch isolation, and exact-capacity feasibility. MAML checks cover gradient connectivity and small real-policy support/query updates. The preparation environment uses Python 3.10.20, PyTorch 2.5.1 (CPU), NumPy 1.23.5, and PyYAML 6.0.3. Runtime, convergence, and paper table reproduction require the corresponding data, checkpoints, and experiment settings.
