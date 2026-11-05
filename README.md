# ASAP

Code for **ASAP: Exploiting the Satisficing Generalization Edge in Neural
Combinatorial Optimization**, Han Fang, Paul Weng, and Yutong Ban, ICML 2026.

[Paper](https://proceedings.mlr.press/v306/fang26t.html) ·
[Project page](https://kasumigaoka-utaha.github.io/asap-web/) ·
[Paper-to-code mapping](docs/PAPER_MAPPING.md)

ASAP (**Adaptive Selection After Proposal**) separates each decision into a
**proposal policy**, which produces a small candidate action set, and a
**selection policy**, which chooses an action from that set. On a new
distribution, ASAP freezes the proposal policy and adapts the selection policy.
The paper studies this framework on online **3D bin packing (3D-BPP)** and
**routing (TSP and CVRP)**.

## Repository

| Directory | Problems and implementation | Instructions |
| --- | --- | --- |
| `ASAP-3DBPP/` | Discrete/continuous online 3D-BPP; PCT backbone and ACKTR training | [3D-BPP README](ASAP-3DBPP/README.md) |
| `ASAP-Routing/` | TSP/CVRP; INViT and POMO backbones; distribution-task MAML | [Routing README](ASAP-Routing/README.md) |
| `docs/` | Method correspondence, implementation scope, cleanup notes | [Mapping](docs/PAPER_MAPPING.md), [changes](docs/CLEANUP.md) |

Run each component from its own directory. They retain separate environments
because 3D-BPP depends on the legacy Gym interface. Installation instructions and
dataset formats are documented in each component README. Data, pretrained
checkpoints, solver binaries, and paper result files are not bundled.

## Training and adaptation

The paper's two **training phases** differ from its two **decision stages**:

1. **Pretraining / initialization:** train a base policy, optionally with MAML
   across distribution tasks.
2. **Cooperative tuning:** initialize the proposal and selection policies, then
   tune them in their respective roles.
3. **Online adaptation:** freeze the proposal policy and update the selection
   policy on a shifted distribution.

The routing MAML implementation treats Uniform, Clustered, Explosion, and
Implosion distributions as distinct tasks, with separate support and query
batches. It was added during repository preparation and is not a recovered
implementation of the checkpoints used for the published tables.

The retained 3D-BPP meta-training routine uses a legacy surrogate update and is
**not a faithful MAML implementation**. See the
[implementation scope](docs/PAPER_MAPPING.md) before interpreting a run as a
reproduction of the paper. Paper-table reproduction requires the corresponding
datasets, seeds, architectures, and training schedules.

## Source validation

From the repository root:

```bash
python tools/check_source.py
```

This checks Python syntax, local import symbols, and documentation links without
requiring the training dependencies. Model-level tests and smoke commands are
documented with their component.

With the corresponding dependencies installed, run the routing MAML and
integration tests from the root, and the packing tests separately:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s ASAP-3DBPP/tests -v
```

To create a source archive without nested Git metadata, caches, virtual
environments, datasets, or checkpoints:

```bash
python tools/export_source.py
```

The output is `dist/ASAP-source.zip`, containing a single `ASAP/` source directory.
Git metadata, dependencies, datasets, and experiment outputs are excluded.

## Citation

```bibtex
@InProceedings{pmlr-v306-fang26t,
  title = {{ASAP}: Exploiting the Satisficing Generalization Edge in Neural Combinatorial Optimization},
  author = {Fang, Han and Weng, Paul and Ban, Yutong},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages = {29417--29446},
  year = {2026},
  volume = {306},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v306/fang26t.html}
}
```

## License and acknowledgements

Original ASAP contributions are licensed under [MIT](LICENSE). The PCT-derived
3D-BPP code retains an **academic-use-only** upstream notice, and other third-party
terms continue to apply. The combined repository is therefore not covered by MIT
alone. See [third-party notices](THIRD_PARTY_NOTICES.md) for the recorded terms and
attributions to PCT, INViT, POMO, TAP-Net, and the solver wrappers.
