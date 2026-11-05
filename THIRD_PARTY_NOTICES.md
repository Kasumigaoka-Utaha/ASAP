# Third-party notices

The [MIT license](LICENSE) covers original ASAP contributions only. Existing
third-party terms continue to apply to their respective code. Attribution does
not grant permission to relicense that code.

## Packing Configuration Trees (PCT)

The environment, packing representation, model, ACKTR/KFAC utilities, vector
environment wrappers, and heuristic code in `ASAP-3DBPP/` build on
[Online-3D-BPP-PCT](https://github.com/alexfrom0815/Online-3D-BPP-PCT), accompanying
*Learning Efficient Online 3D Bin Packing on Packing Configuration Trees*
(Hang Zhao, Yang Yu, Kai Xu; ICLR 2022).

The local upstream revision `0caf883` records this notice in its README:

> This source code is released only for academic use. Please do not use it for commercial purposes without authorization of the author.

This notice is retained here because the previous local README omitted it.
No replacement permission or commercial-use authorization is included in this
release. The PCT-derived component therefore retains this restriction, regardless
of the license chosen for new ASAP contributions.

`ASAP-3DBPP/heuristic.py` also explicitly credits the score function from
[TAP-Net](https://github.com/Juzhan/TAP-Net/blob/master/tools.py). That attribution
is preserved in the source.

## Routing baselines and solver wrappers

The routing implementation builds on the INViT and POMO model families discussed
in the ASAP paper. Preserve the source authors' notices and consult the upstream
terms when redistributing derived implementations:

- INViT: *INViT: A Generalizable Routing Problem Solver with Invariant Nested
  View Transformer*, Fang et al. (2024).
- POMO: *POMO: Policy Optimization with Multiple Optima for Reinforcement
  Learning*, Kwon et al. (2020).
- `ASAP-Routing/baselines/solve_tsp_baselines.py` explicitly credits code copied
  and adapted from `Gurobi-examples/tsp.py`; the attribution remains in that file.

The supplied routing snapshot did not contain standalone upstream license files
for these components. This notice records the identifiable sources; it does not
assert that they are all covered by MIT.

Gurobi, LKH/Elkai, HGS/PyVRP, PyTorch, Gym, and other dependencies retain their
own licenses. Solver binaries and datasets are not bundled with this source.
