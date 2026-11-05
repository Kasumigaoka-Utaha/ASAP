"""Default container and fallback item set for the PCT packing environments.

Explicit experiment datasets supply their own box-set.pt; these defaults only
initialize the environment and serve the standalone heuristic baselines.
"""
from itertools import product

container_size = [20, 20, 20]
PAPER_ITEM_SIZES = {
    'Default': [2, 4, 6, 8, 10],
    'ID-Large': [6, 8, 10],
    'ID-Medium': [4, 6, 8],
    'ID-Small': [2, 4, 6],
    'OOD': list(range(1, 12)),
    'OOD-Large': list(range(6, 12)),
    'OOD-Small': list(range(1, 7)),
}
item_size_set = list(product(PAPER_ITEM_SIZES['Default'], repeat=3))
