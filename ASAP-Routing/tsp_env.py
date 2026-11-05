import torch


class TSPEnvironment:
    """Environment for batched TSP decoding.

    Holds the graph coordinates and decoding state (visited mask, first/last node, tour).
    Uses the same initialization and state updates as TSP_net.forward but exposes step-wise transitions.
    """

    def __init__(self, x: torch.Tensor, start_idx: torch.Tensor = None):
        assert x.dim() == 3, "x must be (bsz, nb_nodes, dim_input)"
        self.x = x
        self.device = x.device
        self.bsz, self.nb_nodes, self.dim_input = x.shape
        # choose start
        if start_idx is None:
            self.start_idx = torch.randint(self.nb_nodes, (self.bsz,), device=self.device)
        else:
            assert start_idx.shape == (self.bsz,)
            self.start_idx = start_idx.to(self.device)

        zero_to_bsz = torch.arange(self.bsz, device=self.device)
        self.first_visited_node = x[zero_to_bsz, self.start_idx, :].view((self.bsz, 1, -1))
        self.last_visited_node = self.first_visited_node.clone()

        # mask: True means unvisited/available, False means already visited
        self.mask_global = torch.ones((self.bsz, self.nb_nodes), device=self.device, dtype=torch.bool)
        self.mask_global[zero_to_bsz, self.start_idx] = False

        self.tours: list[torch.Tensor] = [self.start_idx]
        self.step_idx = 0

    def observation(self) -> dict:
        return {
            "x": self.x,
            "last_visited_node": self.last_visited_node,
            "first_visited_node": self.first_visited_node,
            "mask_global": self.mask_global,
        }

    def step(self, action_global_idx: torch.Tensor) -> tuple[dict, bool]:
        """Apply chosen global indices (bsz,) to update state. Returns (obs, done)."""
        assert action_global_idx.shape == (self.bsz,)
        zero_to_bsz = torch.arange(self.bsz, device=self.device)
        self.last_visited_node = self.x[zero_to_bsz, action_global_idx, :].view((self.bsz, 1, -1))
        self.mask_global[zero_to_bsz, action_global_idx] = False
        self.tours.append(action_global_idx)
        self.step_idx += 1
        done = self.step_idx >= (self.nb_nodes - 1)
        return self.observation(), done

    def get_tour_tensor(self) -> torch.Tensor:
        return torch.stack(self.tours, dim=1)

