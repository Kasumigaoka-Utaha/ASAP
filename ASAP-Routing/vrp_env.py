import torch

from utils.utils_for_model import is_vrp_finished, get_knn_candidate, create_distance_mask_for_knn


class VRPEnvironment:
    """Environment wrapper for VRP rollout state and candidate generation."""

    def __init__(self, x: dict, capacity: float, problem: str = 'cvrp'):
        """
        x: dict with keys 'loc' (bsz, nb_nodes, 2), 'demand' (bsz, nb_nodes), 'depot' (bsz, 2)
        capacity: scalar vehicle capacity
        """
        self.problem = problem
        self.capacity = float(capacity)

        self.nodes: torch.Tensor = x['loc']                      # (bsz, nb_nodes, dim)
        raw_demands: torch.Tensor = x['demand']                  # (bsz, nb_nodes)
        # Force integer demands inside the environment to avoid residual float noise.
        if torch.is_floating_point(raw_demands):
            demand_int = torch.round(raw_demands)
        else:
            demand_int = raw_demands
        self.true_demands: torch.Tensor = demand_int.long()
        self.depot: torch.Tensor = x['depot'].unsqueeze(1)       # (bsz, 1, dim)

        self.bsz = self.nodes.size(0)
        self.nb_nodes = self.nodes.size(1)
        self.dim_input = self.nodes.size(2)
        self.zero_to_bsz = torch.arange(self.bsz, device=self.nodes.device)

        # Append depot to graph/demand
        self.full_graph = torch.cat((self.nodes, self.depot), dim=1)  # (bsz, nb_nodes+1, dim)
        depot_demands = torch.zeros((self.bsz, 1), device=self.nodes.device, dtype=self.true_demands.dtype)
        self.true_demands = torch.cat((self.true_demands, depot_demands), dim=1).detach()

        capacity_int = int(round(self.capacity))
        self.true_capacity_vec = torch.full((self.bsz, 1), fill_value=capacity_int, device=self.nodes.device).long()
        self.true_used_capacity_vec = torch.zeros((self.bsz, 1), device=self.nodes.device).long()

        # Track current position (start at depot)
        self.depot_idx = torch.zeros((self.bsz, 1), device=self.nodes.device, dtype=torch.long) - 1
        self.last_visited_node = self.depot
        self.last_visited_idx = self.depot_idx.clone()
        self._update_full_demands()

    def _update_full_demands(self):
        # Normalized float demands (0-1) fed to policies; clamp to guard against numerical drift.
        self.full_demands = (self.true_demands.float() / self.capacity).clamp_(min=0.0, max=1.0)

    def is_finished(self) -> bool:
        # print(self.true_demands)
        return is_vrp_finished(self.true_demands)

    def build_step_context(self, action_k: int, state_k: list[int], if_use_local_mask: bool = False) -> dict:
        nodes = self.nodes
        demands = self.full_demands[:, :self.nb_nodes]
        remain_capacity_vec = (
            (self.true_capacity_vec - self.true_used_capacity_vec).float() / self.capacity
        ).clamp_(min=0.0, max=1.0)
        finished_mask = ~(demands > 0)

        if self.problem == 'cvrp':
            available_action_mask = ~((demands > 0) * (demands <= remain_capacity_vec + 1e-6))
        elif self.problem == 'sdvrp':
            available_action_mask = ~(demands > 0)
        else:
            raise ValueError(f"Unsupported problem type: {self.problem}")

        depot_bsz = (self.last_visited_idx.squeeze() == -1) * (torch.sum(demands, dim=1) != 0)

        action_idx, action_mask = get_knn_candidate(
            nodes, action_k, self.last_visited_node, self.last_visited_idx, mask=available_action_mask
        )

        state_idx, state_mask = None, None
        if len(state_k) > 0:
            k_state = max(state_k) - action_k
            b_a = torch.arange(0, self.bsz, device=nodes.device).view((-1, 1)).repeat((1, action_k))
            action_bsz = b_a[~action_mask]
            ref_idx = action_idx[~action_mask]
            mask_for_state = finished_mask.clone()
            mask_for_state[action_bsz, ref_idx] = True
            state_idx, state_mask = get_knn_candidate(
                nodes, k_state, self.last_visited_node, self.last_visited_idx, mask=mask_for_state
            )
            state_idx = torch.cat((action_idx, state_idx), dim=1)
            state_mask = torch.cat((action_mask, state_mask), dim=1)

        if if_use_local_mask:
            action_mask = create_distance_mask_for_knn(self.last_visited_node, action_idx, nodes, action_mask)

        action_idx_for_choice = torch.cat((action_idx, self.depot_idx), dim=1)

        return {
            "action_idx": action_idx,
            "action_mask": action_mask,
            "state_idx": state_idx,
            "state_mask": state_mask,
            "action_idx_for_choice": action_idx_for_choice,
            "depot_bsz": depot_bsz,
            "demands": demands,
            "remain_capacity_vec": remain_capacity_vec,
            "finished_mask": finished_mask,
        }

    def step(self, next_node_idx: torch.Tensor):
        """Apply chosen action and update internal state."""
        last_node = self.full_graph[self.zero_to_bsz, next_node_idx].view((self.bsz, 1, self.dim_input))
        self.last_visited_node = last_node
        self.last_visited_idx = next_node_idx.view((self.bsz, 1))

        if self.problem == 'cvrp':
            last_visited_demand = self.true_demands[self.zero_to_bsz, next_node_idx]
            at_depot = next_node_idx == -1
            new_used_capacity_vec = torch.where(
                at_depot,
                torch.zeros_like(self.true_used_capacity_vec.squeeze()),
                last_visited_demand + self.true_used_capacity_vec.squeeze(),
            )
            self.true_demands[self.zero_to_bsz, next_node_idx] = 0
        elif self.problem == 'sdvrp':
            last_visited_demand = self.true_demands[self.zero_to_bsz, next_node_idx].unsqueeze(dim=1)
            true_remain_capacity_vec = self.true_capacity_vec - self.true_used_capacity_vec
            true_filled_demand = torch.min(torch.cat((last_visited_demand, true_remain_capacity_vec), dim=1), dim=1).values
            at_depot = next_node_idx == -1
            new_used_capacity_vec = torch.where(
                at_depot,
                torch.zeros_like(self.true_used_capacity_vec.squeeze()),
                true_filled_demand + self.true_used_capacity_vec.squeeze(),
            )
            updated_demands = self.true_demands[self.zero_to_bsz, next_node_idx] - true_filled_demand.long()
            self.true_demands[self.zero_to_bsz, next_node_idx] = updated_demands.clamp_min(0)
        else:
            raise ValueError(f"Unsupported problem type: {self.problem}")

        self.true_used_capacity_vec = new_used_capacity_vec.unsqueeze(dim=1)
        self._update_full_demands()
        # print(self.true_used_capacity_vec)

    def get_tour_tensor(self, tours: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tours, dim=1)
