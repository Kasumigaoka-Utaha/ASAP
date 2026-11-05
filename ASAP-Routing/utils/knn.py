"""Batched KNN with an optional torch-cluster accelerator."""

import torch

try:
    from torch_cluster import knn as _accelerated_knn
except (ImportError, OSError):
    _accelerated_knn = None


def knn(x, y, k, batch_x=None, batch_y=None):
    """Return [query indices, reference indices], matching torch_cluster.knn.

    The fallback computes exact Euclidean neighbors separately for each batch.
    It supports the arguments used by this repository and requires no extension.
    Equal-distance neighbors can be ordered differently by the two backends.
    """
    if _accelerated_knn is not None:
        return _accelerated_knn(x, y, k, batch_x, batch_y)
    if k < 1:
        raise ValueError('k must be positive')
    if batch_x is None:
        batch_x = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    if batch_y is None:
        batch_y = torch.zeros(y.size(0), dtype=torch.long, device=y.device)
    query_rows, reference_rows = [], []
    for batch_id in torch.unique(batch_y, sorted=True):
        reference_idx = (batch_x == batch_id).nonzero(as_tuple=True)[0]
        query_idx = (batch_y == batch_id).nonzero(as_tuple=True)[0]
        count = min(k, reference_idx.numel())
        if not count or not query_idx.numel():
            continue
        distances = torch.cdist(y[query_idx], x[reference_idx])
        local_neighbors = distances.topk(count, dim=1, largest=False, sorted=True).indices
        query_rows.append(query_idx.unsqueeze(1).expand(-1, count).reshape(-1))
        reference_rows.append(reference_idx[local_neighbors].reshape(-1))
    if not query_rows:
        return torch.empty((2, 0), device=x.device, dtype=torch.long)
    return torch.stack((torch.cat(query_rows), torch.cat(reference_rows)))
