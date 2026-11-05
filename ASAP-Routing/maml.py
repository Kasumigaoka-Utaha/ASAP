"""Differentiable MAML updates shared by routing initialization experiments.

Each task supplies independent support and query batches. Inner SGD creates new
parameter tensors; it never copies weights into the base module or steps its
optimizer. The default retains the support Hessian in the outer gradient.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import nn

try:
    from torch.func import functional_call as _functional_call
except ImportError:  # PyTorch 1.12/1.13
    from torch.nn.utils.stateless import functional_call as _functional_call


@dataclass
class DistributionTask:
    distribution: str
    support: Sequence[Any]
    query: Any


def call_with_parameters(module, parameters, buffers, *args, **kwargs):
    """Evaluate a module without replacing its persistent parameters/buffers."""
    state = {**parameters, **buffers}
    return _functional_call(module, state, args, kwargs, strict=False)


def adapt_parameters(
    module: nn.Module,
    support_batches: Sequence[Any],
    loss_fn: Callable,
    inner_lr: float,
    first_order: bool = False,
):
    """Return task-specific parameters linked to the common initialization.

    ``loss_fn(module, parameters, buffers, batch)`` must return a scalar. Buffers
    are task-local clones so running statistics cannot leak between tasks.
    First-order MAML detaches the support gradient, preserving the identity
    path from the adapted weights back to the initialization.
    """
    if not support_batches:
        raise ValueError("Each MAML task needs at least one support batch")
    if inner_lr <= 0:
        raise ValueError("inner_lr must be positive")
    parameters = OrderedDict((name, value) for name, value in module.named_parameters()
                             if value.requires_grad)
    if not parameters:
        raise ValueError("MAML requires trainable parameters")
    buffers = OrderedDict((name, value.detach().clone())
                          for name, value in module.named_buffers())
    for batch in support_batches:
        loss = loss_fn(module, parameters, buffers, batch)
        gradients = torch.autograd.grad(
            loss, tuple(parameters.values()), create_graph=not first_order,
            allow_unused=True,
        )
        parameters = OrderedDict(
            (name, value if gradient is None else value - inner_lr * gradient)
            for (name, value), gradient in zip(parameters.items(), gradients)
        )
    return parameters, buffers


def maml_task_loss(module, task, loss_fn, inner_lr, first_order=False):
    """Adapt on support data, then evaluate on held-out query data."""
    parameters, buffers = adapt_parameters(
        module, task.support, loss_fn, inner_lr, first_order=first_order
    )
    return loss_fn(module, parameters, buffers, task.query)


def maml_objective(module, tasks, loss_fn, inner_lr, first_order=False):
    """Average task query losses; every task starts from the same base weights."""
    if not tasks:
        raise ValueError("A meta-batch must contain at least one distribution task")
    losses = [maml_task_loss(module, task, loss_fn, inner_lr, first_order)
              for task in tasks]
    return torch.stack(losses).mean()
