from typing import Callable
import torch


def hvp(loss_fn: Callable[[], torch.Tensor], params, v: torch.Tensor) -> torch.Tensor:
    """Compute Hessian-vector product H v for current loss wrt params.
    loss_fn: closure returning scalar loss with create_graph=True upstream
    params: iterable of parameters
    v: flattened vector (same numel as concatenated params)
    Returns flattened H v.
    """
    idx = 0
    grads = torch.autograd.grad(loss_fn(), params, create_graph=True, retain_graph=True)
    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])

    Hv = torch.autograd.grad(flat_grads, params, grad_outputs=v, retain_graph=True)
    flat_Hv = torch.cat([h.contiguous().view(-1) for h in Hv])
    return flat_Hv
