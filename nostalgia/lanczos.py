from typing import Dict, Iterable, Tuple
import torch
from hessian_eigenthings import compute_hessian_eigenthings


def topk_eigs_with_eigenthings(model: torch.nn.Module,
                               loss_fn,
                               dataloader,
                               num_eigenthings: int = 32,
                               use_gpu: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k eigenpairs (eigenvalues, eigenvectors) of the loss Hessian using hessian-eigenthings.
    Returns (eigvals[k], eigvecs[k, D]) where eigvecs are flattened in parameter order.
    """
    device = next(model.parameters()).device
    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        num_eigenthings=num_eigenthings,
        mode='lanczos',
        max_steps=50,
        use_gpu=use_gpu,
    )
    # eigenvecs is typically a list of tensors per eigenvector matching parameter shapes.
    # Flatten to [k, D]
    flat_vecs = []
    for vec in eigenvecs:
        flat = torch.cat([v.contiguous().view(-1) for v in vec])
        flat_vecs.append(flat)
    V = torch.stack(flat_vecs, dim=0)
    return eigenvals, V


def build_param_basis(model: torch.nn.Module,
                      eigenvecs_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Map flattened eigenvectors [k, D] to per-parameter bases {name: [k, numel(name_param)]}.
    Returns row-orthonormal bases per parameter name.
    """
    # Build slices for each parameter
    slices = {}
    start = 0
    for name, p in model.named_parameters():
        numel = p.numel()
        slices[name] = slice(start, start + numel)
        start += numel

    basis: Dict[str, torch.Tensor] = {}
    for name, slc in slices.items():
        q = eigenvecs_flat[:, slc]  # [k, numel]
        # Orthonormalize rows via QR on transposed
        qt, _ = torch.linalg.qr(q.t(), mode='reduced')  # [numel, k]
        basis[name] = qt.t().contiguous()               # [k, numel]
    return basis
