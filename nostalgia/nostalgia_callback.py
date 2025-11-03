from typing import Dict, List, Optional
import torch
import pytorch_lightning as pl


class NostalgiaGradProjector(pl.Callback):
    """Projects gradients of selected parameters onto the complement of top-k eigendirections.

    Q_basis maps parameter names -> tensor of shape [k, param_numel] (orthonormal rows).
    Only parameters with names matching any of the substrings in project_name_filter are projected.
    """

    def __init__(self, project_name_filter: Optional[List[str]] = None):
        super().__init__()
        self.project_name_filter = project_name_filter or ["lora"]
        self.enabled = False
        self.Q_basis: Dict[str, torch.Tensor] = {}

    def set_basis(self, basis: Dict[str, torch.Tensor]):
        self.Q_basis = basis or {}
        self.enabled = len(self.Q_basis) > 0

    def clear_basis(self):
        self.Q_basis = {}
        self.enabled = False

    def _should_project(self, name: str) -> bool:
        return any(s in name for s in self.project_name_filter)

    @torch.no_grad()
    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        if not self.enabled:
            return
        # For each parameter with grad and in filter, project gradient
        named_params = dict(pl_module.named_parameters())
        for name, p in named_params.items():
            if p.grad is None or not p.requires_grad:
                continue
            if not self._should_project(name):
                continue
            q = self.Q_basis.get(name)
            if q is None:
                continue
            g = p.grad.view(-1)
            # q: [k, D]; project: g_perp = g - q^T(q g) with rows orthonormal
            # compute (q @ g) -> [k], then q.t() @ that -> [D]
            qt_g = torch.matmul(q, g)
            proj = torch.matmul(q.t(), qt_g)
            g_perp = g - proj
            p.grad.copy_(g_perp.view_as(p))
