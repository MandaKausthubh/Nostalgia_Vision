from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification import Accuracy


_TV_WEIGHTS = {
    'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
    'resnet50': models.ResNet50_Weights.IMAGENET1K_V2,
    'resnet101': models.ResNet101_Weights.IMAGENET1K_V2,
}


class LitTorchvisionClassifier(pl.LightningModule):
    def __init__(self, arch: str, num_labels: int, lr: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()
        arch = arch.lower()
        if arch not in _TV_WEIGHTS:
            raise ValueError(f"Unsupported torchvision arch: {arch}")
        weights = _TV_WEIGHTS[arch]
        if arch == 'resnet18':
            net = models.resnet18(weights=weights)
        elif arch == 'resnet50':
            net = models.resnet50(weights=weights)
        elif arch == 'resnet101':
            net = models.resnet101(weights=weights)
        else:
            raise ValueError(arch)
        in_feat = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.feature_dim = in_feat
        self.heads = nn.ModuleDict()
        self.active_task = None
        self.add_task_head('default', num_labels)
        self.set_active_task('default')

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc: Optional[Accuracy] = None
        self.val_acc: Optional[Accuracy] = None
        self._reset_metrics_for_head(num_labels)

    def _reset_metrics_for_head(self, num_labels: int):
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels).to(self.device)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels).to(self.device)

    def add_task_head(self, task_id: str, num_labels: int):
        if task_id not in self.heads:
            self.heads[task_id] = nn.Linear(self.feature_dim, num_labels)

    def expand_task_head(self, task_id: str, new_num_labels: int):
        if task_id not in self.heads:
            self.heads[task_id] = nn.Linear(self.feature_dim, new_num_labels)
            return
        head = self.heads[task_id]
        old_out = head.out_features
        if new_num_labels <= old_out:
            return
        new_head = nn.Linear(self.feature_dim, new_num_labels)
        with torch.no_grad():
            new_head.weight[:old_out].copy_(head.weight)
            if head.bias is not None and new_head.bias is not None:
                new_head.bias[:old_out].copy_(head.bias)
        self.heads[task_id] = new_head
        if self.active_task == task_id:
            self._reset_metrics_for_head(new_num_labels)

    def set_active_task(self, task_id: str):
        if task_id not in self.heads:
            raise ValueError(f"Unknown task head: {task_id}")
        self.active_task = task_id
        self._reset_metrics_for_head(self.heads[task_id].out_features)

    def forward(self, x):
        feats = self.backbone(x)
        head = self.heads[self.active_task]
        logits = head(feats)
        return logits

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return opt

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        acc = self.train_acc(pred, y) if stage == "train" else self.val_acc(pred, y)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")
