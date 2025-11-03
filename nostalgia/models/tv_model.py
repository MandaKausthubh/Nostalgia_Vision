from typing import Any, Dict, Optional
import torch
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
        net.fc = torch.nn.Linear(in_feat, num_labels)
        self.model = net
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)

    def forward(self, x):
        logits = self.model(x)
        return torch.nn.functional.softmax(logits, dim=1) if self.training is False else logits

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return opt

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self.model(x)
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
