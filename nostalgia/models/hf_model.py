from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model
from torchmetrics.classification import Accuracy


class LitHFClassifier(pl.LightningModule):
    def __init__(self, model_id: str, num_labels: int, lr: float, weight_decay: float,
                 use_lora: bool = True, lora_cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.save_hyperparameters()
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.config = AutoConfig.from_pretrained(model_id)
        self.backbone: nn.Module = AutoModel.from_pretrained(model_id, config=self.config)
        if use_lora:
            lcfg = LoraConfig(r=lora_cfg.get("r", 8), lora_alpha=lora_cfg.get("alpha", 16),
                              lora_dropout=lora_cfg.get("dropout", 0.05),
                              target_modules=lora_cfg.get("target_modules", ["query","key","value","output.dense"]))
            self.backbone = get_peft_model(self.backbone, lcfg)
        hidden = getattr(self.config, 'hidden_size', None)
        if hidden is None:
            # Fallback: run a dummy forward later to infer
            hidden = 768
        self.feature_dim = hidden
        self.heads = nn.ModuleDict()
        self.active_task: Optional[str] = None
        # Initialize a default head for the initial dataset
        self.add_task_head('default', num_labels)
        self.set_active_task('default')

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)

    # Task-head management
    def add_task_head(self, task_id: str, num_labels: int):
        if task_id not in self.heads:
            self.heads[task_id] = nn.Linear(self.feature_dim, num_labels)

    def set_active_task(self, task_id: str):
        if task_id not in self.heads:
            raise ValueError(f"Unknown task head: {task_id}")
        self.active_task = task_id

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            feats = out.pooler_output
        else:
            # Use CLS token
            feats = out.last_hidden_state[:, 0]
        return feats

    def forward(self, x):
        feats = self.extract_features(x)
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
