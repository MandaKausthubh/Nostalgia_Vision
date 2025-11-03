from typing import Any, Dict, Optional
import torch
import pytorch_lightning as pl
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import LoraConfig, get_peft_model
from torchmetrics.classification import Accuracy


class LitHFClassifier(pl.LightningModule):
    def __init__(self, model_id: str, num_labels: int, lr: float, weight_decay: float,
                 use_lora: bool = True, lora_cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.save_hyperparameters()
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id, num_labels=num_labels)
        if use_lora:
            lcfg = LoraConfig(r=lora_cfg.get("r", 8), lora_alpha=lora_cfg.get("alpha", 16),
                              lora_dropout=lora_cfg.get("dropout", 0.05),
                              target_modules=lora_cfg.get("target_modules", ["query","key","value","output.dense"]))
            self.model = get_peft_model(self.model, lcfg)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)

    def forward(self, x):
        return self.model(pixel_values=x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return opt

    def _step(self, batch, stage: str):
        x, y = batch
        out = self(x)
        logits = out.logits
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
