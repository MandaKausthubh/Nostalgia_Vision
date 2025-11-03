from typing import Optional, List, Tuple
import os
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def build_loggers(save_dir: str, name: str) -> Tuple[List, List]:
    os.makedirs(save_dir, exist_ok=True)
    tb = TensorBoardLogger(save_dir=save_dir, name=name, default_hp_metric=False)
    csv = CSVLogger(save_dir=save_dir, name=name)
    ckpt = ModelCheckpoint(dirpath=os.path.join(save_dir, name, "checkpoints"),
                           filename="epoch{epoch:03d}-val_acc{val/acc:.4f}",
                           save_last=True, save_top_k=1, monitor="val/acc", mode="max", auto_insert_metric_name=False)
    lr = LearningRateMonitor(logging_interval="step")
    return [tb, csv], [ckpt, lr]
