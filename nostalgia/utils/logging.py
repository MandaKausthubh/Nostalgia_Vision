from typing import Optional, List, Tuple
import os
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def build_loggers(
    save_dir: str,
    name: str,
    use_wandb: bool = False,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    mode: Optional[str] = None,
) -> Tuple[List, List]:
    """Builds loggers (TensorBoard, CSV, and optional W&B) and common callbacks.

    Parameters
    ----------
    save_dir: Base directory where log files/artifacts are saved.
    name: Experiment name (used as run name for TB/W&B).
    use_wandb: Whether to initialize a Weights & Biases logger.
    project, entity, tags, group, mode: W&B configuration. `mode` can be 'online' or 'offline'.
    """
    os.makedirs(save_dir, exist_ok=True)
    loggers: List = []

    tb = TensorBoardLogger(save_dir=save_dir, name=name, default_hp_metric=False)
    csv = CSVLogger(save_dir=save_dir, name=name)
    loggers.extend([tb, csv])

    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
            offline = (mode == 'offline') if mode is not None else False
            wb = WandbLogger(
                project=project or name,
                name=name,
                save_dir=save_dir,
                entity=entity,
                offline=offline,
                tags=tags,
                group=group,
                log_model=False,
            )
            loggers.append(wb)
        except Exception as e:
            # Soft-fail if wandb not installed or misconfigured
            print(f"[logging] W&B logger not enabled ({e}); continuing without it.")

    ckpt = ModelCheckpoint(dirpath=os.path.join(save_dir, name, "checkpoints"),
                           filename="epoch{epoch:03d}-val_acc{val/acc:.4f}",
                           save_last=True, save_top_k=1, monitor="val/acc", mode="max", auto_insert_metric_name=False)
    lr = LearningRateMonitor(logging_interval="step")
    return loggers, [ckpt, lr]
