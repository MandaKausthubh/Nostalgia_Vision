import os
from typing import Dict
import argparse
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import Callback
from nostalgia.utils.seed import set_seed
from nostalgia.utils.logging import build_loggers
from nostalgia.utils.memory import MemoryComputeLogger
from nostalgia.data_utils.datasets import get_cifar100, get_caltech256, get_imagenet1k
from nostalgia.data_utils.datasets import get_loaders_and_num_classes
from nostalgia.models.hf_model import LitHFClassifier
from nostalgia.models.tv_model import LitTorchvisionClassifier
from nostalgia.nostalgia_callback import NostalgiaGradProjector
from nostalgia.lanczos import topk_eigs_with_eigenthings, build_param_basis
from nostalgia.scripts.plot_all import main as plot_main
from nostalgia.scripts.plot_all import run_plots


def _build_loaders(cfg: DictConfig):
    framework = cfg.model.get("framework", "hf")
    model_id = cfg.model.model_id if framework == 'hf' else None
    name = cfg.dataset.name.lower()
    if name == 'cifar100':
        return get_cifar100(cfg.dataset.root, model_id, cfg.dataset.image_size, cfg.dataset.batch_size, cfg.dataset.num_workers)
    if name == 'caltech256':
        return get_caltech256(cfg.dataset.root, model_id, cfg.dataset.image_size, cfg.dataset.batch_size, cfg.dataset.num_workers)
    if name in ('imagenet1k', 'imagenet', 'ilsvrc'):
        return get_imagenet1k(cfg.dataset.root, model_id, cfg.dataset.image_size, cfg.dataset.batch_size, cfg.dataset.num_workers)
    raise ValueError(f"Unknown dataset: {cfg.dataset.name}")


def _build_model(cfg: DictConfig):
    framework = cfg.model.get("framework", "hf")
    if framework == 'hf':
        use_lora = cfg.model.get("use_lora", True)
        lora_cfg = cfg.model.get("lora", {})
        model = LitHFClassifier(
            model_id=cfg.model.model_id,
            num_labels=cfg.dataset.num_classes,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            use_lora=use_lora,
            lora_cfg=lora_cfg,
        )
    elif framework == 'torchvision':
        model = LitTorchvisionClassifier(
            arch=cfg.model.arch,
            num_labels=cfg.dataset.num_classes,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    else:
        raise ValueError(f"Unknown framework: {framework}")
    return model


@hydra.main(config_path="configs", config_name="config", version_base=None)
def hydra_entry(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.experiment.seed)

    # Multi-task sequences configuration
    # If cfg.experiment.tasks is provided, use that list of dataset names in order;
    # else, repeat cfg.dataset.name for num_tasks times (backward compatibility).
    tasks = list(getattr(cfg.experiment, 'tasks', []))
    if not tasks:
        tasks = [cfg.dataset.name for _ in range(cfg.experiment.get('num_tasks', 2))]

    # Build model once; heads will be added per dataset
    model = _build_model(cfg)

    # Logging & callbacks (unchanged)
    logger_cfg = getattr(cfg, 'logger', {})
    loggers, cb_list = build_loggers(
        logger_cfg.get('save_dir', cfg.experiment.output_dir),
        logger_cfg.get('name', cfg.experiment.name),
        use_wandb=logger_cfg.get('use_wandb', False),
        project=logger_cfg.get('project', None),
        entity=logger_cfg.get('entity', None),
        tags=logger_cfg.get('tags', None),
        group=logger_cfg.get('group', None),
        mode=logger_cfg.get('mode', None),
    )
    mem_cb = MemoryComputeLogger(os.path.join(cfg.experiment.output_dir, 'logs', 'memory.csv'))
    cb_list.append(mem_cb)

    nostalgia_cb = NostalgiaGradProjector(project_name_filter=["lora"])
    cb_list.append(nostalgia_cb)

    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        logger=loggers,
        callbacks=cb_list,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=getattr(cfg.train, 'num_sanity_val_steps', 0),
    )

    per_task_acc = []

    for idx, ds_name in enumerate(tasks, start=1):
        task_id = f"task{idx}_{ds_name.lower()}"
        print(f"=== Task {idx}/{len(tasks)}: {ds_name} ===")

        # Build loaders for this dataset (HF first, TV fallback)
        framework = cfg.model.get("framework", "hf")
        model_id = cfg.model.model_id if framework == 'hf' else None
        train_loader, val_loader, n_classes = get_loaders_and_num_classes(
            ds_name, cfg.dataset.root, model_id, cfg.dataset.image_size, cfg.dataset.batch_size, cfg.dataset.num_workers
        )

        # Add/switch head for this dataset (support expanding head if reusing same task id)
        if hasattr(model, 'add_task_head'):
            if hasattr(model, 'expand_task_head'):
                # Create if missing, else expand in case the new dataset has more classes
                model.add_task_head(task_id, n_classes)
                model.expand_task_head(task_id, n_classes)
            else:
                model.add_task_head(task_id, n_classes)
            model.set_active_task(task_id)

        # First task or before projection start: clear nostalgia basis
        if idx == 1 or not cfg.nostalgia.enabled or idx < cfg.nostalgia.projection_start_task:
            nostalgia_cb.clear_basis()

        # Fit on this dataset
        trainer.fit(model, train_loader, val_loader)

        # Record validation accuracy
        val_metrics = trainer.callback_metrics
        acc = float(val_metrics.get('val/acc', 0.0))
        per_task_acc.append(acc)

        # Update basis from current dataset on backbone only
        def loss_fn(batch):
            x, y = batch
            logits = model(x)
            return model.criterion(logits, y)

        try:
            backbone = model.backbone if hasattr(model, 'backbone') else model.model
            evals, evecs_flat = topk_eigs_with_eigenthings(
                backbone,
                lambda m, b: loss_fn(b),
                train_loader,
                num_eigenthings=cfg.nostalgia.top_k,
                use_gpu=True,
            )
            basis = build_param_basis(backbone, evecs_flat)
            basis = {k: v for k, v in basis.items() if 'head' not in k and 'heads' not in k}
            if cfg.nostalgia.project_params == 'lora-only':
                basis = {k: v for k, v in basis.items() if 'lora' in k}
            nostalgia_cb.set_basis(basis)
        except Exception as e:
            print(f"Warning: Hessian eigenthings failed ({e}); continuing without projection.")
            nostalgia_cb.clear_basis()

    # Save results and plots (unchanged)
    results_dir = os.path.join(cfg.experiment.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'per_task_acc.json'), 'w') as f:
        json.dump(per_task_acc, f)

    log_root = os.path.join(cfg.logger.save_dir, cfg.logger.name)
    metrics_csv = None
    try:
        latest_version = sorted(os.listdir(log_root))[-1]
        csv_dir = os.path.join(log_root, latest_version)
        maybe_csv = os.path.join(csv_dir, 'metrics.csv')
        if os.path.exists(maybe_csv):
            metrics_csv = maybe_csv
    except Exception:
        pass
    mem_csv = os.path.join(cfg.experiment.output_dir, 'logs', 'memory.csv')

    run_plots(results_dir, metrics_csv, mem_csv)


def main():
    parser = argparse.ArgumentParser(description='Run Nostalgia experiments and generate plots.')
    parser.add_argument('--config-name', type=str, default='config', help='Hydra config name')
    # Accept arbitrary Hydra overrides after --
    args, unknown = parser.parse_known_args()

    # Build a synthetic argv for Hydra run
    import sys
    argv = [sys.argv[0]] + [f'config_name={args.config_name}'] + unknown
    # Delegate to Hydra entry
    hydra_entry()


if __name__ == "__main__":
    main()
