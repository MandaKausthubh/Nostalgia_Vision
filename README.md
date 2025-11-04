# Nostalgia_Vision

Lightweight experimental framework for the Nostalgia gradient-projection method (CVPR submission context). It uses Hydra + PyTorch Lightning, integrates Hessian eigenspaces (hessian-eigenthings), and supports Hugging Face and Torchvision backbones with optional LoRA adapters (PEFT).

- Config/Orchestration: Hydra (configs under `nostalgia/configs`)
- Training: PyTorch Lightning (Trainer, callbacks, loggers)
- Models: ViT/DeiT (HF) and ResNet-18/50/101 (torchvision) with per-task classifier heads
- Continual learning: nostalgia gradient projection on the backbone; first task trains without projection
- Logging: TensorBoard, CSV, optional Weights & Biases
- Plots: figures saved after training (accuracy vs task, forgetting, avg accuracy, loss, memory)

## Setup

Requirements: Python 3.10+, CUDA recommended.

```bash
# Create/activate your env, then
pip install -r requirements.txt
pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings
# Optional: for W&B
wandb login
```

## Datasets

- CIFAR-10/100 and CalTech-256 auto-download to `dataset.root`.
- ImageNet-1K expects an ImageFolder:
  - `${root}/train/class_x/*.JPEG`, `${root}/val/class_x/*.JPEG`
- TinyImageNet, ImageNet-100, ImageNet-R, ImageNet-A: also ImageFolder with `train/` and `val/` splits.

Configure locations via Hydra, e.g. `dataset.root=/data/imagenet`.

## Quickstart

Run ViT on CIFAR-100 with 2 sequential tasks (same dataset) and default 50 epochs/task:

```bash
python -m nostalgia.train dataset=cifar100 model=vit experiment.num_tasks=2
```

- Override epochs or batch size:

```bash
python -m nostalgia.train dataset=cifar100 model=vit experiment.num_tasks=2 \
  train.epochs=20 dataset.batch_size=128
```

Use a Torchvision backbone (ResNet-18):

```bash
python -m nostalgia.train dataset=cifar100 model.framework=torchvision model.arch=resnet18
```

Use ImageNet-1K (expects ImageFolder at `dataset.root`):

```bash
python -m nostalgia.train dataset=imagenet1k dataset.root=/data/imagenet \
  model=vit train.epochs=30
```

## Live logging (TensorBoard and W&B)

TensorBoard is on by default and logs under `${experiment.output_dir}`.

```bash
# After starting a run, in another terminal
tensorboard --logdir runs
```

Enable Weights & Biases for live dashboards:

```bash
python -m nostalgia.train dataset=cifar100 model=vit experiment.num_tasks=2 \
  logger.use_wandb=true logger.project=nostalgia logger.entity=YOUR_ENTITY
```

Notes:

- TensorBoard + CSV always run; W&B is optional and can be toggled via Hydra flags.

## Outputs

- Hydra run dir: `${experiment.output_dir}/YYYY-MM-DD_HH-MM-SS/`
- Checkpoints: `${logger.save_dir}/${logger.name}/checkpoints/`
- Metrics CSV: `${logger.save_dir}/${logger.name}/<version>/metrics.csv`
- Results JSON: `${experiment.output_dir}/results/per_task_acc.json`
- Figures: `${experiment.output_dir}/results/figures/*.png`

Figures are generated automatically at the end of training. To re-generate:

```bash
python -m nostalgia.scripts.plot_all --results_dir <path> \
  --log_csv <metrics.csv> --mem_csv <memory.csv>
```

## Configuration

Hydra config groups live under `nostalgia/configs`:

- `dataset/`: `cifar100.yaml`, `caltech256.yaml`, `imagenet1k.yaml`
- `model/`: `vit.yaml`, `deit_base_distilled.yaml`, `resnet18.yaml`, `resnet50.yaml`, `resnet101.yaml`
- `nostalgia/`: `default.yaml` (projection settings)
- `train/`: `default.yaml` (epochs, lr, precision, devices, logging cadence)
- `logger/`: `tensorboard.yaml` (also carries W&B flags)

Print the composed config at runtime to verify overrides.

## Continual Learning

- Per-task heads are created automatically: `task1`, `task2`, ...
- First task: no projection; subsequent tasks: gradients are projected onto the complement of top-k Hessian eigendirections of the backbone.
- Projection scope can be limited to LoRA parameters (set in config).

Current release trains multiple tasks on the same dataset; multi-dataset task sequences are being staged (loaders for CIFAR-10, TinyImageNet, ImageNet-100/R/A are included).

## Tests

```bash
pytest -q
```

## Tips

- Change epochs: `train.epochs=<N>` (applies per task)
- Mixed precision: `train.precision=16` (default)
- Devices: `train.accelerator=gpu train.devices=1`
- Reproducibility: `experiment.seed=<int>`

---

For issues or requests, open an issue in this repo.
