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

This project prefers Hugging Face Datasets when available, with torchvision/ImageFolder fallbacks.

- HF auto-downloads to `~/.cache/huggingface` (set `HF_HOME` to change).
- Torchvision auto-downloads to `dataset.root` where supported.
- ImageFolder fallback expects:
  - `${root}/train/class_x/*` and `${root}/val/class_x/*`

Common dataset keys used in sequences:

- cifar10, cifar100 (HF + torchvision)
- tinyimagenet (HF: `Maysee/tiny-imagenet`, else ImageFolder)
- caltech256 (HF and torchvision)
- cub200 (HF: `caltech_birds2011`)
- flowers102 (HF: `oxford_flowers102`)
- imagenet100, imagenet-r, imagenet-a (HF may vary; ImageFolder fallback recommended)

Configure locations via Hydra, e.g. `dataset.root=/data`.

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

## Experiment suites (multi-dataset sequences)

Set the ordered list of datasets via `experiment.tasks=[...]`. Each entry becomes one training task with its own classifier head; the backbone is shared and nostalgia projection starts from task 2 (configurable).

- Main: CIFAR-100 → TinyImageNet → CalTech-256

```bash
python -m nostalgia.train experiment.name=main_seq \
  experiment.tasks=[cifar100,tinyimagenet,caltech256] \
  model=vit dataset.root=/data
```

- Robustness: ImageNet-100 → ImageNet-R → ImageNet-A

```bash
# HF-first; otherwise point to ImageFolder trees under /data/robust/<dataset>/{train,val}
python -m nostalgia.train experiment.name=robust_seq \
  experiment.tasks=[imagenet100,imagenet-r,imagenet-a] \
  model=vit dataset.root=/data/robust
```

- Fine-grained: CIFAR-100 → CUB-200 → Flowers-102

```bash
python -m nostalgia.train experiment.name=fine_seq \
  experiment.tasks=[cifar100,cub200,flowers102] \
  model=vit dataset.root=/data
```

- Ablations (toy): CIFAR-10 → MNIST

```bash
python -m nostalgia.train experiment.name=abl_seq \
  experiment.tasks=[cifar10,mnist] \
  model=vit dataset.root=/data
```

Notes:

- You can swap models (e.g., `model.framework=torchvision model.arch=resnet50`).
- Each task runs `train.epochs` (default 50). Adjust with `train.epochs=<N>`.
- When HF lacks a dataset, the loader falls back to torchvision or ImageFolder.

## Live logging (TensorBoard and W&B)

TensorBoard is on by default and logs under `${experiment.output_dir}`.

```bash
# After starting a run, in another terminal
tensorboard --logdir runs
```

Enable Weights & Biases for live dashboards:

```bash
python -m nostalgia.train experiment.tasks=[cifar100,tinyimagenet,caltech256] model=vit \
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

- Per-task heads are created automatically and named with dataset: `task1_<ds>`, `task2_<ds>`, ...
- First task: no projection; subsequent tasks: gradients are projected onto the complement of top-k Hessian eigendirections of the backbone.
- Projection scope can be limited to LoRA parameters (set in config via `nostalgia.project_params`).

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
