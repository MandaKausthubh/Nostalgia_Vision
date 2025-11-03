import os
import sys
import pathlib

# Ensure project root on sys.path when running tests directly
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from nostalgia.data_utils.datasets import get_cifar100


def test_cifar100_dataloaders_fast(tmp_path, capsys):
    root = os.path.join(tmp_path, 'cifar')
    model_id = 'google/vit-base-patch16-224'
    train_loader, val_loader = get_cifar100(root=root, model_id=model_id, image_size=64, batch_size=2, num_workers=0)
    xb, yb = next(iter(train_loader))
    print(f"cifar100: train_batch={tuple(xb.shape)}, val_len={len(val_loader)}")
    captured = capsys.readouterr()
    assert "cifar100:" in captured.out
    assert xb.shape[0] == 2
    assert xb.ndim == 4  # NCHW
    assert yb.shape[0] == 2
