import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import pytest
import torch
from nostalgia.models.hf_model import LitHFClassifier


@pytest.mark.timeout(60)
def test_hf_model_forward_fast(capsys):
    # Allow running offline by skipping if HF cache is missing and no internet
    try:
        model = LitHFClassifier(model_id='google/vit-base-patch16-224', num_labels=100, lr=1e-4, weight_decay=0.0, use_lora=False)
    except Exception as e:
        pytest.skip(f"Skipping HF model download test due to: {e}")
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    logits = out.logits
    print(f"logits shape={tuple(logits.shape)}, mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
    captured = capsys.readouterr()
    assert "logits shape=" in captured.out
    assert logits.shape == (1, 100)
