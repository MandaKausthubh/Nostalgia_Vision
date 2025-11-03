import os
import csv
import time
import torch
import resource
from pytorch_lightning.callbacks import Callback


class MemoryComputeLogger(Callback):
    def __init__(self, out_csv: str):
        super().__init__()
        self.out_csv = out_csv
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(self.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['time', 'step', 'gpu_mem_mb', 'cpu_mem_mb'])
            w.writeheader()

    def _cpu_mem_mb(self) -> float:
        # ru_maxrss is KB on macOS
        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage_kb / 1024.0

    def _gpu_mem_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        return 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        row = {
            'time': time.time(),
            'step': step,
            'gpu_mem_mb': self._gpu_mem_mb(),
            'cpu_mem_mb': self._cpu_mem_mb(),
        }
        with open(self.out_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['time', 'step', 'gpu_mem_mb', 'cpu_mem_mb'])
            w.writerow(row)
