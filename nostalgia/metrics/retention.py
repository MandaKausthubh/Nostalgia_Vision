from typing import Dict, List
import numpy as np


def forgetting_curve(acc_per_task: List[float]) -> List[float]:
    """Compute forgetting measure E_t as drop from best-so-far.
    acc_per_task: accuracy on task i after finishing task t >= i (final accuracies per task in sequence)
    returns list of forgetting values per task.
    """
    best_so_far = np.maximum.accumulate(acc_per_task)
    return (best_so_far - np.array(acc_per_task)).tolist()


def average_accuracy(acc_matrix: np.ndarray) -> float:
    """Average accuracy across tasks (mean over tasks at the end)."""
    return float(np.mean(acc_matrix))
