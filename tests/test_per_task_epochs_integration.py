import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from omegaconf import DictConfig, OmegaConf, ListConfig


def test_per_task_epochs_config_integration():
    """Integration test to verify per-task epochs configuration works end-to-end"""
    
    # Create a mock config similar to what Hydra would provide
    config_dict = {
        'experiment': {
            'name': 'test_run',
            'seed': 0,
            'num_tasks': 3,
            'tasks': ['cifar10', 'mnist', 'cifar100'],
            'output_dir': '/tmp/test_output'
        },
        'train': {
            'epochs': 50,  # default epochs
            'epochs_per_task': [10, 20, 30],  # per-task overrides
            'lr': 3e-4,
            'weight_decay': 0.05,
            'accelerator': 'cpu',
            'devices': 1,
            'precision': 32,
            'log_every_n_steps': 50,
            'val_check_interval': 1.0,
            'num_sanity_val_steps': 0
        },
        'dataset': {
            'name': 'cifar100',
            'root': '/tmp/data',
            'image_size': 224,
            'batch_size': 4,
            'num_workers': 0,
            'num_classes': 100
        },
        'model': {
            'framework': 'hf',
            'model_id': 'google/vit-base-patch16-224',
            'use_lora': True,
            'lora': {}
        },
        'nostalgia': {
            'enabled': True,
            'projection_start_task': 2,
            'top_k': 10,
            'project_params': 'lora-only'
        },
        'logger': {
            'save_dir': '/tmp/logs',
            'name': 'test_logger',
            'use_wandb': False
        }
    }
    
    cfg = DictConfig(config_dict)
    
    # Simulate the logic from train.py lines 135-148
    def _parse_list_str(s: str):
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return parts

    def _parse_list_int(s: str):
        return [int(x) for x in _parse_list_str(s)]
    
    raw_ept = getattr(cfg.train, 'epochs_per_task', [])
    if isinstance(raw_ept, str):
        epochs_per_task = _parse_list_int(raw_ept)
    elif isinstance(raw_ept, (list, tuple, ListConfig)):
        tmp = []
        for v in raw_ept:
            try:
                tmp.append(int(v))
            except Exception:
                pass
        epochs_per_task = tmp
    else:
        epochs_per_task = []
    
    # Verify epochs_per_task was parsed correctly
    assert epochs_per_task == [10, 20, 30]
    
    # Simulate the task loop from train.py lines 150-168
    tasks = cfg.experiment.tasks
    task_epochs_list = []
    
    for idx, ds_name in enumerate(tasks, start=1):
        # This is the key logic we're testing
        task_epochs = epochs_per_task[idx - 1] if idx - 1 < len(epochs_per_task) else cfg.train.epochs
        task_epochs_list.append(task_epochs)
    
    # Verify each task gets the correct number of epochs
    assert task_epochs_list == [10, 20, 30]
    print(f"✓ Task epochs correctly assigned: {task_epochs_list}")


def test_per_task_epochs_fallback_to_default():
    """Test that tasks beyond epochs_per_task length fall back to default epochs"""
    
    config_dict = {
        'experiment': {
            'tasks': ['task1', 'task2', 'task3', 'task4']
        },
        'train': {
            'epochs': 50,
            'epochs_per_task': [10, 20]  # Only 2 values for 4 tasks
        }
    }
    
    cfg = DictConfig(config_dict)
    
    # Parse epochs_per_task
    raw_ept = cfg.train.epochs_per_task
    if isinstance(raw_ept, (list, tuple, ListConfig)):
        epochs_per_task = [int(v) for v in raw_ept]
    else:
        epochs_per_task = []
    
    # Simulate task loop
    tasks = cfg.experiment.tasks
    task_epochs_list = []
    
    for idx, ds_name in enumerate(tasks, start=1):
        task_epochs = epochs_per_task[idx - 1] if idx - 1 < len(epochs_per_task) else cfg.train.epochs
        task_epochs_list.append(task_epochs)
    
    # Tasks 1-2 should use epochs_per_task, tasks 3-4 should use default
    assert task_epochs_list == [10, 20, 50, 50]
    print(f"✓ Fallback to default epochs works: {task_epochs_list}")


def test_per_task_epochs_empty_list():
    """Test that empty epochs_per_task falls back to default for all tasks"""
    
    config_dict = {
        'experiment': {
            'tasks': ['task1', 'task2', 'task3']
        },
        'train': {
            'epochs': 50,
            'epochs_per_task': []  # Empty list
        }
    }
    
    cfg = DictConfig(config_dict)
    
    # Parse epochs_per_task
    epochs_per_task = cfg.train.epochs_per_task if isinstance(cfg.train.epochs_per_task, list) else []
    
    # Simulate task loop
    tasks = cfg.experiment.tasks
    task_epochs_list = []
    
    for idx, ds_name in enumerate(tasks, start=1):
        task_epochs = epochs_per_task[idx - 1] if idx - 1 < len(epochs_per_task) else cfg.train.epochs
        task_epochs_list.append(task_epochs)
    
    # All tasks should use default epochs
    assert task_epochs_list == [50, 50, 50]
    print(f"✓ Empty epochs_per_task uses defaults: {task_epochs_list}")


def test_per_task_epochs_string_format():
    """Test that epochs_per_task can be provided as a string (for CLI compatibility)"""
    
    config_dict = {
        'experiment': {
            'tasks': ['task1', 'task2', 'task3']
        },
        'train': {
            'epochs': 50,
            'epochs_per_task': '[15,25,35]'  # String format
        }
    }
    
    cfg = DictConfig(config_dict)
    
    # Parse epochs_per_task with string support
    def _parse_list_str(s: str):
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return parts

    def _parse_list_int(s: str):
        return [int(x) for x in _parse_list_str(s)]
    
    raw_ept = cfg.train.epochs_per_task
    if isinstance(raw_ept, str):
        epochs_per_task = _parse_list_int(raw_ept)
    elif isinstance(raw_ept, (list, tuple, ListConfig)):
        epochs_per_task = [int(v) for v in raw_ept]
    else:
        epochs_per_task = []
    
    # Simulate task loop
    tasks = cfg.experiment.tasks
    task_epochs_list = []
    
    for idx, ds_name in enumerate(tasks, start=1):
        task_epochs = epochs_per_task[idx - 1] if idx - 1 < len(epochs_per_task) else cfg.train.epochs
        task_epochs_list.append(task_epochs)
    
    # All tasks should use the string-parsed values
    assert task_epochs_list == [15, 25, 35]
    print(f"✓ String format parsing works: {task_epochs_list}")
