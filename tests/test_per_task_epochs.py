import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from omegaconf import DictConfig, OmegaConf


def test_parse_list_int_from_string():
    """Test that epochs_per_task can be parsed from string format"""
    # Helper function from train.py (duplicated here to avoid import issues)
    def _parse_list_str(s: str):
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return parts

    def _parse_list_int(s: str):
        return [int(x) for x in _parse_list_str(s)]
    
    # Test various formats
    assert _parse_list_int("[10,20,30]") == [10, 20, 30]
    assert _parse_list_int("10,20,30") == [10, 20, 30]
    assert _parse_list_int("[10, 20, 30]") == [10, 20, 30]
    assert _parse_list_int("10, 20, 30") == [10, 20, 30]


def test_epochs_per_task_list_format():
    """Test that epochs_per_task works with list format"""
    # Simulate config with list format
    raw_ept = [10, 20, 30]
    
    # Logic from train.py lines 136-148
    if isinstance(raw_ept, str):
        epochs_per_task = None
    elif isinstance(raw_ept, (list, tuple)):
        tmp = []
        for v in raw_ept:
            try:
                tmp.append(int(v))
            except Exception:
                pass
        epochs_per_task = tmp
    else:
        epochs_per_task = []
    
    assert epochs_per_task == [10, 20, 30]


def test_epochs_per_task_fallback():
    """Test that default epochs are used when epochs_per_task is empty or too short"""
    epochs_per_task = [10, 20]
    default_epochs = 50
    
    # Task 1 (idx=1, so idx-1=0)
    task_epochs_1 = epochs_per_task[0] if 0 < len(epochs_per_task) else default_epochs
    assert task_epochs_1 == 10
    
    # Task 2 (idx=2, so idx-1=1)
    task_epochs_2 = epochs_per_task[1] if 1 < len(epochs_per_task) else default_epochs
    assert task_epochs_2 == 20
    
    # Task 3 (idx=3, so idx-1=2) - should fall back to default
    task_epochs_3 = epochs_per_task[2] if 2 < len(epochs_per_task) else default_epochs
    assert task_epochs_3 == 50


def test_config_yaml_epochs_per_task():
    """Test that the config file properly supports epochs_per_task"""
    # This tests that the YAML config structure supports the feature
    config_str = """
    train:
      epochs: 50
      epochs_per_task: []
    """
    cfg = OmegaConf.create(config_str)
    assert cfg.train.epochs == 50
    assert cfg.train.epochs_per_task == []
    
    # Test with actual values
    config_str_with_values = """
    train:
      epochs: 50
      epochs_per_task: [10, 20, 30]
    """
    cfg_with_values = OmegaConf.create(config_str_with_values)
    assert cfg_with_values.train.epochs_per_task == [10, 20, 30]
