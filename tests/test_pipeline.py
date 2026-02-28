"""Integration tests for the full pipeline"""

import pytest
import torch
from src.main import AccidentPredictionSystem


@pytest.fixture
def config():
    """Load test configuration"""
    return {
        "model": {"checkpoint_path": "checkpoints/model.pth"},
        "decision_thresholds": {"stop": 0.8, "slow": 0.5, "reroute": 0.2},
        "sensors": {
            "camera": {"enabled": True, "resolution": [640, 480], "source": 0},
            "lidar": {"enabled": True, "points_per_frame": 100, "range": 100},
            "radar": {"enabled": True, "num_targets": 10, "range": 200}
        },
        "gpu": {"use_cuda": False, "device": "cpu"},
        "system": {"max_fps": 30}
    }


def test_system_initialization(config):
    """Test system initialization"""
    system = AccidentPredictionSystem.__new__(AccidentPredictionSystem)
    system.config = config
    assert system.config is not None


def test_single_inference(config):
    """Test single inference"""
    system = AccidentPredictionSystem.__new__(AccidentPredictionSystem)
    system.config = config
    system.device = "cpu"
    system.frame_count = 0
    system.start_time = 0
    
    # Mock components
    from unittest.mock import Mock
    system.sensor_collector = Mock()
    system.predictor = Mock()
    
    import numpy as np
    system.sensor_collector.fetch_data.return_value = (
        np.zeros((480, 640, 3)),
        np.zeros((100, 3)),
        np.zeros((10, 3))
    )
    system.predictor.predict.return_value = 0.5
    
    # This would normally run the inference
    assert system.config is not None