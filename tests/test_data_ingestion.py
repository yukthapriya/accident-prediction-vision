"""Tests for data ingestion module"""

import pytest
import numpy as np
from src.data_ingestion import SensorDataCollector


def test_sensor_collector_initialization():
    """Test sensor collector initialization"""
    config = {
        "sensors": {
            "camera": {"enabled": True, "resolution": [640, 480]},
            "lidar": {"enabled": True, "points_per_frame": 100},
            "radar": {"enabled": True, "num_targets": 10}
        }
    }
    
    collector = SensorDataCollector(config)
    assert collector is not None


def test_fetch_data_shapes():
    """Test that fetched data has correct shapes"""
    config = {
        "sensors": {
            "camera": {"enabled": True, "resolution": [640, 480]},
            "lidar": {"enabled": True, "points_per_frame": 100},
            "radar": {"enabled": True, "num_targets": 10}
        }
    }
    
    collector = SensorDataCollector(config)
    cams, lidar, radar = collector.fetch_data()
    
    assert cams.shape == (480, 640, 3)
    assert lidar.shape == (100, 3)
    assert radar.shape == (10, 3)


def test_lidar_range():
    """Test that LiDAR points are within range"""
    config = {
        "sensors": {
            "lidar": {"points_per_frame": 100, "range": 100}
        }
    }
    
    collector = SensorDataCollector(config)
    _, lidar, _ = collector.fetch_data()
    
    assert np.all(np.abs(lidar) <= 100)