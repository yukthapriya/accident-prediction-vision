"""Tests for sensor fusion module"""

import pytest
import numpy as np
from src.fusion import fuse_sensors, SensorFusion


@pytest.fixture
def sample_sensor_data():
    """Create sample sensor data"""
    camera = np.random.randn(480, 640, 3).astype(np.float32)
    lidar = np.random.randn(100, 3).astype(np.float32)
    radar = np.random.randn(10, 3).astype(np.float32)
    return camera, lidar, radar


def test_early_fusion(sample_sensor_data):
    """Test early fusion"""
    camera, lidar, radar = sample_sensor_data
    fused = SensorFusion.early_fusion(camera, lidar, radar)
    assert fused is not None
    assert isinstance(fused, np.ndarray)


def test_late_fusion(sample_sensor_data):
    """Test late fusion"""
    camera, lidar, radar = sample_sensor_data
    fused = SensorFusion.late_fusion(camera, lidar, radar)
    assert fused is not None
    assert isinstance(fused, np.ndarray)


def test_deep_fusion(sample_sensor_data):
    """Test deep fusion"""
    camera, lidar, radar = sample_sensor_data
    fused = SensorFusion.deep_fusion(camera, lidar, radar)
    assert fused is not None
    assert isinstance(fused, np.ndarray)


def test_fuse_sensors_methods(sample_sensor_data):
    """Test fuse_sensors with different methods"""
    camera, lidar, radar = sample_sensor_data
    
    for method in ["early", "late", "deep"]:
        fused = fuse_sensors(camera, lidar, radar, method=method)
        assert fused is not None