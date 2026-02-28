"""
Multi-sensor data ingestion module
Handles camera, LiDAR, and radar data streams
"""

import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class SensorDataCollector:
    """Collects data from all sensors"""
    
    def __init__(self, config):
        self.config = config
        self.camera_source = config.get("sensors", {}).get("camera", {}).get("source", 0)
        self.cap = None
        
        if self.config.get("sensors", {}).get("camera", {}).get("enabled", True):
            self._init_camera()
    
    def _init_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            if not self.cap.isOpened():
                logger.warning("Failed to open camera")
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
    
    def fetch_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fetch current sensor data"""
        cams = self._get_camera_frame()
        lidar = self._get_lidar_data()
        radar = self._get_radar_data()
        return cams, lidar, radar
    
    def _get_camera_frame(self) -> np.ndarray:
        """Get camera frame"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize to configured resolution
                resolution = self.config.get("sensors", {}).get("camera", {}).get("resolution", [640, 480])
                frame = cv2.resize(frame, tuple(resolution))
                return frame
        
        # Return dummy frame if camera unavailable
        resolution = self.config.get("sensors", {}).get("camera", {}).get("resolution", [640, 480])
        return np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    
    def _get_lidar_data(self) -> np.ndarray:
        """Get LiDAR point cloud data"""
        # Placeholder: return dummy LiDAR data
        # In production, integrate with actual LiDAR device
        num_points = self.config.get("sensors", {}).get("lidar", {}).get("points_per_frame", 100)
        lidar_range = self.config.get("sensors", {}).get("lidar", {}).get("range", 100)
        
        # Generate dummy point cloud (x, y, z coordinates)
        lidar_data = np.random.uniform(-lidar_range, lidar_range, (num_points, 3)).astype(np.float32)
        return lidar_data
    
    def _get_radar_data(self) -> np.ndarray:
        """Get radar data"""
        # Placeholder: return dummy radar data
        # In production, integrate with actual radar device
        num_targets = self.config.get("sensors", {}).get("radar", {}).get("num_targets", 10)
        radar_range = self.config.get("sensors", {}).get("radar", {}).get("range", 200)
        
        # Generate dummy radar data (x, y, velocity)
        radar_data = np.random.uniform(-radar_range, radar_range, (num_targets, 3)).astype(np.float32)
        return radar_data
    
    def __del__(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()


def fetch_sensor_data(config=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience function to fetch sensor data"""
    if config is None:
        config = {}
    
    collector = SensorDataCollector(config)
    return collector.fetch_data()