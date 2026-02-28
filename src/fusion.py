"""
Multi-sensor fusion module
Combines camera, LiDAR, and radar data
"""

import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)


class SensorFusion:
    """Handles multi-sensor fusion"""
    
    @staticmethod
    def early_fusion(camera: np.ndarray, lidar: np.ndarray, radar: np.ndarray) -> np.ndarray:
        """
        Early fusion: concatenate all sensor data before processing
        
        Args:
            camera: Camera frame (H, W, C)
            lidar: LiDAR point cloud (N, 3)
            radar: Radar data (M, 3)
        
        Returns:
            Fused feature vector
        """
        # Flatten camera frame
        camera_flat = camera.flatten().astype(np.float32)
        
        # Flatten LiDAR and radar
        lidar_flat = lidar.flatten().astype(np.float32)
        radar_flat = radar.flatten().astype(np.float32)
        
        # Concatenate
        fused = np.concatenate([camera_flat, lidar_flat, radar_flat])
        
        # Normalize
        fused = (fused - fused.mean()) / (fused.std() + 1e-8)
        
        return fused
    
    @staticmethod
    def late_fusion(camera: np.ndarray, lidar: np.ndarray, radar: np.ndarray) -> np.ndarray:
        """
        Late fusion: process each sensor separately, then combine
        
        Args:
            camera: Camera frame (H, W, C)
            lidar: LiDAR point cloud (N, 3)
            radar: Radar data (M, 3)
        
        Returns:
            Fused feature vector
        """
        # Process each modality
        camera_features = SensorFusion._process_camera(camera)
        lidar_features = SensorFusion._process_lidar(lidar)
        radar_features = SensorFusion._process_radar(radar)
        
        # Concatenate
        fused = np.concatenate([camera_features, lidar_features, radar_features])
        
        return fused
    
    @staticmethod
    def deep_fusion(camera: np.ndarray, lidar: np.ndarray, radar: np.ndarray) -> np.ndarray:
        """
        Deep fusion: iterative feature interaction
        
        Args:
            camera: Camera frame (H, W, C)
            lidar: LiDAR point cloud (N, 3)
            radar: Radar data (M, 3)
        
        Returns:
            Fused feature vector
        """
        # Extract modality-specific features
        camera_features = SensorFusion._process_camera(camera)
        lidar_features = SensorFusion._process_lidar(lidar)
        radar_features = SensorFusion._process_radar(radar)
        
        # Cross-modal attention (simplified)
        camera_features = camera_features * (1 + lidar_features[:len(camera_features)])
        lidar_features = lidar_features * (1 + radar_features[:len(lidar_features)])
        
        # Concatenate
        fused = np.concatenate([camera_features, lidar_features, radar_features])
        
        return fused
    
    @staticmethod
    def _process_camera(camera: np.ndarray) -> np.ndarray:
        """Process camera data"""
        # Simple processing: flatten and normalize
        features = camera.flatten().astype(np.float32)
        features = (features - features.mean()) / (features.std() + 1e-8)
        return features
    
    @staticmethod
    def _process_lidar(lidar: np.ndarray) -> np.ndarray:
        """Process LiDAR data"""
        # Compute statistics: min, max, mean, std per dimension
        features = np.concatenate([
            lidar.min(axis=0),
            lidar.max(axis=0),
            lidar.mean(axis=0),
            lidar.std(axis=0)
        ]).astype(np.float32)
        
        features = (features - features.mean()) / (features.std() + 1e-8)
        return features
    
    @staticmethod
    def _process_radar(radar: np.ndarray) -> np.ndarray:
        """Process radar data"""
        # Compute statistics: min, max, mean, std per dimension
        features = np.concatenate([
            radar.min(axis=0),
            radar.max(axis=0),
            radar.mean(axis=0),
            radar.std(axis=0)
        ]).astype(np.float32)
        
        features = (features - features.mean()) / (features.std() + 1e-8)
        return features


def fuse_sensors(
    camera: np.ndarray,
    lidar: np.ndarray,
    radar: np.ndarray,
    method: str = "early"
) -> np.ndarray:
    """
    Fuse multi-sensor data
    
    Args:
        camera: Camera frame
        lidar: LiDAR point cloud
        radar: Radar data
        method: Fusion method (early, late, deep)
    
    Returns:
        Fused feature vector
    """
    if method == "late":
        return SensorFusion.late_fusion(camera, lidar, radar)
    elif method == "deep":
        return SensorFusion.deep_fusion(camera, lidar, radar)
    else:  # early
        return SensorFusion.early_fusion(camera, lidar, radar)