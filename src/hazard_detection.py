"""
Hazard detection module
Detects objects and potential hazards in fused sensor data
"""

import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class HazardDetector:
    """Detects hazards from fused sensor data"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.hazard_classes = [
            "vehicle", "pedestrian", "cyclist",
            "obstacle", "pothole", "traffic_sign"
        ]
    
    def detect(self, fused_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect hazards in fused sensor data
        
        Args:
            fused_data: Fused sensor feature vector
        
        Returns:
            List of detected hazards with confidence scores
        """
        hazards = []
        
        # Placeholder: simple threshold-based detection
        data_energy = np.sum(np.abs(fused_data))
        
        if data_energy > np.percentile(np.abs(fused_data), 75):
            # Generate dummy detections
            num_hazards = np.random.randint(1, 5)
            for i in range(num_hazards):
                hazard = {
                    "type": np.random.choice(self.hazard_classes),
                    "confidence": np.random.uniform(0.5, 1.0),
                    "bbox": [
                        np.random.randint(0, 600),
                        np.random.randint(0, 400),
                        np.random.randint(640, 1280),
                        np.random.randint(400, 800)
                    ],
                    "position": {
                        "x": np.random.uniform(-50, 50),
                        "y": np.random.uniform(0, 100),
                        "z": np.random.uniform(-2, 2)
                    }
                }
                
                if hazard["confidence"] > self.confidence_threshold:
                    hazards.append(hazard)
        
        return hazards
    
    def filter_hazards(self, hazards: List[Dict], max_hazards: int = 10) -> List[Dict]:
        """
        Filter and rank hazards by confidence
        
        Args:
            hazards: List of detected hazards
            max_hazards: Maximum number of hazards to keep
        
        Returns:
            Filtered and ranked hazards
        """
        # Sort by confidence
        hazards = sorted(hazards, key=lambda x: x["confidence"], reverse=True)
        
        # Keep top K
        return hazards[:max_hazards]


def detect_hazards(
    fused_data: np.ndarray,
    confidence_threshold: float = 0.5,
    max_hazards: int = 10
) -> List[Dict[str, Any]]:
    """
    Detect hazards in fused sensor data
    
    Args:
        fused_data: Fused sensor feature vector
        confidence_threshold: Confidence threshold for detections
        max_hazards: Maximum number of hazards to return
    
    Returns:
        List of detected hazards
    """
    detector = HazardDetector(confidence_threshold=confidence_threshold)
    hazards = detector.detect(fused_data)
    hazards = detector.filter_hazards(hazards, max_hazards=max_hazards)
    
    return hazards