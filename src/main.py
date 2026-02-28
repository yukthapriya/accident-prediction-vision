"""
Main entry point for the Real-Time Accident Prediction System
"""

import yaml
import logging
import argparse
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.data_ingestion import fetch_sensor_data, SensorDataCollector
from src.fusion import fuse_sensors
from src.hazard_detection import detect_hazards
from src.transformer_predictor import TransformerPredictor
from src.output_decider import decide_output
from src.gpu_utils import setup_gpu

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AccidentPredictionSystem:
    """Main system orchestrator"""
    
    def __init__(self, config_path="configs/config.yaml"):
        """Initialize the system"""
        self.config = self._load_config(config_path)
        self.device = setup_gpu(self.config)
        
        # Initialize components
        logger.info("Initializing components...")
        self.sensor_collector = SensorDataCollector(self.config)
        self.predictor = TransformerPredictor.load_from_checkpoint(
            self.config["model"]["checkpoint_path"],
            device=self.device
        )
        
        self.frame_count = 0
        self.start_time = time.time()
        
        logger.info("System initialization complete")
    
    @staticmethod
    def _load_config(config_path):
        """Load configuration from YAML file"""
        if not Path(config_path).exists():
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_single_inference(self, visualize=False):
        """Run a single inference cycle"""
        # Data collection
        cams, lidar, radar = self.sensor_collector.fetch_data()
        
        # Sensor fusion
        fused = fuse_sensors(cams, lidar, radar, method=self.config.get("fusion", {}).get("method", "early"))
        
        # Hazard detection
        hazards = detect_hazards(fused)
        
        # Collision prediction
        collision_prob = self.predictor.predict(hazards)
        
        # Decision making
        thresholds = self.config.get("decision_thresholds", {})
        decision = decide_output(collision_prob, thresholds)
        
        self.frame_count += 1
        
        # Logging
        if self.frame_count % 10 == 0:
            fps = self.frame_count / (time.time() - self.start_time)
            logger.info(
                f"Frame {self.frame_count} | "
                f"Collision Prob: {collision_prob:.2%} | "
                f"Decision: {decision} | "
                f"FPS: {fps:.1f}"
            )
        
        return {
            "collision_prob": collision_prob,
            "decision": decision,
            "hazards": hazards,
            "fused_data": fused
        }
    
    def run_continuous(self):
        """Run the system continuously"""
        logger.info("Starting continuous inference...")
        try:
            while True:
                result = self.run_single_inference()
                time.sleep(1.0 / self.config.get("system", {}).get("max_fps", 30))
        except KeyboardInterrupt:
            logger.info("System shutdown requested")
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info(f"System shutdown. Processed {self.frame_count} frames")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Real-Time Accident Prediction System")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--mode", type=str, choices=["single", "continuous"], default="continuous", help="Run mode")
    args = parser.parse_args()
    
    # Initialize system
    system = AccidentPredictionSystem(args.config)
    
    # Run
    if args.mode == "single":
        result = system.run_single_inference()
        logger.info(f"Result: {result}")
    else:
        system.run_continuous()


if __name__ == "__main__":
    main()