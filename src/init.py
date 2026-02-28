"""
Real-Time Vision System for Accident Prediction
"""

__version__ = "0.1.0"
__author__ = "yukthapriya"

from src.data_ingestion import fetch_sensor_data
from src.fusion import fuse_sensors
from src.hazard_detection import detect_hazards
from src.transformer_predictor import TransformerPredictor
from src.output_decider import decide_output

__all__ = [
    "fetch_sensor_data",
    "fuse_sensors",
    "detect_hazards",
    "TransformerPredictor",
    "decide_output",
]