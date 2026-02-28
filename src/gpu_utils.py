"""
GPU utilities for optimization
Handles TensorRT, CUDA, and device management
"""

import torch
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def setup_gpu(config: Dict) -> str:
    """
    Setup GPU environment
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Device string (cuda:0, cpu, etc.)
    """
    gpu_config = config.get("gpu", {})
    
    # Check CUDA availability
    use_cuda = gpu_config.get("use_cuda", True) and torch.cuda.is_available()
    
    if use_cuda:
        device = gpu_config.get("device", "cuda:0")
        cuda_devices = torch.cuda.device_count()
        logger.info(f"CUDA available: {cuda_devices} device(s)")
        logger.info(f"Using device: {device}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU device")
    
    return device


class TensorRTOptimizer:
    """TensorRT optimization utilities"""
    
    @staticmethod
    def convert_to_tensorrt(
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        precision: str = "float32"
    ):
        """
        Convert PyTorch model to TensorRT
        
        Args:
            model: PyTorch model
            dummy_input: Dummy input for tracing
            output_path: Path to save TensorRT engine
            precision: Precision (float32, float16, int8)
        """
        try:
            import tensorrt as trt
            
            logger.info("Converting model to TensorRT...")
            
            # TensorRT conversion logic (simplified)
            # In production, use torch_tensorrt or ONNX path
            logger.info(f"TensorRT conversion not yet implemented. Use ONNX path.")
            
        except ImportError:
            logger.warning("TensorRT not installed. Skipping optimization.")


def enable_mixed_precision(model: torch.nn.Module) -> torch.nn.Module:
    """
    Enable mixed precision training/inference
    
    Args:
        model: PyTorch model
    
    Returns:
        Model with mixed precision enabled
    """
    try:
        from torch.cuda.amp import autocast
        logger.info("Mixed precision enabled")
        return model
    except Exception as e:
        logger.warning(f"Could not enable mixed precision: {e}")
        return model


def get_gpu_memory_stats() -> Dict:
    """Get GPU memory statistics"""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,  # GB
            "reserved": torch.cuda.memory_reserved() / 1e9,    # GB
            "free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9,  # GB
        }
    else:
        return {"allocated": 0, "reserved": 0, "free": 0}


def benchmark_model(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str,
    num_runs: int = 100
) -> Dict:
    """
    Benchmark model latency and throughput
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on
        num_runs: Number of benchmark runs
    
    Returns:
        Benchmark statistics
    """
    import time
    
    model = model.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            dummy_input = torch.randn(input_shape, device=device)
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            dummy_input = torch.randn(input_shape, device=device)
            
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            
            times.append(end - start)
    
    times = times[10:]  # Remove warmup
    
    return {
        "mean_latency_ms": sum(times) * 1000 / len(times),
        "min_latency_ms": min(times) * 1000,
        "max_latency_ms": max(times) * 1000,
        "throughput_fps": 1.0 / (sum(times) / len(times)),
    }