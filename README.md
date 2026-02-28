# 🚗 Real‑Time Vision System for Accident Prediction Using Multi‑Sensor Fusion

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/yukthapriya/accident-prediction-vision?style=social)](https://github.com/yukthapriya/accident-prediction-vision)

A **state-of-the-art real-time multi-sensor fusion system** for autonomous accident prediction. Ingests camera, LiDAR, and radar data simultaneously, detects hazards in real-time, predicts collision probability using transformer-based temporal models, and provides decision-ready outputs (STOP, SLOW, REROUTE).

**Perfect for**: Tesla Autopilot, NVIDIA autonomous vehicles, robotaxis, and edge deployment.

---

## 🎯 Key Features

| Feature | Details |
|---------|---------|
| **🔀 Multi-Sensor Fusion** | Combines camera (RGB), LiDAR (3D point clouds), and radar (velocity) streams |
| **⚡ Real-Time Hazard Detection** | Deep learning-based object detection and classification at 14+ FPS |
| **🤖 Transformer-Based Prediction** | Temporal attention model for collision probability (0-1) |
| **🎬 GPU-Optimized** | TensorRT and CUDA kernel support for NVIDIA edge deployment |
| **🚦 Decision-Ready Outputs** | Autonomy-level action recommendations (STOP/SLOW/REROUTE/PROCEED) |
| **📦 Production-Grade** | Docker containerization, GitHub Actions CI/CD, comprehensive testing |
| **⚙️ Configurable** | YAML-based configuration for thresholds, sensor settings, and fusion methods |

---

## 📊 System Overview

### Architecture Diagram
