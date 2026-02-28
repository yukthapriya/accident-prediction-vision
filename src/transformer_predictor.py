"""
Transformer-based collision probability predictor
Uses temporal attention to model hazard sequences
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head attention"""
    
    def __init__(self, hidden_size: int, num_heads: int, ff_size: int = 2048):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, hidden_size)
        )
        
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        
        return x


class CollisionPredictor(nn.Module):
    """Transformer-based collision probability predictor"""
    
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        temporal_window: int = 5
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.temporal_window = temporal_window
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Temporal embedding
        self.temporal_embed = nn.Embedding(temporal_window, hidden_size)
        
        # Transformer stack
        self.transformer = nn.Sequential(
            *[TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            Collision probability (batch_size, 1)
        """
        batch_size = x.shape[0]
        seq_len = min(x.shape[1] if len(x.shape) > 1 else 1, self.temporal_window)
        
        # Project input
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.input_proj(x)  # (batch, seq, hidden)
        
        # Add temporal embedding
        positions = torch.arange(x.shape[1], device=x.device)
        temporal_emb = self.temporal_embed(positions % self.temporal_window)
        x = x + temporal_emb.unsqueeze(0)
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, hidden)
        
        # Output
        prob = self.head(x)
        
        return prob


class TransformerPredictor:
    """Wrapper for collision prediction with checkpointing"""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.hazard_history = []
        self.max_history = 5
    
    @staticmethod
    def load_from_checkpoint(
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs
    ) -> "TransformerPredictor":
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to load on
            **kwargs: Additional model arguments
        
        Returns:
            TransformerPredictor instance
        """
        # Default configuration
        config = {
            "input_size": 512,
            "hidden_size": 256,
            "num_heads": 8,
            "num_layers": 3,
            "temporal_window": 5
        }
        config.update(kwargs)
        
        # Create model
        model = CollisionPredictor(**config)
        
        # Load checkpoint if it exists
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Using random initialization.")
        else:
            logger.info(f"Checkpoint not found at {checkpoint_path}. Using random initialization.")
        
        return TransformerPredictor(model, device)
    
    def predict(self, hazards: List[Dict]) -> float:
        """
        Predict collision probability
        
        Args:
            hazards: List of detected hazards
        
        Returns:
            Collision probability (0-1)
        """
        # Update hazard history
        if hazards:
            self.hazard_history.append(len(hazards))
        else:
            self.hazard_history.append(0)
        
        self.hazard_history = self.hazard_history[-self.max_history:]
        
        # Create input tensor
        # Simple feature: number of hazards + hazard scores
        if hazards:
            scores = np.array([h.get("confidence", 0.5) for h in hazards])
            features = np.array([
                len(hazards),
                scores.mean(),
                scores.max(),
                len(self.hazard_history),
                np.mean(self.hazard_history),
            ])
        else:
            features = np.zeros(5)
        
        # Pad to model input size
        input_size = self.model.input_proj.in_features
        if len(features) < input_size:
            features = np.pad(features, (0, input_size - len(features)))
        else:
            features = features[:input_size]
        
        # Forward pass
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(self.device)
            prob = self.model(x).item()
        
        return float(prob)
    
    def save_checkpoint(self, checkpoint_path: str, metadata: Dict = None):
        """Save model checkpoint"""
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
        }
        
        if metadata:
            checkpoint.update(metadata)
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")