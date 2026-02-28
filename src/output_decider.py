"""
Decision making module
Converts collision probability to actionable commands
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ActionDecider:
    """Converts collision probabilities to actions"""
    
    ACTIONS = {
        "STOP": 0,
        "SLOW": 1,
        "REROUTE": 2,
        "PROCEED": 3
    }
    
    @staticmethod
    def decide(
        collision_prob: float,
        thresholds: Dict[str, float] = None
    ) -> str:
        """
        Decide action based on collision probability
        
        Args:
            collision_prob: Collision probability (0-1)
            thresholds: Decision thresholds
        
        Returns:
            Action string: STOP, SLOW, REROUTE, or PROCEED
        """
        if thresholds is None:
            thresholds = {
                "stop": 0.8,
                "slow": 0.5,
                "reroute": 0.2
            }
        
        if collision_prob > thresholds.get("stop", 0.8):
            return "STOP"
        elif collision_prob > thresholds.get("slow", 0.5):
            return "SLOW"
        elif collision_prob > thresholds.get("reroute", 0.2):
            return "REROUTE"
        else:
            return "PROCEED"
    
    @staticmethod
    def get_urgency(action: str) -> int:
        """Get urgency level (0=low, 3=high)"""
        return ActionDecider.ACTIONS.get(action, -1)


def decide_output(
    collision_prob: float,
    thresholds: Dict[str, float] = None
) -> str:
    """
    Convert collision probability to action
    
    Args:
        collision_prob: Collision probability (0-1)
        thresholds: Decision thresholds
    
    Returns:
        Action string
    """
    return ActionDecider.decide(collision_prob, thresholds)