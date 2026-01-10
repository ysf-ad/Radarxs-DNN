from abc import ABC, abstractmethod
import numpy as np

class Planner(ABC):
    """
    Abstract Base Class for Radar Planners.
    """
    def __init__(self, max_trackers=500):
        self.max_trackers = max_trackers

    @abstractmethod
    def plan(self, obs: dict, budget_ms: float = 200.0) -> list:
        """
        Generate a sequence of actions given the observation and time budget.
        
        Args:
            obs: Dictionary containing 'grid', 'trackers', 't_dwell', etc.
                 - grid: (GRID_SIZE,) float array of staleness
                 - trackers: (MAX_TRACKERS, FEATURES) float array
                 - t_dwell: (MAX_TRACKERS,) float array of estimated dwell times
                 - priority: (MAX_TRACKERS,) float array (target priority)
                 - active_mask: (MAX_TRACKERS,) bool array
            budget_ms: Time budget in milliseconds for the window.
            
        Returns:
            List of integers representing action indices.
            0 = Search
            1..N = Track Target i-1
        """
        pass
