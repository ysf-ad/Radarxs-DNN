import numpy as np
from pufferlib.ocean.radarxs.models.planner import Planner

class EDFPlanner(Planner):
    SEARCH_ACTION = 0

    def __init__(self, max_trackers=500):
        super().__init__(max_trackers)

    def plan(self, obs, budget_ms=200):
        """
        Generate simple EST schedule: Sort all active tasks by t_desired.
        Args:
            obs: Observation dict.
            budget_ms: Time budget to fill (used to size the search buffer).
        """
        candidates = []
        
        # 1. Search Candidates
        # Add enough search tasks to fill the budget (plus small buffer)
        search_dwell = 10.0 # Assumption from radarxs.h
        limit = int(budget_ms / search_dwell) + 2
        
        # Sort grid values to prioritize the truly urgent sectors individually
        # This prevents over-scheduling search if only 1 sector is stale
        grid_values = np.sort(obs['grid'])
        
        for i in range(limit):
            # Use the i-th worst grid value as the urgency for the i-th search candidate
            # Clamp to ensuring valid index if limit > grid size (unlikely 300)
            urgency = float(grid_values[min(i, len(grid_values)-1)])
            
            candidates.append({
                'action': self.SEARCH_ACTION,
                'time': urgency
            })
        
        # 2. Track Candidates
        time_metric = obs['t_deadline']
        active_mask = obs['active_mask']
        active_indices = np.where(active_mask)[0]
        
        for idx in active_indices:
            candidates.append({
                'action': int(idx) + 1,
                'time': time_metric[idx]
            })
            
        # 3. Sort EST
        candidates.sort(key=lambda x: x['time'])
        
        # 4. Return full sorted schedule
        plan = [c['action'] for c in candidates]

        # Safety fallback
        if not plan:
            plan = [self.SEARCH_ACTION]
            
        return plan