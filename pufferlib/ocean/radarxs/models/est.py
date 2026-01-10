import numpy as np
# refer to ../est_test.py to run
class ESTPlanner:
    def __init__(self, max_trackers=500, steps_per_window=20):
        self.max_trackers = max_trackers
        self.steps_per_window = steps_per_window
        self.SEARCH_ACTION = 0

    def plan(self, obs):
        """
        Generate simple EST schedule: Sort all active tasks by t_desired.
        """
        candidates = []
        
        # Consider most urgent search(es)
        # We add enough search candidates to fill the window if necessary
        limit = self.steps_per_window if self.steps_per_window is not None else 20
        grid_min = float(np.min(obs['grid']))
        
        for _ in range(limit):
            candidates.append({
                'action': self.SEARCH_ACTION,
                'time': grid_min
            })
        
        # 2. Track Candidates (All active targets)
        t_desired = obs['t_desired']
        active_mask = obs['active_mask']
        active_indices = np.where(active_mask)[0]
        
        for idx in active_indices:
            candidates.append({
                'action': int(idx) + 1,
                'time': t_desired[idx]
            })
            
        # 3. Sort EST
        candidates.sort(key=lambda x: x['time'])
        
        # 4. Return schedule (limited by window steps)
        plan = [c['action'] for c in candidates]
        
        if self.steps_per_window is not None:
             plan = plan[:self.steps_per_window]
             
        # Safety fallback
        if not plan:
            plan = [self.SEARCH_ACTION]
            
        return plan