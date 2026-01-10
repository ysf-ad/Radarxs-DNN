"""
Radar Engine: 200ms Window Execution for Radar Task Scheduling.
Connects planners (EST, MCTS, Transformer) to the C binding environment.

NOTE: Original radarxs.h has 3 features per tracker:
  - t_desired: Time until desired update (decrements each step)
  - t_deadline: Time until deadline (decrements each step)  
  - t_dwell: Estimated dwell time for this target

Priority is stored as the 4th "virtual" feature (actually part of Target struct, 
but we derive it from t_deadline or use NO_TARGET=-1 as inactive marker).
"""
import numpy as np
from . import binding


# Environment constants (from radarxs.h)
GRID_SIZE = 300  # 30 az * 10 el slices (MAX_AZ_SLICES * MAX_EL_SLICES)
MAX_TRACKERS = 500
FEATURES_PER_TRACKER = 4  # t_desired, t_deadline, t_dwell, priority (MATCH compiled .so!)
NO_TARGET = -1


def get_obs_from_buf(obs_buf, max_trackers=MAX_TRACKERS):
    """
    Convert flat observation buffer to planner-compatible format.
    
    Args:
        obs_buf: Flat observation buffer from binding, shape (1, obs_size).
                 Layout: [Grid(300)] + [Tracker0(4), Tracker1(4), ...] + [sensor_id]
        max_trackers: Maximum number of trackers.
    
    Returns:
        dict: Observation with keys:
            - 'grid': (300,) array of sector freshness values
            - 't_desired': (max_trackers,) time until desired update
            - 't_deadline': (max_trackers,) time until deadline
            - 't_dwell': (max_trackers,) dwell time estimate
            - 'priority': (max_trackers,) priority (-1 = inactive)
            - 'active_mask': (max_trackers,) boolean mask of active targets
            - 'sensor_id': Current sensor (0=S-band, 1=X-band)
    """
    obs_flat = obs_buf[0]
    
    # Extract grid (sector freshness)
    grid = obs_flat[:GRID_SIZE]
    
    # Extract tracker data
    base_idx = GRID_SIZE
    end_idx = base_idx + max_trackers * FEATURES_PER_TRACKER
    flat_trackers = obs_flat[base_idx:end_idx].reshape(max_trackers, FEATURES_PER_TRACKER)
    
    t_desired = flat_trackers[:, 0]
    t_deadline = flat_trackers[:, 1]
    t_dwell = flat_trackers[:, 2]
    priority = flat_trackers[:, 3]
    
    # Active mask: target is active if priority != NO_TARGET (-1)
    active_mask = priority != NO_TARGET
    
    # Sensor ID (last element)
    sensor_id = int(obs_flat[end_idx]) if end_idx < len(obs_flat) else 0
    
    return {
        'grid': grid,
        't_desired': t_desired,
        't_deadline': t_deadline,
        't_dwell': t_dwell,
        'priority': priority,
        'active_mask': active_mask,
        'sensor_id': sensor_id,
    }


class RadarEngine:
    """
    Variable-length Window Execution Engine for Radar Task Scheduling.
    
    Connects any planner (EST, MCTS, Transformer) to the C environment binding.
    Each call to step_window() generates an action plan (length determined by planner) and executes it.
    """
    
    def __init__(self, planner, initial_targets=50, max_trackers=MAX_TRACKERS, seed=1):
        """
        Args:
            planner: Planner instance with a plan() method (EST, MCTS, or Transformer).
            initial_targets: Number of targets to initialize.
            max_trackers: Maximum tracker capacity.
            seed: Random seed.
        """
        self.planner = planner
        self.initial_targets = initial_targets
        self.max_trackers = max_trackers
        self.seed = seed
        
        # Environment buffers
        obs_size = GRID_SIZE + max_trackers * FEATURES_PER_TRACKER + 1
        self.num_envs = 1
        self.obs_buf = np.zeros((self.num_envs, obs_size), dtype=np.float32)
        self.act_buf = np.zeros((self.num_envs,), dtype=np.int32)
        self.rew_buf = np.zeros((self.num_envs,), dtype=np.float32)
        self.term_buf = np.zeros((self.num_envs,), dtype=np.uint8)
        self.trunc_buf = np.zeros((self.num_envs,), dtype=np.uint8)
        
        # Initialize environment
        self.env = binding.vec_init(
            self.obs_buf, self.act_buf, self.rew_buf, 
            self.term_buf, self.trunc_buf, self.num_envs, seed,
            initial_targets=initial_targets, max_trackers=max_trackers
        )
        
        # Statistics
        self.total_reward = 0.0
        self.total_steps = 0
        self.windows_completed = 0
    
    def reset(self, seed=None):
        """Reset the environment."""
        if seed is None:
            seed = self.seed
        binding.vec_reset(self.env, seed)
        self.total_reward = 0.0
        self.total_steps = 0
        self.windows_completed = 0
    
    def step_window(self):
        """
        Execute a single planning window.
        
        1. Convert observation to MCTS format.
        2. Call planner.plan() to get action sequence.
        3. Execute each action via binding.vec_step().
        
        Returns:
            float: Total reward accumulated in this window.
        """
        # Get observation in MCTS format
        mcts_obs = get_obs_from_buf(self.obs_buf, self.max_trackers)
        
        # Generate plan
        plan = self.planner.plan(mcts_obs)
        
        # Execute plan
        window_reward = 0.0
        for action in plan:
            self.act_buf[0] = int(action)
            binding.vec_step(self.env)
            window_reward += self.rew_buf[0]
            self.total_steps += 1
            
            # Check for episode termination
            if self.term_buf[0]:
                break
        
        self.total_reward += window_reward
        self.windows_completed += 1
        
        return window_reward
    
    def run_episode(self, num_windows=50):
        """
        Run a full episode of specified windows.
        
        Args:
            num_windows: Number of planning windows to execute.
        
        Returns:
            dict: Episode statistics.
        """
        self.reset()
        
        for w in range(num_windows):
            window_reward = self.step_window()
            
            if self.term_buf[0]:
                break
        
        return {
            "total_reward": self.total_reward,
            "total_steps": self.total_steps,
            "windows_completed": self.windows_completed,
            "avg_reward_per_window": self.total_reward / max(1, self.windows_completed),
            "avg_reward_per_step": self.total_reward / max(1, self.total_steps),
        }
    
    def get_active_count(self):
        """Get the number of currently active (tracked) targets."""
        end_idx = GRID_SIZE + self.max_trackers * FEATURES_PER_TRACKER
        flat_trackers = self.obs_buf[0, GRID_SIZE:end_idx].reshape(-1, FEATURES_PER_TRACKER)
        # Active if priority (index 3) != NO_TARGET (-1)
        return int(np.sum(flat_trackers[:, 3] != NO_TARGET))
    
    def close(self):
        """Close the environment."""
        if not getattr(self, '_closed', False):
            binding.vec_close(self.env)
            self._closed = True


def benchmark_planner(planner, target_counts=[50, 100, 200, 500], num_windows=50, seed=1):
    """
    Benchmark a planner across different target loads.
    
    Args:
        planner: Planner instance with a plan() method.
        target_counts: List of target counts to test.
        num_windows: Windows per test.
        seed: Random seed.
    
    Returns:
        dict: Results mapping target_count -> avg_reward.
    """
    results = {}
    
    for n_targets in target_counts:
        engine = RadarEngine(planner, initial_targets=n_targets, seed=seed)
        stats = engine.run_episode(num_windows)
        results[n_targets] = stats["avg_reward_per_step"]
        engine.close()
        
        print(f"  {n_targets} targets: {stats['avg_reward_per_step']:.4f} reward/step")
    
    return results
