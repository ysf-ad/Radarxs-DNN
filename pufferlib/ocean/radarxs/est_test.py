"""
EST Benchmark - Using ESTPlanner Class
"""
import sys
sys.path.insert(0, 'PufferLib')

import numpy as np
import matplotlib.pyplot as plt
from pufferlib.ocean.radarxs import binding, engine
from pufferlib.ocean.radarxs.models import est

# Constants (Must match environment)
# Constants (Must match environment)
FEATURES_PER_TRACKER = 4
MAX_TRACKERS = 300  # Cap at 300 to show flatline
GRID_SIZE = 300

def run_est(n_targets, total_steps=5000, seed=42):
    """Run EST Planner using the standard RadarEngine."""
    # Simulate "Ignoring Excess": Cap init targets at MAX
    actual_targets = min(n_targets, MAX_TRACKERS)
    
    planner = est.ESTPlanner(max_trackers=MAX_TRACKERS)
    
    # Initialize Engine
    rad_engine = engine.RadarEngine(planner, initial_targets=actual_targets, max_trackers=MAX_TRACKERS, seed=seed)
    
    # Debug: Check active count
    start_obs = None
    end_obs = None
    
    # We need to manually run loop to inspect
    obs = rad_engine.reset()
    start_obs = engine.get_obs_from_buf(rad_engine.obs_buf, MAX_TRACKERS)
    start_active = np.sum(start_obs['active_mask'])
    
    total_steps = 0
    total_reward = 0
    
    # Run simulation for ~5000 steps
    num_windows = 250
    for _ in range(num_windows):
        rew = rad_engine.step_window()
        total_reward += rew
        total_steps += 20 # Approx
        
    end_obs = engine.get_obs_from_buf(rad_engine.obs_buf, MAX_TRACKERS)
    end_active = np.sum(end_obs['active_mask'])
    
    rad_engine.close()
    
    return {
        'n_targets': n_targets, 
        'total_reward': total_reward,
        'avg_reward': total_reward / total_steps,
        'start_active': start_active,
        'end_active': end_active
    }

def main():
    print("=" * 60)
    print("EST Benchmark (Max Trackers = 300 -> Flatline)")
    print("=" * 60)
    
    # Paper-like sweep: 10 to 500
    target_counts = list(range(10, 520, 20))
    # Add critical points
    target_counts = [1, 5, 290, 300, 310, 350] + target_counts
    target_counts = sorted(list(set(target_counts)))
    
    seeds = [42, 43, 44] # 3 Trials
    
    results = {} # n -> [rewards...]
    
    for n in target_counts:
        run_rewards = []
        for seed in seeds:
            res = run_est(n, total_steps=5000, seed=seed)
            run_rewards.append(res['avg_reward'])
        
        avg = np.mean(run_rewards)
        std = np.std(run_rewards)
        results[n] = {'mean': avg, 'std': std} 
        
        print(f"Load {n:3d}: Mean={avg:+.4f}, Std={std:.4f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    loads = sorted(results.keys())
    means = np.array([results[n]['mean'] for n in loads])
    stds = np.array([results[n]['std'] for n in loads])
    
    # Line with Error Band
    ax.plot(loads, means, 'b-', label='EST Mean Reward', linewidth=2)
    ax.fill_between(loads, means - stds, means + stds, color='blue', alpha=0.2, label='±1 Std Dev')
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Number of Targets', fontsize=12)
    ax.set_ylabel('Mean Reward per Step', fontsize=12)
    ax.set_title('EST Performance: Reward vs Load', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('est_paper_plot.png', dpi=150)
    print(f"\nPlot saved to: est_paper_plot.png")


if __name__ == "__main__":
    main()
