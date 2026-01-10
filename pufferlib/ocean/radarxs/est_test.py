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
FEATURES_PER_TRACKER = 4
MAX_TRACKERS = 500
GRID_SIZE = 300

def run_est(n_targets, total_steps=5000, seed=42):
    """Run EST Planner using the standard RadarEngine."""
    planner = est.ESTPlanner(max_trackers=MAX_TRACKERS)
    
    # Initialize Engine
    rad_engine = engine.RadarEngine(planner, initial_targets=n_targets, max_trackers=MAX_TRACKERS, seed=seed)
    
    # Run simulation for ~5000 steps if windows are around 20 steps
    stats = rad_engine.run_episode(num_windows=250)
    
    rad_engine.close()
    
    return {
        'n_targets': n_targets,
        'total_reward': stats['total_reward'],
        'avg_reward': stats['avg_reward_per_step'], # Use the per-step metric we added
    }



def main():
    print("=" * 60)
    print("EST Benchmark (Continuous/Ideal Mode)")
    print("=" * 60)
    
    target_counts = [25, 50, 75, 100, 150, 200, 300, 400, 500]
    total_steps = 5000
    
    results = []
    for n in target_counts:
        result = run_est(n, total_steps=total_steps)
        print(f"Load {n:3d}: Avg={result['avg_reward']:+.4f}/step")
        results.append(result)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    loads = [r['n_targets'] for r in results]
    avg_rewards = [r['avg_reward'] for r in results]
    
    colors = ['green' if r > 0 else 'red' for r in avg_rewards]
    ax.bar(loads, avg_rewards, color=colors, alpha=0.7, edgecolor='black', width=20)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel('Initial Targets', fontsize=12)
    ax.set_ylabel('Avg Reward/Step', fontsize=12)
    ax.set_title('EST Performance (New Simple Logic)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (x, y) in enumerate(zip(loads, avg_rewards)):
        ax.annotate(f'{y:+.2f}', (x, y), textcoords="offset points", 
                   xytext=(0, 5 if y > 0 else -15), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('est_simple_benchmark.png', dpi=150)
    print(f"\nPlot saved to: est_simple_benchmark.png")
    # plt.show() # Disabled for headless


if __name__ == "__main__":
    main()
