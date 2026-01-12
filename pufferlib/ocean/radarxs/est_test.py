"""
EST vs EDF vs Transformer Benchmark
Includes individual trial scatter plots.
"""
import sys
sys.path.insert(0, 'PufferLib')

import numpy as np
import matplotlib.pyplot as plt
import torch
from pufferlib.ocean.radarxs import binding, engine
from pufferlib.ocean.radarxs.models import est, edf
from pufferlib.ocean.radarxs.models.transformer_mcts import TransformerMCTSPlanner

# Constants (Must match environment)
FEATURES_PER_TRACKER = 4
MAX_TRACKERS = 300
GRID_SIZE = 300

def run_planner(planner_cls, n_targets, total_steps=5000, seed=42, model=None):
    """Run specified Planner using the standard RadarEngine."""
    actual_targets = min(n_targets, MAX_TRACKERS)
    
    # Initialize planner
    kw = {'max_trackers': MAX_TRACKERS}
    if model:
        kw['model'] = model
        kw['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    planner = planner_cls(**kw)
    
    rad_engine = engine.RadarEngine(planner, initial_targets=actual_targets, max_trackers=MAX_TRACKERS, seed=seed)
    rad_engine.reset()
    
    total_steps = 0
    total_reward = 0
    
    num_windows = 250
    for _ in range(num_windows):
        rew = rad_engine.step_window()
        total_reward += rew
        total_steps += 20
        
    rad_engine.close()
    
    return total_reward / total_steps

def main():
    print("=" * 60)
    print("EST vs EDF vs Transformer Benchmark (10 Trials)")
    print("=" * 60)
    
    target_counts = list(range(10, 320, 20))
    target_counts = [1, 5] + target_counts
    target_counts = sorted(list(set(target_counts)))
    
    seeds = list(range(42, 52)) # 10 seeds
    
    # Pre-load transformer model to speed up benchmark
    print("Loading Transformer model once...")
    temp_planner = TransformerMCTSPlanner(checkpoint_path='transformer_est.pth', max_trackers=MAX_TRACKERS)
    transformer_model = temp_planner.model
    
    planners = {
        'EST': (est.ESTPlanner, None),
        'EDF': (edf.EDFPlanner, None),
        'Transformer': (TransformerMCTSPlanner, transformer_model),
    }
    
    colors = {
        'EST': 'red',
        'EDF': 'blue',
        'Transformer': 'orange'
    }
    
    # Store results: name -> {load -> [rewards...]}
    results = {name: {n: [] for n in target_counts} for name in planners}
    
    for name, (p_cls, model_obj) in planners.items():
        print(f"\nRunning {name}...")
        for n in target_counts:
            print(f"  Load {n:3d}: ", end='', flush=True)
            for seed in seeds:
                avg_rew = run_planner(p_cls, n, total_steps=5000, seed=seed, model=model_obj)
                results[name][n].append(avg_rew)
                print(".", end='', flush=True)
            
            mean = np.mean(results[name][n])
            print(f" Mean={mean:+.4f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for name, load_dict in results.items():
        c = colors[name]
        
        loads = []
        means = []
        all_x = []
        all_y = []
        
        for n in target_counts:
            rewards = load_dict[n]
            loads.append(n)
            means.append(np.mean(rewards))
            
            # For scatter
            all_x.extend([n] * len(rewards))
            all_y.extend(rewards)
        
        # Plot mean line with shade
        ax.plot(loads, means, color=c, label=f"{name} (Mean)", linewidth=2, zorder=10)
        
        # Plot individual dots
        ax.scatter(all_x, all_y, color=c, s=15, alpha=0.5, label=f"{name} (Trials)", zorder=5)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Number of Targets', fontsize=12)
    ax.set_ylabel('Mean Reward per Step', fontsize=12)
    ax.set_title('Scheduler Benchmarks: EST (Red), EDF (Blue), Transformer (Orange)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Custom legend to avoid duplicate dots
    handles, labels = ax.get_legend_handles_labels()
    # Filter to keep only lines, or simplified legend
    # For now default legend is okay, but might be crowded
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_final.png', dpi=150)
    print(f"\nPlot saved to: benchmark_final.png")

if __name__ == "__main__":
    main()
