"""
Train Transformer to clone EST policy via Behavioral Cloning.
Collects (obs, action) pairs from EST expert and trains supervised.
"""
import sys
sys.path.insert(0, 'PufferLib')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pufferlib.ocean.radarxs import engine
from pufferlib.ocean.radarxs.models import est
from pufferlib.ocean.radarxs.models.transformer_mcts import PretrainedPureTransformer

# Config
MAX_TRACKERS = 300
NUM_TASKS = MAX_TRACKERS + 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

def collect_data(num_episodes=100, steps_per_episode=50, max_trackers=MAX_TRACKERS):
    """Collect expert (EDF) demonstrations."""
    X = []  # observations (8-feature format)
    Y = []  # actions
    
    expert = est.ESTPlanner(max_trackers=max_trackers)
    
    for ep in range(num_episodes):
        # Random target count
        n_targets = np.random.randint(50, max_trackers)
        rad = engine.RadarEngine(expert, initial_targets=n_targets, max_trackers=max_trackers, seed=ep)
        rad.reset()
        
        for step in range(steps_per_episode):
            obs = engine.get_obs_from_buf(rad.obs_buf, max_trackers)
            
            # Get expert action
            plan = expert.plan(obs, budget_ms=200)
            if not plan:
                break
            action = plan[0]
            
            # Convert obs to 8-feature format
            adapted = adapt_obs_for_training(obs, max_trackers)
            X.append(adapted)
            Y.append(action)
            
            # Step environment
            rad.step_window()
        
        rad.close()
        if (ep + 1) % 10 == 0:
            print(f"Collected {ep+1}/{num_episodes} episodes, {len(X)} samples", flush=True)
    
    return np.array(X), np.array(Y)

def adapt_obs_for_training(obs, max_trackers):
    """Convert 4-feature obs to 8-feature format for Transformer."""
    num_tasks = max_trackers + 1
    adapted = np.zeros((num_tasks, 8), dtype=np.float32)
    
    # Task 0 = Search
    adapted[0, :] = 0.0
    
    # Tasks 1..N = Trackers
    adapted[1:, 0] = obs['t_desired']      # start_time
    adapted[1:, 1] = obs['t_dwell']        # exec_time
    adapted[1:, 2] = np.maximum(0, -obs['t_desired']) * 0.01  # tardiness
    adapted[1:, 3] = obs['t_deadline']     # drop_time
    adapted[1:, 4] = 100.0                 # drop_cost
    adapted[1:, 5] = 0.0                   # scheduled_exec
    adapted[1:, 6] = 0.0                   # scheduled
    adapted[1:, 7] = (~obs['active_mask']).astype(np.float32)  # inactive
    
    return adapted

def train_model(X, Y, epochs=50, batch_size=64, lr=1e-4):
    """Train Transformer via supervised learning."""
    # Sanitize NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    model = PretrainedPureTransformer(num_tasks=NUM_TASKS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    Y_tensor = torch.from_numpy(Y).long().to(DEVICE)
    
    dataset_size = len(X)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle
        perm = torch.randperm(dataset_size)
        X_shuffled = X_tensor[perm]
        Y_shuffled = Y_tensor[perm]
        
        for i in range(0, dataset_size, batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_Y = Y_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'transformer_est.pth')
    
    print(f"\nBest loss: {best_loss:.4f}")
    print("Saved checkpoint: transformer_est.pth")
    return model

def validate_model(model, num_episodes=10, max_trackers=MAX_TRACKERS):
    """Validate trained model against EST expert."""
    model.eval()
    expert = est.ESTPlanner(max_trackers=max_trackers)
    
    matches = 0
    total = 0
    
    for ep in range(num_episodes):
        n_targets = 100
        rad = engine.RadarEngine(expert, initial_targets=n_targets, max_trackers=max_trackers, seed=100+ep)
        rad.reset()
        
        for _ in range(20):
            obs = engine.get_obs_from_buf(rad.obs_buf, max_trackers)
            
            expert_plan = expert.plan(obs, budget_ms=200)
            if not expert_plan:
                break
            expert_action = expert_plan[0]
            
            adapted = adapt_obs_for_training(obs, max_trackers)
            adapted_tensor = torch.from_numpy(adapted).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(adapted_tensor)
                model_action = int(torch.argmax(logits).item())
            
            if model_action == expert_action:
                matches += 1
            total += 1
            
            rad.step_window()
        
        rad.close()
    
    accuracy = matches / total if total > 0 else 0
    print(f"\nValidation Accuracy: {accuracy:.2%} ({matches}/{total})")
    return accuracy

def main():
    print("=" * 60)
    print("Training Transformer to Clone EST Policy")
    print("=" * 60)
    
    # Collect data
    print("\n[1/3] Collecting expert demonstrations...")
    X, Y = collect_data(num_episodes=500, steps_per_episode=100)
    print(f"Collected {len(X)} samples")
    
    # Train
    print("\n[2/3] Training Transformer...")
    model = train_model(X, Y, epochs=100, batch_size=128, lr=1e-4)  # Reduced LR for stability
    
    # Validate
    print("\n[3/3] Validating trained model...")
    validate_model(model)

if __name__ == "__main__":
    main()
