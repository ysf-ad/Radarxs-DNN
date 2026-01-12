"""
Transformer-guided MCTS Planner for Radar Task Scheduling.
Uses 4 features per tracker (t_desired, t_deadline, t_dwell, priority).

NOTE: Transformer expects 8-feature format. adapt_obs() converts from 4-feature.
Falls back to pure MCTS if no checkpoint loaded.
"""
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .mcts import MCTSPlanner, Node
from .planner import Planner


class PretrainedPureTransformer(nn.Module if TORCH_AVAILABLE else object):
    """Transformer for task scheduling (8-feature input)."""
    
    def __init__(self, num_tasks=501, num_features=8, d_model=128, nhead=8, nlayers=4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.d_model = d_model
        
        self.task_embedding = nn.Linear(num_features, d_model)
        self.position_embedding = nn.Embedding(num_tasks, d_model)
        self.cls_token = nn.Parameter(torch.randn(d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        positions = torch.arange(self.num_tasks, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        embeddings = self.task_embedding(x) + self.position_embedding(positions)
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        output = self.transformer(embeddings)
        task_outputs = output[:, 1:, :]
        return self.policy_head(task_outputs).squeeze(-1)
    
    def predict(self, x):
        x = np.array(x)
        if x.ndim == 4:
            x = x.squeeze(-1).squeeze(0).T.reshape(1, self.num_tasks, self.num_features)
        elif x.ndim == 2:
            x = x.reshape(1, self.num_tasks, self.num_features)
        
        x = torch.from_numpy(x).float().to(next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1).cpu().numpy()


class TransformerMCTSPlanner(Planner):
    """Transformer-guided MCTS (falls back to pure MCTS if no model)."""
    
    def __init__(self, checkpoint_path=None, model=None, max_trackers=500, num_rollouts=50, device='cuda', use_search=True):
        super().__init__(max_trackers)
        self.max_trackers = max_trackers
        self.num_tasks = max_trackers + 1
        self.SEARCH_ACTION = 0
        
        # Fallback MCTS
        self.pure_mcts = MCTSPlanner(max_trackers=max_trackers, num_rollouts=num_rollouts)
        
        # Load Transformer if available
        self.model = model
        if self.model is None and TORCH_AVAILABLE and checkpoint_path and os.path.exists(checkpoint_path):
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            self.model = PretrainedPureTransformer(num_tasks=self.num_tasks).to(self.device)
            
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model_state = self.model.state_dict()
            filtered = {k: v for k, v in state_dict.items() 
                       if k in model_state and v.shape == model_state[k].shape}
            model_state.update(filtered)
            self.model.load_state_dict(model_state)
            self.model.eval()
            print(f"Loaded Transformer from {checkpoint_path}")
    
    def adapt_obs(self, obs):
        """Convert 4-feature radarxs obs to 8-feature Transformer format."""
        adapted = np.zeros((1, 8, self.num_tasks, 1), dtype=np.float32)
        
        # Task 0 = Search
        adapted[0, :, 0, 0] = 0.0
        
        # Tasks 1..N = Trackers
        adapted[0, 0, 1:, 0] = obs['t_desired']      # start_time
        adapted[0, 1, 1:, 0] = obs['t_dwell']        # exec_time
        adapted[0, 2, 1:, 0] = np.maximum(0, -obs['t_desired']) * 0.01  # tardiness
        adapted[0, 3, 1:, 0] = obs['t_deadline']     # drop_time
        adapted[0, 4, 1:, 0] = 100.0                 # drop_cost
        adapted[0, 5, 1:, 0] = 0.0                   # scheduled_exec
        adapted[0, 6, 1:, 0] = 0.0                   # scheduled
        adapted[0, 7, 1:, 0] = (~obs['active_mask']).astype(np.float32)  # inactive
        
        return adapted
    
    
    def plan(self, obs, budget_ms=200):
        """Generate action plan using model-guided MCTS."""
        if self.model is None:
            return self.pure_mcts.plan(obs, budget_ms=budget_ms)
        
        # Determine steps-to-plan (limited by budget)
        max_steps = int(budget_ms / 10.0) + 2
        
        root = Node(
            t_desired=obs['t_desired'],
            t_deadline=obs['t_deadline'],
            t_dwell=obs['t_dwell'],
            priority=obs['priority'],
            active_mask=obs['active_mask']
        )
        
        # Guided MCTS loop
        for _ in range(self.pure_mcts.num_rollouts):
            node = root
            # 1. Select
            while node.expanded and node.children and not node.is_terminal():
                node = self.pure_mcts._ucb_select(node)
            
            # 2. Expand with Model Priors
            if not node.is_terminal() and not node.expanded:
                adapted = self.adapt_obs({
                    't_desired': node.t_desired,
                    't_deadline': node.t_deadline,
                    't_dwell': node.t_dwell,
                    'priority': node.priority,
                    'active_mask': node.active_mask
                })
                priors = self.model.predict(adapted)[0]
                self.pure_mcts._expand(node, priors=priors, top_k=10)
            
            # 3. Simulate (Value)
            reward = self.pure_mcts._simulate(node)
            
            # 4. Backprop
            self.pure_mcts._backprop(node, reward)
        
        # Extract path
        plan = []
        node = root
        for _ in range(max_steps):
            if not node.children:
                # Fallback to model greedy if search depth reached
                adapted = self.adapt_obs({
                    't_desired': node.t_desired,
                    't_deadline': node.t_deadline,
                    't_dwell': node.t_dwell,
                    'priority': node.priority,
                    'active_mask': node.active_mask
                })
                probs = self.model.predict(adapted)[0]
                
                # Mask out inactive trackers and index 0 if targets exist
                probs[0] = 0 # No search if possible
                for i in range(self.max_trackers):
                    if i >= len(node.active_mask) or not node.active_mask[i]:
                        probs[i+1] = 0
                
                action = int(np.argmax(probs)) if np.sum(probs) > 0 else self.SEARCH_ACTION
                plan.append(action)
                if action > 0:
                    node.active_mask[action-1] = False
                continue
                
            best_child = max(node.children, key=lambda c: c.total_reward / max(1, c.visits))
            plan.append(best_child.action)
            node = best_child
            if node.is_terminal(): break
            
        return plan
