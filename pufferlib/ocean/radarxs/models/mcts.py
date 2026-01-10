"""
Pure MCTS Planner for Radar Task Scheduling.
Uses heuristic-based tree search without neural network guidance.

NOTE: Uses 4 features per tracker (t_desired, t_deadline, t_dwell, priority)
"""
import numpy as np


class Node:
    """MCTS Tree Node."""
    
    def __init__(self, t_desired, t_deadline, priority, active_mask, parent=None, action=None):
        self.t_desired = t_desired.copy()
        self.t_deadline = t_deadline.copy()
        self.priority = priority.copy()
        self.active_mask = active_mask.copy()
        
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.expanded = False
    
    def is_terminal(self):
        return not np.any(self.active_mask)
    
    def get_valid_actions(self):
        """Return list of valid actions (0=Search, 1..N=Track)."""
        valid = [0]  # Search always valid
        valid.extend((np.where(self.active_mask)[0] + 1).tolist())
        return valid
    
    def calc_urgency_cost(self):
        """Lower is better - penalize overdue and near-deadline targets."""
        if not np.any(self.active_mask):
            return 0.0
        overdue = np.maximum(0, -self.t_desired) * self.active_mask
        deadline_urgency = np.maximum(0, 100 - self.t_deadline) * self.active_mask
        return np.sum(overdue) + 0.1 * np.sum(deadline_urgency)


class MCTSPlanner:
    """Pure MCTS with EST heuristic rollout."""
    
    def __init__(self, max_trackers=500, num_rollouts=50, exploration_constant=1.414):
        self.max_trackers = max_trackers
        self.num_rollouts = num_rollouts
        self.c = exploration_constant
        self.SEARCH_ACTION = 0
    
    def plan(self, obs, max_steps=None):
        """
        Generate action plan using MCTS.
        
        Args:
            obs: Dict with 't_desired', 't_deadline', 'priority', 'active_mask'
            max_steps: Max actions to return (None = all active targets)
        
        Returns:
            List[int]: Actions
        """
        root = Node(
            t_desired=obs['t_desired'],
            t_deadline=obs['t_deadline'],
            priority=obs['priority'],
            active_mask=obs['active_mask']
        )
        
        # Determine steps
        if max_steps is None:
            max_steps = int(np.sum(obs['active_mask'])) + 1  # All active + 1 search
        
        # Run MCTS
        for _ in range(self.num_rollouts):
            leaf = self._select(root)
            if not leaf.is_terminal() and not leaf.expanded:
                self._expand(leaf)
            reward = self._simulate(leaf)
            self._backprop(leaf, reward)
        
        # Extract plan
        plan = []
        node = root
        for _ in range(max_steps):
            if not node.children:
                action = self._est_action(node)
                plan.append(action)
                if action > 0 and action - 1 < len(node.active_mask):
                    node.active_mask[action - 1] = False
                continue
            
            best_child = max(node.children, key=lambda c: c.total_reward / max(1, c.visits))
            plan.append(best_child.action)
            node = best_child
            
            if node.is_terminal():
                break
        
        return plan
    
    def _select(self, node):
        while node.expanded and node.children and not node.is_terminal():
            node = self._ucb_select(node)
        return node
    
    def _ucb_select(self, node):
        log_parent = np.log(node.visits + 1)
        best_score, best_child = -np.inf, node.children[0] if node.children else None
        
        for child in node.children:
            if child.visits == 0:
                return child
            exploit = child.total_reward / child.visits
            explore = self.c * np.sqrt(log_parent / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score, best_child = score, child
        return best_child
    
    def _expand(self, node, force_engagement=True):
        valid_actions = node.get_valid_actions()
        if force_engagement and len(valid_actions) > 1:
            valid_actions = [a for a in valid_actions if a != 0]
        
        for action in valid_actions:
            child_active = node.active_mask.copy()
            if action > 0:
                child_active[action - 1] = False
            
            child = Node(
                t_desired=node.t_desired,
                t_deadline=node.t_deadline,
                priority=node.priority,
                active_mask=child_active,
                parent=node,
                action=action
            )
            node.children.append(child)
        node.expanded = True
    
    def _simulate(self, node):
        urgency_cost = node.calc_urgency_cost()
        max_cost = self.max_trackers * 100
        reward = 1.0 - min(1.0, urgency_cost / max_cost)
        reward += 0.01 * np.sum(~node.active_mask)
        return reward
    
    def _backprop(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def _est_action(self, node):
        if not np.any(node.active_mask):
            return self.SEARCH_ACTION
        t_desired_masked = np.where(node.active_mask, node.t_desired, np.inf)
        return int(np.argmin(t_desired_masked)) + 1
