import numpy as np
from collections import deque

class ActionScheduler:
    def __init__(self, horizon=16, max_history=3, motion_threshold=0.2, gripper_stable_steps=3):
        self.horizon = horizon
        self.max_history = max_history
        self.motion_threshold = motion_threshold
        self.gripper_stable_steps = gripper_stable_steps
        
        self.history = deque(maxlen=max_history)
        
        self.gripper_width_q = deque(maxlen=gripper_stable_steps)
        
    def decide_chunk_size(self, pred_action_dict, gripper_width_q):
        delta_positions = pred_action_dict["action.end_effector_position"]
        gripper = pred_action_dict["action.gripper_close"]
    
    
        avg_motion = np.mean(np.linalg.norm(delta_positions, axix = 1))
        
        self.prev_gripper_states.append[gripper[0]]
        
        gripper_stable = len(set(self.prev_gripper_states)) == 1
        
        if avg_motion < self.motion_threshold:
            chunk_size = 2
        elif gripper_stable:
            chunk_size = min(self.horizon, 8)
        else:
            chunk_size = 4
        
        return chunk_size
    
    def update_history(self, pred_actions):
        self.history.append(pred_actions)
    
    def get_ensemble_actions(self, step_idx):
        aligned_actions = []
        weights = []
        
        for i, pred in enumerate(self.history):
            if step_idx < len(pred):
                aligned_actions.append(pred[step_idx])
                weights.append(i+1)
        
        if not aligned_actions:
            return None

        aligned_actions = np.stack(aligned_actions)
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum()
        
        return np.sum(aligned_actions * weights[:, None], axis=0)

    def step(self, pred_actions):
        
        self.update_history(pred_actions)
        
        chunk_size = self.decide_chunk_size(pred_actions)
        
        exec_actions = []
        
        for t in range(chunk_size):
            action = self.get_ensemble_actions(t)
            exec_actions.append(action)
            
        return np.stack(exec_actions), chunk_size