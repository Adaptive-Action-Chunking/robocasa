import numpy as np

def euclidean_distance(src, tar, reduction='mean'):
    # B, (N), T, D
    diff = src - tar
    dist = np.linalg.norm(diff, axis=-1)
    if reduction == 'mean':
        return dist.mean(axis=-1)
    elif reduction == 'none':
        return dist


#------------------------ method to select chunk id ----------------------------------------#

def select_chunk_id(pred_action_dict: dict, method, last_action_dict, pred_chunk_szie, full_action_horizon=16, beta=0.99):
    
    if method == "mean":
        normalized_action = pred_action_dict["normalized_action"][:, :, :7]
        N, T, D = normalized_action.shape
        flattened = normalized_action.reshape(N, -1) # 20, 16*7
        mean_vector = flattened.mean(axis=0)
        distances = np.linalg.norm(flattened - mean_vector, axis=1)
        best_idx = np.argmin(distances)
    
    elif method == "backward":
        use_predicted_chunk_size_only = True # whether to use all rest unexecuted actions from last prediction 
        if not last_action_dict:
            return 0 # first inference
        last_chunk_size = last_action_dict["chunk_size"]
        
        if last_chunk_size == full_action_horizon:
            return 0 # no overlap actions, use the first action by default
        
        
        if use_predicted_chunk_size_only:
            last_action_overlap = last_action_dict["normalized_action"][:, last_chunk_size : last_chunk_size + pred_chunk_szie, :7]
        else:
            last_action_overlap = last_action_dict["normalized_action"][:, last_chunk_size:, :7]
        
        curr_action_overlap = pred_action_dict["normalized_action"][:, :last_action_overlap.shape[1], :7]
        
        dist_raw = euclidean_distance(src=curr_action_overlap, tar=last_action_overlap, reduction="none")
        
        weights = np.array([beta**i for i in range(last_action_overlap.shape[1])])
        weights = weights / weights.sum()
        dist_weighted = dist_raw * weights.reshape(1, 1, last_action_overlap.shape[1])
        dist_strong_sum = dist_weighted.sum(axis=2)
        cross_index = np.argsort(dist_strong_sum, axis=1)
        best_idx = cross_index[0][0]
    
    return best_idx

