from collections import OrderedDict, deque
import torch
import random
import numpy as np
import cv2
import math

PORT_MAP = {
    8080: 59049,
    8081: 15118,  
    8082: 64098, 
    8083: 2645,
    8084: 34738,
    8085: 63347,
    8086: 56973,
    8087: 57156,
    8088: 32953,
    8089: 51198,
    8090: 38094
    }


def choose_option(
    options, option_name, show_keys=False, default=None, default_message=None
):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("Here is a list of {}s:\n".format(option_name))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1,
                default_message,
            )
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    # Return the chosen environment name
    return choice


TASKS = OrderedDict(
    [
        ("PnPCabToCounter", "pick and place from cabinet to counter"), # added
        ("PnPCounterToCab", "pick and place from counter to cabinet"), 
        ("PnPCounterToMicrowave", "pick and place from counter to microwave"), # added
        ("PnPCounterToSink", "pick and place from counter to sink"),
        ("PnPCounterToStove", "pick and place from counter to stove"), # added
        ("PnPMicrowaveToCounter", "pick and place from microwave to counter"),
        ("PnPSinkToCounter", "pick and place from sink to counter"), # added
        ("PnPStoveToCounter", "pick and place from stove to counter"),

        ("OpenDrawer", "open drawer"), # added        
        ("OpenSingleDoor", "open cabinet or microwave door"),
        ("OpenDoubleDoor", "open double door"), # added
        ("CloseDrawer", "close drawer"),
        ("CloseSingleDoor", "close single door"), # added
        ("CloseDoubleDoor", "close double door"), # added

        ("TurnOnMicrowave", "turn on microwave"),
        ("TurnOnSinkFaucet", "turn on sink faucet"),
        ("TurnOnStove", "turn on stove"),

        ("TurnOffMicrowave", "turn off microwave"), # added
        ("TurnOffSinkFaucet", "turn off sink faucet"), # added
        ("TurnOffStove", "turn off stove"), # added

        ("TurnSinkSpout", ""), # added

        ("CoffeePressButton", ""), # added
        ("CoffeeServeMug", ""), # added
        ("CoffeeSetupMug", ""), # added
    ]
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def decide_chunk_size(pred_action_dict, start_position, gripper_width_q,
                       motion_threshold=0.2, consective_steps=2,
                       grip_consistency=3):
    
    # check phase
    if gripper_width_q.get_state() == "carry":
        # print("gripper state: carray.")
        return len(pred_action_dict["action.end_effector_position"])

    pred_eef_pos = pred_action_dict["action.end_effector_position"]
    gripper_control = pred_action_dict["action.gripper_close"] # 0 or 1
    
    # find gripper close index
    close_idx = None
    for i in range(len(gripper_control) - grip_consistency + 1):
        window = gripper_control[i : i + grip_consistency]
        if all( g > 0 for g in window):
            close_idx = i + grip_consistency
            # close_idx = min(8, i)
            close_idx = max(4, close_idx)
            break
    
    # find near object index
    near_idx = None
    diffs = np.linalg.norm(pred_eef_pos, axis=1)
    for i in range(len(diffs) - consective_steps):
        if np.all(diffs[i:i+consective_steps] < motion_threshold):
            near_idx = min(8, i)
            near_idx = max(4, near_idx)
            break
    
    # print("diffs:", diffs)
    # print("near_idx:", near_idx)
    # print("close_idx",close_idx)
    # set chunk size
    if close_idx is None:
         return near_idx if near_idx is not None else len(pred_eef_pos)
    else:
        if near_idx is None:
            return close_idx
        else:
            # return max(near_idx, close_idx)
            return close_idx



def collect_observation(obs, lang):
    # create observation
    obs_dict = {}

    # video keys
    obs_dict['video.left_view'] = obs['robot0_agentview_left_image'][None, ...]
    obs_dict['video.right_view'] = obs['robot0_agentview_right_image'][None, ...]
    obs_dict['video.wrist_view'] = obs['robot0_eye_in_hand_image'][None, ...]

    # state keys
    obs_dict['state.end_effector_position_relative'] = obs['robot0_base_to_eef_pos'][None, ...]
    obs_dict['state.end_effector_rotation_relative'] = obs['robot0_base_to_eef_quat'][None, ...]
    obs_dict['state.gripper_qpos'] = obs['robot0_gripper_qpos'][None, ...]
    obs_dict["state.base_position"] = obs['robot0_base_pos'][None, ...]
    obs_dict["state.base_rotation"] = obs['robot0_base_quat'][None, ...]

    # language
    obs_dict["annotation.human.action.task_description"] = [lang]

    return obs_dict

def merge_cam_view(obs, cam_names, trial_id, sim_step, lang, chunk_to_execute=None):
    merge_view_list = []
    for _, show_cam in enumerate(cam_names):
        image_array = obs[show_cam + "_image"]
        bgr_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        merge_view_list.append(bgr_img)
        pass
    merge_view = np.concatenate(merge_view_list, axis=1)
    if chunk_to_execute:
        cv2.putText(merge_view, f"trial {trial_id}, sim_t {sim_step}, chunk:{chunk_to_execute}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
    else:
        cv2.putText(merge_view, f"trial {trial_id}, sim_t {sim_step}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
    cv2.putText(merge_view, f"Instruction: {lang}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return merge_view

def merge_cam_view_ac(obs, cam_names, trial_id, sim_step, lang, status):
    merge_view_list = []
    for _, show_cam in enumerate(cam_names):
        image_array = obs[show_cam + "_image"]
        bgr_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        merge_view_list.append(bgr_img)
        pass
    merge_view = np.concatenate(merge_view_list, axis=1)
    
    cv2.putText(merge_view, f"trial {trial_id}, sim_t {sim_step}, {status}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
    cv2.putText(merge_view, f"Instruction: {lang}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return merge_view


def decide_chunk_by_distance(chunk_size, short_chunk, gripper_width, last_width, d, last_d):
    if gripper_width > 0.075 and d < 0.30 and d-last_d<0: # 夹爪张开并且靠近物体
        # if chunk_to_execute == chunk_size:
        chunk_to_execute = short_chunk
        # print(f"gripper is close to target object, change chunk_size to {chunk_to_execute}.")
    else:
        if gripper_width <= 0.075 and abs(last_width - gripper_width) < 0.002: # 夹爪闭合而且稳定
            chunk_to_execute = chunk_size
            # print(f"change chunk_size to {chunk_size}.")
        
        if d > 0.3:
            chunk_to_execute = chunk_size
            # print(f"change chunk_size to {chunk_size}.")
    
    return chunk_to_execute


class GrippperWidthQueue:
    
    def __init__(self, max_len=10, gripper_closed_thresh=0.078, gripper_open_thresh=0.078, stable_delta=0.002):
        self.queue = deque(maxlen=max_len)
        self.gripper_closed_thresh = gripper_closed_thresh
        self.gripper_open_thresh = gripper_open_thresh
        self.stable_delta = stable_delta
        
    def add(self, width):
        self.queue.append(width)
    
    def get_state(self):
        if len(self.queue) == 0:
            return "open"
        
        avg_width = np.mean(self.queue)
        # print("avg_width:", avg_width)
        width_range = max(self.queue) - min(self.queue)
        
        if avg_width <= self.gripper_closed_thresh and width_range <= self.stable_delta:
            return "carry"
        elif avg_width >= self.gripper_open_thresh:
            return "open"
        else:
            return "grasp"


def axisangle2quat(vec):
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def quat_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions (q1 * q0).

    E.g.:
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True

    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion
        quaternion0 (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
    )
    
def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def compose_rotations(delta_rs):
    q_total = np.array([0.0, 0.0, 0.0, 1.0])
    for r in delta_rs:
        q = axisangle2quat(r)
        q_total = quat_multiply(q, q_total)
    return quat2axisangle(q_total)

def convert_action_dict_to_list(pred_action_dict):
    action_dict = {}
    action_dict["right"] = np.concatenate( (pred_action_dict["action.end_effector_position"],pred_action_dict["action.end_effector_rotation"]), axis = 1) # T, D
    action_dict["right_gripper"] = (pred_action_dict["action.gripper_close"]-0.5)*2# convert from [0, 1] to [-1, 1]
    action_array = np.concatenate([action_dict["right"], action_dict["right_gripper"][..., None]], axis=1) # T,D
    return action_array.tolist()

def cosine_similarity(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim