import robosuite
from robosuite.controllers import load_composite_controller_config
from termcolor import colored
import robocasa.macros as macros
import numpy as np
from termcolor import colored
from robocasa.demos.policy_services import ExternalRobotInferenceClient
import cv2
from robocasa.demos.help_function import set_seed, collect_observation, merge_cam_view
import argparse
from datetime import datetime
import os
from robocasa.demos.results_logger import ResultLogger, VideoLogger
from robocasa.demos.action_optimization.action_entropy_v2 import LiveEntropyPlot, select_chunk_size
from robocasa.demos.action_optimization.action_sampler import select_chunk_id
import json
import copy
import csv

def run_task():
    # Parse arguments
    parser = argparse.ArgumentParser()
    
    # debug
    parser.add_argument("--debug", type=bool, default=False, help="debug mode")
    parser.add_argument("--render", type=bool, default=False, help="Whether to render the environment")
    parser.add_argument("--vis_plots", type=bool, default=False, help="visualize action entropy plots")
    parser.add_argument("--selected_trial_id", type=int, default=None, help="specify one trial id to debug")
    
    # files to save
    parser.add_argument("--out_path", type=str, default="/home/user/Videos/robocasa/", help="path to save the videos")
    parser.add_argument("--wait_frames", type=bool, default=False, help="whether to visualize wait frames")
    parser.add_argument("--save_videos", type=bool, default=True, help="Whether to save the videos")
    parser.add_argument("--save_logs", type=bool, default=True, help="whether to save csv logs")
    parser.add_argument("--save_plots", type=bool, default=False, help="whether to save plots")
    
    # network
    parser.add_argument("--host", type=str, default="10.72.1.16", help="server address")
    parser.add_argument("--port", type=int, default=8089, help="server port")
    
    # simulation config
    parser.add_argument("--task", type=str, default="PnPCounterToCab", help="simulation task")
    # PnPCabToCounter, PnPCounterToCab, PnPCounterToMicrowave, PnPCounterToSink, PnPCounterToStove, PnPMicrowaveToCounter, PnPSinkToCounter, PnPStoveToCounter
    parser.add_argument("--trials", type=int, default=100, help="number of total trials")
    parser.add_argument("--max_steps", type=int, default=600, help="max simulation steps per trial")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--completion_count", type=int, default=3, help="count for task success")
    
    # algorithm related
    parser.add_argument("--chunk_size_selector", type=str, default="fixed_16", help="method to choose chunk size")
    # gaussian_bernoulli, gaussian_only, variance, separate, binning, fixed_16
    parser.add_argument("--chunk_id_selector", type=str, default="0", help="method to choose chunk id")
    # "0"， "backward" 
    parser.add_argument("--move_th", type=float, default=3.0, help="Minimum movement threshold")
    
    
    args = parser.parse_args()
    
    task = args.task
    print(f"\n Running task {task}...")
    
    if args.debug:
        # args.render = True
        args.save_videos = False
        args.save_logs = False
        args.save_plots = False
    

    # Create robosuite argument configuration
    cam_names = ['robot0_agentview_left','robot0_eye_in_hand', 'robot0_agentview_right']
    robosuite_config = {
        "env_name": task,
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
        "translucent_robot": False,
        "render_camera" : "robot0_eye_in_hand", # "robot0_frontview",
        "has_renderer": args.render,
        "has_offscreen_renderer": True,
        "ignore_done":False,
        "use_camera_obs": True,
        "camera_names": cam_names,
        "camera_heights": 256, "camera_widths": 256,
        "control_freq": 20,
        "renderer":"mjviewer",
        "horizon":args.max_steps,
    } 

    print(colored(f"Initializing environment...", "yellow"))
    
    now = datetime.now()
    args_dict = vars(args)
    args.out_path = os.path.join(args.out_path + "_" + task + "_" + args.chunk_size_selector + "_" + args.chunk_id_selector,
                                 task + "_" + now.strftime("%Y%m%d_%H%M%S")) # add time to out_path
    
    
    
    if args.save_videos:
        video_logger = VideoLogger(output_folder=args.out_path)
    if args.save_logs:
        logger = ResultLogger(output_folder=args.out_path, test_name=task + f"_{args.trials}_trials_")
        with open(os.path.join(args.out_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=4)

    num_trials = args.trials
    success_count = 0
    chunk_size_stats = []
    for trial_id in range(num_trials):
        print(f"trial {trial_id}")
        set_seed(seed=42)
        env_seed = args.selected_trial_id if args.selected_trial_id else trial_id
        # initialize simulation environment
        env = robosuite.make(
            **robosuite_config,
            seed = env_seed
            )
        env.reset()
        
        # get task description
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", None)
        assert lang is not None
        
        print(colored(f"Instruction: {lang}", "green"))

        if args.render:
            env.render()

        zero_action = np.zeros(env.action_dim)
        # do a dummy step thru base env to initalize things, but don't record the step
        obs, _, _, _ = env.step(zero_action)

        policy = ExternalRobotInferenceClient(host=args.host, port=args.port, timeout_ms=5000)

        trial_success = False
        end_episode = False
        lift_success = False
        task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
        sim_step = 0
        inference_id = 0
        
        if obs.get("obj_pos") is not None:
            obj_initial_height = obs["obj_pos"][2]
        frames = []
        
        if args.save_plots or args.vis_plots:
            path = args.out_path if args.save_plots else None
            entropy_plotter = LiveEntropyPlot(trial_id, method=args.chunk_size_selector,
                                              save_dir=path, visualize=args.vis_plots)


        last_action_dict = None
        
        while end_episode==False and trial_success==False:
            
            # ----------------------- setp 1 collect observation  ------------------------------------#

            # create observation
            obs_dict = collect_observation(copy.deepcopy(obs), lang)
                    
                    
            # ----------------------- setp 2 get action  ------------------------------------#
            
            if args.save_videos and args.wait_frames:
                wait_frame = np.zeros((robosuite_config["camera_heights"], robosuite_config["camera_widths"]*3, 3), dtype=np.uint8)
                cv2.putText(wait_frame, f"observation point", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                for _ in range (10):
                    frames.append(wait_frame)
            pred_action_dict = policy.get_action(obs_dict)
            pred_action_dict_original = copy.deepcopy(pred_action_dict)
            
            
            # ----------------------- setp 3 process action  ------------------------------------#

            # selelct chunk size  
            if "fixed" in args.chunk_size_selector:
                chunk_size = int(args.chunk_size_selector.split("_")[-1])
                for k, v in pred_action_dict.items():
                    pred_action_dict[k] = pred_action_dict[k][:, :chunk_size]
            else:
                chunk_size, breakdown = select_chunk_size(pred_action_dict, method=args.chunk_size_selector, move_th=args.move_th)
                if args.save_plots or args.vis_plots: 
                    merge_view = merge_cam_view( copy.deepcopy(obs), cam_names, trial_id, sim_step, lang, chunk_size)
                    entropy_plotter.update(sim_step, breakdown, chunk_size, img=copy.deepcopy(merge_view))
                for k, v in pred_action_dict.items():
                    pred_action_dict[k] = pred_action_dict[k][:, :chunk_size]

            
            # slelect chunk id
            if args.chunk_id_selector == "mean":
                chunk_id = select_chunk_id(pred_action_dict, method=args.chunk_id_selector) 
                for k, v in pred_action_dict.items():
                    pred_action_dict[k] = pred_action_dict[k][chunk_id, :]
            elif args.chunk_id_selector == "backward":
                chunk_id = select_chunk_id(pred_action_dict_original, method=args.chunk_id_selector, last_action_dict=last_action_dict, pred_chunk_szie=chunk_size)
                for k, v in pred_action_dict.items():
                    pred_action_dict[k] = pred_action_dict[k][chunk_id, :]
            else:
                chunk_id = 0 # select the first chunk by default
                for k, v in pred_action_dict.items():
                    pred_action_dict[k] = pred_action_dict[k][chunk_id, :]
            
            last_action_dict = pred_action_dict_original
            for k, v in last_action_dict.items():
                last_action_dict[k] = last_action_dict[k][chunk_id:chunk_id+1, :]
            
            last_action_dict["chunk_size"] = chunk_size
            # ----------------------- setp 4 execute action in simulation ------------------------------------#   
            chunk_to_execute = pred_action_dict[next(iter(pred_action_dict))].shape[0]
            chunk_size_stats.append({
                "trial_id": trial_id,
                "inference_id": inference_id,
                "sim_step": sim_step,
                "chunk_size": chunk_to_execute
            })
            inference_id += 1
            # print("chunk_to_execute:", chunk_to_execute)
            # cv2.waitKey(0)
            for i in range(chunk_to_execute):
                
                # real time visualization
                if args.render or args.save_videos:
                    merge_view = merge_cam_view(copy.deepcopy(obs), cam_names, trial_id, sim_step, lang, chunk_to_execute) 
                    if args.render:
                        env.render()
                        cv2.imshow("video", merge_view)
                        if cv2.waitKey(1) & 0xFF == 27:
                            print("exit.")
                            env.close()
                            exit()            
                    if args.save_videos:    
                        frames.append(merge_view)
                
                # create env action dict 
                action_dict = {}
                action_dict["right"] = np.concatenate( (pred_action_dict["action.end_effector_position"], pred_action_dict["action.end_effector_rotation"]), axis = 1)[i].tolist()
                action_dict["right_gripper"] = [(pred_action_dict["action.gripper_close"][i]-0.5)*2]
                active_robot = env.robots[0]
                env_action = active_robot.create_action_vector(action_dict) # "right" (0,6), "right_gripper" (6,7), "base" (7,10), "torso" (10,11)

                # execute one step
                obs, reward, done, info = env.step(env_action)
                sim_step+=1
                
                # check lift success
                if obs.get("obj_pos") is not None:
                    obj_height = obs["obj_pos"][2]
                    # print(obj_height)
                    if obj_height - obj_initial_height > 0.02: # 0.1
                        if not lift_success:
                            print("Lift success.")
                            lift_success = True
                else:
                    lift_success = False
            
                # check if we complete the task
                if task_completion_hold_count == 0:
                    trial_success = True
                    success_count += 1
                    if args.save_logs:
                        lift_tag = 1 if lift_success else 0
                        logger.log(trial_id = trial_id, instruction = lang, lifted = lift_tag, success=1, end_time = sim_step)
                    if args.save_videos:
                        video_logger.save_video(frames, task, trial_id, success=True)
                    if args.save_plots or args.vis_plots:
                        entropy_plotter.close()
                    print(f"Task success! current success rate: {success_count/(trial_id+1):.2f}")
                    break

                # check if we reach the simulation time limit
                if done:
                    end_episode = True
                    if args.save_logs:
                        lift_tag = 1 if lift_success else 0
                        logger.log(trial_id = trial_id, instruction = lang, lifted = lift_tag, success=0, end_time = sim_step)
                    if args.save_videos:
                        video_logger.save_video(frames, task, trial_id, success=False) 
                    if args.save_plots or args.vis_plots:
                        entropy_plotter.close()
                    print("End of episode.")
                    break

                # state machine to check for having a success for a few consecutive timesteps
                if env._check_success():
                    if task_completion_hold_count > 0:
                        task_completion_hold_count -= 1  # latched state, decrement count
                    else:
                        task_completion_hold_count = args.completion_count  # reset count on first success timestep
                else:
                    task_completion_hold_count = -1  # null the counter if there's no success
            
            # end of action chunk execution
        
        # end of one trial 
        env.close()
        if args.save_plots or args.vis_plots:
            entropy_plotter.close()
    
    # finish  all trials 
    if args.save_logs:
        logger.save()
        txt_file = os.path.join(args.out_path, "success_rate.txt")
        with open(txt_file, "w", encoding="utf-8") as file:
            content = f"success_count:{success_count}, num_trials:{num_trials}, success_rate:{success_count/num_trials}"
            file.write(content)
        
        chunk_size_csv = os.path.join(args.out_path, "chunk_size_log.csv")
        with open(chunk_size_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["trial_id", "inference_id", "sim_step", "chunk_size"])
            writer.writeheader()
            writer.writerows(chunk_size_stats)
    
    print(f"success_count:{success_count}, num_trials:{num_trials}, success_rate:{success_count/num_trials}")
    

if __name__ == "__main__":
    run_task()