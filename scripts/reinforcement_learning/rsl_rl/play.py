# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# Custom velocity ranges
parser.add_argument("--use_custom_velocity", action="store_true", default=False, 
                    help="Use custom velocity ranges instead of the ones in the config")
parser.add_argument("--lin_vel_x_min", type=float, default=-1.0, help="Minimum linear velocity in x direction")
parser.add_argument("--lin_vel_x_max", type=float, default=1.0, help="Maximum linear velocity in x direction")
parser.add_argument("--lin_vel_y_min", type=float, default=-1.0, help="Minimum linear velocity in y direction")
parser.add_argument("--lin_vel_y_max", type=float, default=1.0, help="Maximum linear velocity in y direction")
parser.add_argument("--ang_vel_z_min", type=float, default=-1.0, help="Minimum angular velocity around z axis")
parser.add_argument("--ang_vel_z_max", type=float, default=1.0, help="Maximum angular velocity around z axis")
parser.add_argument("--heading_min", type=float, default=-math.pi, help="Minimum heading angle")
parser.add_argument("--heading_max", type=float, default=math.pi, help="Maximum heading angle")

# Add these arguments after the existing custom velocity arguments
parser.add_argument("--use_velocity_sequence", action="store_true", default=False,
                    help="Use a sequence of velocities that change over time")
parser.add_argument("--velocity_sequence_interval", type=int, default=10,
                    help="Number of timesteps between velocity changes")
parser.add_argument("--velocity_sequence_json", type=str, default="",
                    help="JSON string or path to JSON file containing velocity sequence")

# Add these arguments after the existing velocity sequence arguments
parser.add_argument("--log_data", action="store_true", default=False,
                    help="Save observations, linear velocities, and actions to a JSON file")
parser.add_argument("--log_file", type=str, default="robot_data.json",
                    help="Filename to save the logged data")
parser.add_argument("--log_interval", type=int, default=1,
                    help="Interval (in timesteps) at which to log data")

# Add these arguments after the existing log_data arguments
parser.add_argument("--save_metadata", action="store_true", default=False,
                    help="Save metadata about observations, actions, and velocities to a separate JSON file")
parser.add_argument("--metadata_file", type=str, default="metadata.json",
                    help="Filename for the metadata")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from isaaclab.managers import ObservationTermCfg

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    if args_cli.use_custom_velocity:
        print("[INFO] Using custom velocity ranges from command line arguments")
        # Set velocity ranges directly in the config
        
        # Create new ranges object with the command line values
        lin_vel_x = (args_cli.lin_vel_x_min, args_cli.lin_vel_x_max)
        lin_vel_y = (args_cli.lin_vel_y_min, args_cli.lin_vel_y_max)
        ang_vel_z = (args_cli.ang_vel_z_min, args_cli.ang_vel_z_max)
        heading = (args_cli.heading_min, args_cli.heading_max)
        
        # Update the ranges in the config
        env_cfg.commands.base_velocity.ranges.lin_vel_x = lin_vel_x
        env_cfg.commands.base_velocity.ranges.lin_vel_y = lin_vel_y
        env_cfg.commands.base_velocity.ranges.ang_vel_z = ang_vel_z
        env_cfg.commands.base_velocity.ranges.heading = heading
        
        print(f"  Linear velocity X: {lin_vel_x}")
        print(f"  Linear velocity Y: {lin_vel_y}")
        print(f"  Angular velocity Z: {ang_vel_z}")
        print(f"  Heading: {heading}")

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        # Create a folder name that includes velocity information if custom velocities are used
        video_folder_name = "play"
        if args_cli.use_custom_velocity:
            video_folder_name = f"play_vel_x_{args_cli.lin_vel_x_min}_{args_cli.lin_vel_x_max}_y_{args_cli.lin_vel_y_min}_{args_cli.lin_vel_y_max}_ang_{args_cli.ang_vel_z_min}_{args_cli.ang_vel_z_max}_heading_{args_cli.heading_min}_{args_cli.heading_max}"
            # Shorten the folder name to avoid excessively long paths
            video_folder_name = video_folder_name.replace(".", "p")  # Replace dots with 'p' for better folder names
            
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", video_folder_name),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ONLY export policy to onnx/jit if custom velocities are not used
    if not args_cli.use_custom_velocity:
        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        )
        export_policy_as_onnx(
            ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    current_velocity_index = 0
    
    # Initialize data logging if enabled
    logged_data = []

    # Parse velocity sequence if provided
    velocity_sequence = []
    if args_cli.use_velocity_sequence and args_cli.velocity_sequence_json:
        import json
        
        try:
            # Check if the input is a file path or a JSON string
            if os.path.isfile(args_cli.velocity_sequence_json):
                with open(args_cli.velocity_sequence_json, 'r') as f:
                    velocity_sequence = json.load(f)
                print(f"[INFO] Loaded velocity sequence from file: {args_cli.velocity_sequence_json}")
            else:
                # Try to parse as a JSON string
                velocity_sequence = json.loads(args_cli.velocity_sequence_json)
                print(f"[INFO] Parsed velocity sequence from command line")
            
            # Validate the sequence format
            for i, entry in enumerate(velocity_sequence):
                if not isinstance(entry, dict) or not all(k in entry for k in ['x', 'y', 'ang']):
                    raise ValueError(f"Entry {i} is not in the correct format. Expected dict with 'x', 'y', 'ang' keys.")
            
            print(f"[INFO] Using velocity sequence with {len(velocity_sequence)} entries")
            for i, entry in enumerate(velocity_sequence):
                print(f"  Sequence {i}: lin_vel_x={entry['x']}, lin_vel_y={entry['y']}, ang_vel_z={entry['ang']}")
        except Exception as e:
            print(f"[ERROR] Failed to parse velocity sequence: {e}")
            print("[INFO] Format should be a JSON array of objects with 'x', 'y', 'ang' keys")
            print("[INFO] Example: '[{\"x\":1.0,\"y\":0.0,\"ang\":0.0},{\"x\":0.0,\"y\":0.0,\"ang\":0.0}]'")
            velocity_sequence = []


    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        
        # Change velocity command if using sequence
        if args_cli.use_velocity_sequence and velocity_sequence and timestep % args_cli.velocity_sequence_interval == 0:
            # Get the next velocity in the sequence (cycling through the list)
            entry = velocity_sequence[current_velocity_index]
            x, y, ang = entry['x'], entry['y'], entry['ang']
            current_velocity_index = (current_velocity_index + 1) % len(velocity_sequence)
            
            print(f"[INFO] Changing velocity command to: x={x}, y={y}, ang={ang}")
                                
            velocity_term = env.unwrapped.command_manager.get_term("base_velocity")
            # Disable heading command to prevent it from overriding angular velocity
            velocity_term.cfg.heading_command = False
            
            # Modify the velocity command directly
            velocity_term.vel_command_b[:, 0] = x  # x velocity
            velocity_term.vel_command_b[:, 1] = y  # y velocity
            velocity_term.vel_command_b[:, 2] = ang  # angular velocity
            
            obs, _ = env.get_observations()

        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # Print information (simplified for readability)
            # if timestep % 10 == 0:  # Only print every 10 steps to reduce output
            #     print(f"Timestep: {timestep}")

            #     # print shape of actions
            #     print(f"Actions shape: {actions.shape}")
            #     # print shape of obs
            #     print(f"Obs shape: {obs.shape}")

            # Log data if enabled
            if args_cli.log_data and timestep % args_cli.log_interval == 0:
                # Get linear velocity from the environment
                lin_vel = None
                if hasattr(env.unwrapped, 'scene'):
                    try:
                        lin_vel = env.unwrapped.scene['robot'].data.root_lin_vel_b.cpu().numpy().tolist()
                    except KeyError:
                        print("Robot entity not found in scene")
                
                # Convert tensors to lists for JSON serialization
                obs_list = obs.cpu().numpy().tolist()
                actions_list = actions.cpu().numpy().tolist()
                
                #     # Access and print linear velocity from the environment
                #     if hasattr(env.unwrapped, 'scene'):
                #         try:
                #             robot = env.unwrapped.scene['robot']
                #             lin_vel = robot.data.root_lin_vel_b
                #             # print shape of lin_vel
                #             print(f"Lin_vel shape: {lin_vel.shape}")
                #             print(f"Actual velocity: x={lin_vel[0, 0]:.2f}, y={lin_vel[0, 1]:.2f}, z={lin_vel[0, 2]:.2f}")
                #         except KeyError:
                #             print("Robot entity not found in scene")
                    # Get command velocity if available

                
                # Store data for this timestep
                timestep_data = {
                    "timestep": timestep,
                    "observations": obs_list,
                    "actions": actions_list,
                    "linear_velocity": lin_vel,

                }
                logged_data.append(timestep_data)
            
            # env stepping
            obs, _, _, _ = env.step(actions)
        
        timestep += 1
        if args_cli.video and timestep == args_cli.video_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Save logged data to JSON file if enabled
    if args_cli.log_data and logged_data:
        import json
        log_path = os.path.join(log_dir, args_cli.log_file)
        with open(log_path, 'w') as f:
            json.dump(logged_data, f, indent=2)
        print(f"[INFO] Saved logged data to {log_path}")
        
        # Generate and save metadata if requested
        if args_cli.save_metadata:
            # Create a simplified metadata structure
            metadata = {
                "description": "Observation terms and data shapes",
                "observation_terms": [],
                "data_shapes": {}
            }

            observation_terms = [[k, v] for k, v in env.cfg.observations.policy.__dict__.items()]
            for val in observation_terms:
                if isinstance(val[1], ObservationTermCfg):
                    # Just add the name of the observation term
                    metadata["observation_terms"].append(val[0])
            
            # Add data shape information
            if len(logged_data) > 0:
                sample_entry = logged_data[0]
                
                # Get shapes of key data elements
                for key, value in sample_entry.items():
                    if key in ["observations", "actions", "linear_velocity"]:
                        if isinstance(value, list):
                            # Determine the shape
                            shape = []
                            current = value
                            while isinstance(current, list) and len(current) > 0:
                                shape.append(len(current))
                                current = current[0]
                            
                            metadata["data_shapes"][key] = shape
            
            # Save metadata
            metadata_path = os.path.join(log_dir, args_cli.metadata_file)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"[INFO] Saved simplified metadata to {metadata_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
