# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

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

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


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
    env = RslRlVecEnvWrapper(env)

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
    
    # Create a counter to print height only every N steps
    print_interval = 20
    print_counter = 0
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            # Print base height every N steps
            print_counter += 1
            if print_counter >= print_interval:
                # Get the robot's base height
                robot_base = env.unwrapped.scene["robot"]
                base_heights = robot_base.data.root_pos_w[:, 2]
                # Print the heights (first few environments)
                print(f"[INFO] Robot base heights: {base_heights[:4].cpu().numpy()} (showing first 5 envs)")
                print_counter = 0
                
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
