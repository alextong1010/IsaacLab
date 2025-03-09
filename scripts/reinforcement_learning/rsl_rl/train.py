# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import json
import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--only_positive_rewards", action="store_true", default=False, 
                    help="Clip rewards to be non-negative")
parser.add_argument("--save_actions", action="store_true", default=False, 
                    help="Save actions to a JSON file after a specified iteration")
parser.add_argument("--save_actions_start_iter", type=int, default=0,
                    help="Iteration to start saving actions")
parser.add_argument("--save_actions_count", type=int, default=240, # 24 steps per env per iteration so 240 is 10 iterations
                    help="Number of actions to save")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Add this class before the main function
class PositiveRewardRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Wrapper for RSL-RL that clips rewards to be non-negative."""
    
    def __init__(self, env, clip_rewards=False):
        super().__init__(env)
        self.clip_rewards = clip_rewards
        
    def step(self, actions):
        # Call the parent step method which properly handles the observation format
        obs, rewards, dones, infos = super().step(actions)
        
        # Only modify the rewards part
        if self.clip_rewards:
            rewards = torch.clamp(rewards, min=0.0)
            
        return obs, rewards, dones, infos

# Add this class after the PositiveRewardRslRlVecEnvWrapper class
class ActionSavingOnPolicyRunner(OnPolicyRunner):
    """Extension of OnPolicyRunner that can save actions to a file in real-time."""
    
    def __init__(self, env, config, save_actions=False, save_actions_start_iter=0, 
                 save_actions_count=240, **kwargs):
        super().__init__(env, config, **kwargs)
        self.save_actions = save_actions
        self.save_actions_start_iter = save_actions_start_iter
        self.save_actions_count = save_actions_count
        self.saved_actions_count = 0
        self.actions_path = None
        
        # Initialize the JSON file if we're saving actions
        if self.save_actions:
            self.actions_path = os.path.join(self.log_dir, "saved_actions.json")
            # Create an empty list in the JSON file
            with open(self.actions_path, 'w') as f:
                json.dump([], f)
            print(f"[INFO] Initialized actions file at {self.actions_path}")
            print(f"[INFO] Will save up to {self.save_actions_count} actions starting from iteration {self.save_actions_start_iter}")
    
    def _append_action_to_json(self, action_data):
        """Append a single action to the JSON file."""
        if not self.actions_path:
            return
            
        # Read the current content
        try:
            with open(self.actions_path, 'r') as f:
                actions = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is empty or doesn't exist, start with empty list
            actions = []
        
        # Convert tensor to list for JSON serialization
        serializable_action = {
            'iteration': action_data['iteration'],
            'step': action_data['step'],
            'actions': action_data['actions'].cpu().numpy().tolist()
        }
        
        # Append the new action
        actions.append(serializable_action)
        
        # Write back to file
        with open(self.actions_path, 'w') as f:
            json.dump(actions, f, indent=2)
    
    def _train_step(self):
        # Execute the original training step
        result = super()._train_step()
        
        # Check if we should save actions from this iteration
        if (self.save_actions and 
            self.current_learning_iteration >= self.save_actions_start_iter and 
            self.saved_actions_count < self.save_actions_count):
            
            # During the rollout phase, save the actions
            for step in range(self.num_steps_per_env):
                # Get the actions from the current batch
                actions = self.batch["actions"][:, step].clone()
                
                # Create action data
                action_data = {
                    'iteration': self.current_learning_iteration,
                    'step': step,  # This resets each iteration as requested
                    'actions': actions
                }
                
                # Save the action to the JSON file immediately
                self._append_action_to_json(action_data)
                self.saved_actions_count += 1
                
                # Stop if we've saved enough actions
                if self.saved_actions_count >= self.save_actions_count:
                    print(f"[INFO] Collected {self.save_actions_count} actions at iteration {self.current_learning_iteration}")
                    break
        
        return result

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    
    # Set the only_positive_rewards flag if specified
    if hasattr(env_cfg, "only_positive_rewards") and args_cli.only_positive_rewards:
        env_cfg.only_positive_rewards = args_cli.only_positive_rewards
        print(f"[INFO] Using only positive rewards: {env_cfg.only_positive_rewards}")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    if hasattr(env_cfg, "only_positive_rewards") and env_cfg.only_positive_rewards:
        env = PositiveRewardRslRlVecEnvWrapper(env, clip_rewards=True)
    else:
        env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    if args_cli.save_actions:
        print(f"[INFO] Will save {args_cli.save_actions_count} actions starting from iteration {args_cli.save_actions_start_iter}")
        runner = ActionSavingOnPolicyRunner(
            env, 
            agent_cfg.to_dict(), 
            log_dir=log_dir, 
            device=agent_cfg.device,
            save_actions=args_cli.save_actions,
            save_actions_start_iter=args_cli.save_actions_start_iter,
            save_actions_count=args_cli.save_actions_count
        )
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
