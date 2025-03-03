#!/bin/bash

# Change to workflow to be one of {rsl_rl, skrl, sb3, or rl_games}
# Unitree Go2 env officially supports rsl_rl and skrl
SCRIPT_PATH="scripts/reinforcement_learning/rsl_rl/train.py"
TASK="Custom-Isaac-Velocity-Rough-Unitree-Go2-v0"
NUM_ENVS="4096"
MAX_ITERATIONS="10000"
SEED="100"
VIDEO_LENGTH="200" # in steps
VIDEO_INTERVAL="10000" # in steps

# Use this to toggle which cuda device gets used for training
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1


# Run the command with specified arguments
./isaaclab.sh -p "$SCRIPT_PATH" --task "$TASK" --num_envs "$NUM_ENVS" --seed "$SEED" --headless --max_iterations "$MAX_ITERATIONS" --video --video_length "$VIDEO_LENGTH" --video_interval "$VIDEO_INTERVAL"