#!/bin/bash

# Change to workflow to be one of {rsl_rl, skrl, sb3, or rl_games}
# Unitree Go2 env officially supports rsl_rl and skrl
SCRIPT_PATH="scripts/reinforcement_learning/rsl_rl/train.py"
TASK="Isaac-Velocity-Flat-Unitree-Go2-v0"
NUM_ENVS="4096"
MAX_ITERATIONS="10000"
SEED="100"
VIDEO_LENGTH="200" # in steps
VIDEO_INTERVAL="10000" # in steps
# ONLY_POSITIVE_REWARDS="--only_positive_rewards"
# Or comment it out to use normal rewards
ONLY_POSITIVE_REWARDS=""

# # Action saving parameters
# SAVE_ACTIONS="--save_actions"
# SAVE_ACTIONS_START_ITER="1"
# SAVE_ACTIONS_COUNT="48"

# Use this to toggle which cuda device gets used for training
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

# Run the command with specified arguments
./isaaclab.sh -p "$SCRIPT_PATH" --task "$TASK" --num_envs "$NUM_ENVS" \
--seed "$SEED" --headless --max_iterations "$MAX_ITERATIONS" --video --video_length "$VIDEO_LENGTH" \
--video_interval "$VIDEO_INTERVAL" $ONLY_POSITIVE_REWARDS $SAVE_ACTIONS \
# --save_actions_start_iter "$SAVE_ACTIONS_START_ITER" --save_actions_count "$SAVE_ACTIONS_COUNT"