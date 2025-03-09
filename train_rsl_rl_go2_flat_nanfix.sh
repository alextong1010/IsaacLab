#!/bin/bash

# Change to workflow to be one of {rsl_rl, skrl, sb3, or rl_games}
# Unitree Go2 env officially supports rsl_rl and skrl
SCRIPT_PATH="scripts/reinforcement_learning/rsl_rl/train.py"
TASK="Custom-Isaac-Velocity-Flat-Unitree-Go2-v0"
NUM_ENVS="4"
MAX_ITERATIONS="10000"
SEED="100"
VIDEO_LENGTH="200" # in steps
VIDEO_INTERVAL="2000" # in steps

# Checkpoint parameters
RESUME="True"
LOAD_RUN="2025-03-05_13-55-46"  # Replace with your actual run folder name
LOAD_CHECKPOINT="model_8540.pt"  # Replace with the iteration number to load

# Use this to toggle which cuda device gets used for training
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

# Run the command with specified arguments
./isaaclab.sh -p "$SCRIPT_PATH" --task "$TASK" --num_envs "$NUM_ENVS" --seed "$SEED" \
  --max_iterations "$MAX_ITERATIONS" --video --video_length "$VIDEO_LENGTH" \
  --video_interval "$VIDEO_INTERVAL" --resume "$RESUME" --load_run "$LOAD_RUN" \
  --checkpoint "$LOAD_CHECKPOINT"