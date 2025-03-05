# REMEMBER TO CHANGE THE custom_velocity_env_cfg.py file to have the correct velocity commands

SCRIPT_PATH="scripts/reinforcement_learning/rsl_rl/play.py"
TASK="Custom-Isaac-Velocity-Rough-Unitree-Go2-v0"
NUM_ENVS="4"
VIDEO_LENGTH="800" # in steps

# Checkpoint for DEFAULT training
# LOAD_CHECKPOINT="./logs/rsl_rl/custom_unitree_go2_flat/2025-03-02_01-15-06/model_9999.pt" 

# Checkpoint for DEFAULT without base linear velocity
LOAD_CHECKPOINT="./logs/rsl_rl/custom_unitree_go2_rough/2025-02-27_02-24-32/model_8350.pt" 


# Use this to toggle which cuda device gets used for training
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

./isaaclab.sh -p "$SCRIPT_PATH" --task "$TASK" --headless --num_envs "$NUM_ENVS" --video --video_length "$VIDEO_LENGTH" --checkpoint "$LOAD_CHECKPOINT"

# ./isaaclab.sh -p "$SCRIPT_PATH" \
#     --task "$TASK" \
#     --headless \
#     --num_envs "$NUM_ENVS" \
#     --video \
#     --video_length "$VIDEO_LENGTH" \
#     --checkpoint "$LOAD_CHECKPOINT" \
#     --use_custom_velocity \
#     --lin_vel_x_min -1.0 \
#     --lin_vel_x_max 1.0 \
#     --lin_vel_y_min 0.0 \
#     --lin_vel_y_max 0.0 \
#     --ang_vel_z_min 0.0 \
#     --ang_vel_z_max 0.0 \
#     --heading_min 0.0 \
#     --heading_max 0.0