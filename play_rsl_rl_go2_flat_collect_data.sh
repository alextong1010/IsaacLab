# REMEMBER TO CHANGE THE custom_velocity_env_cfg.py file to have the correct velocity commands
SCRIPT_PATH="scripts/reinforcement_learning/rsl_rl/play.py"
TASK="Isaac-Velocity-Flat-Unitree-Go2-Play-v0"
NUM_ENVS="1"
VIDEO_LENGTH="2000" # in steps

# Checkpoint for DEFAULT training
# LOAD_CHECKPOINT="./logs/rsl_rl/unitree_go2_flat/2025-03-02_01-15-06/model_9999.pt" 

# Checkpoint for DEFAULT without base linear velocity
# LOAD_CHECKPOINT="./logs/rsl_rl/unitree_go2_flat/2025-02-25_19-58-59/model_9999.pt" 

LOAD_CHECKPOINT="./logs/rsl_rl/unitree_go2_flat/2025-03-10_13-55-55/model_9999.pt"

# Use this to toggle which cuda device gets used for training
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

VELOCITY_SEQUENCE_PATH="velocity_sequence.json"
# VELOCITY_SEQUENCE='[
#   {"x": 1.0, "y": 0.0, "ang": 0.0},
#   {"x": 0.0, "y": 0.0, "ang": 0.0},
#   {"x": -1.0, "y": 0.0, "ang": 0.0},
#   {"x": 0.0, "y": 0.0, "ang": 0.0},
#   {"x": 0.0, "y": 0.0, "ang": 1.0},
#   {"x": 0.0, "y": 0.0, "ang": 0.0},
#   {"x": 0.0, "y": 0.0, "ang": -1.0},
#   {"x": 0.0, "y": 0.0, "ang": 0.0}
# ]'
SEQUENCE_INTERVAL="200"  # Change velocity every 10 timesteps

# Data logging settings
LOG_DATA="true"
LOG_FILE="robot_data.json"
LOG_INTERVAL="1"
SAVE_METADATA="true"
METADATA_FILE="metadata.json"

# Run with velocity sequence from JSON string and data logging
./isaaclab.sh -p "$SCRIPT_PATH" \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --video \
    --video_length "$VIDEO_LENGTH" \
    --checkpoint "$LOAD_CHECKPOINT" \
    --use_velocity_sequence \
    --velocity_sequence_json "$VELOCITY_SEQUENCE_PATH" \
    --velocity_sequence_interval "$SEQUENCE_INTERVAL" \
    --log_data \
    --log_file "$LOG_FILE" \
    --log_interval "$LOG_INTERVAL" \
    --save_metadata \
    --metadata_file "$METADATA_FILE"

# --velocity_sequence_json "$VELOCITY_SEQUENCE" \

# ./isaaclab.sh -p "$SCRIPT_PATH" --task "$TASK" --headless --num_envs "$NUM_ENVS" --video --video_length "$VIDEO_LENGTH" --checkpoint "$LOAD_CHECKPOINT"

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