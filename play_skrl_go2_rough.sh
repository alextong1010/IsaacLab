SCRIPT_PATH="scripts/reinforcement_learning/skrl/play.py"
TASK="Isaac-Velocity-Rough-Unitree-Go2-Play-v0"
NUM_ENVS="4"
VIDEO_LENGTH="800" # in steps
# CHECKPOINT_PATH="./logs/skrl/unitree_go2_rough/2025-02-28_00-21-07_ppo_torch/checkpoints/best_agent.pt"
LOAD_CHECKPOINT="./logs/skrl/unitree_go2_rough/2025-02-28_00-21-07_ppo_torch/checkpoints/best_agent.pt"  # Replace with the iteration number to load

# Use this to toggle which cuda device gets used for training
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

./isaaclab.sh -p "$SCRIPT_PATH" --task "$TASK" --headless --num_envs "$NUM_ENVS" --video --video_length "$VIDEO_LENGTH" --checkpoint "$LOAD_CHECKPOINT"