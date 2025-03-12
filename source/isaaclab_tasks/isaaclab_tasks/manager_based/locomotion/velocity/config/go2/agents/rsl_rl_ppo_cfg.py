# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    # clip_actions: float | list | None = None
    # """The clipping value for actions. 
    # If float, all actions are clipped to [-clip_actions, clip_actions].
    # If list, each action dimension is clipped to the corresponding range.
    # If None, then no clipping is done."""
        
    # clip_actions = None
    clip_actions = 20
    # Define per-joint action limits
    # Note: These need to be scaled by the action scale (0.25) from rough_env_cfg.py
    # action_scale = 1
    # clip_actions = [
    #     [-1.0472 * action_scale, 1.0472 * action_scale],    # FL_hip_joint
    #     [-1.0472 * action_scale, 1.0472 * action_scale],    # FR_hip_joint
    #     [-1.0472 * action_scale, 1.0472 * action_scale],    # RL_hip_joint
    #     [-1.0472 * action_scale, 1.0472 * action_scale],    # RR_hip_joint
    #     [-1.5708 * action_scale, 3.4907 * action_scale],    # FL_thigh_joint
    #     [-1.5708 * action_scale, 3.4907 * action_scale],    # FR_thigh_joint
    #     [-0.5236 * action_scale, 4.5379 * action_scale],    # RL_thigh_joint
    #     [-0.5236 * action_scale, 4.5379 * action_scale],    # RR_thigh_joint
    #     [-2.7227 * action_scale, -0.83776 * action_scale],  # FL_calf_joint
    #     [-2.7227 * action_scale, -0.83776 * action_scale],  # FR_calf_joint
    #     [-2.7227 * action_scale, -0.83776 * action_scale],  # RL_calf_joint
    #     [-2.7227 * action_scale, -0.83776 * action_scale],  # RR_calf_joint
    # ]



@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "unitree_go2_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
