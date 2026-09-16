"""Validate Solo12 v3.1 HIMLoco integration without updating a policy."""

import os

from isaacgym import gymapi
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args


STAGES = {
    "solo12_v3_1_r0": ([0.0, 0.0], False, [0.0, 0.2]),
    "solo12_v3_1_r1": ([0.0, 10.0], False, [0.0, 0.2]),
    "solo12_v3_1_r2": ([0.0, 20.0], False, [0.0, 0.2]),
    "solo12_v3_1_r3": ([0.0, 30.0], False, [0.0, 0.2]),
    "solo12_v3_1_r4": ([0.0, 30.0], True, [0.0, 0.1]),
    "solo12_v3_1_r5": ([0.0, 30.0], True, [0.0, 0.2]),
    "solo12_v3_1_r6": ([0.0, 40.0], True, [0.0, 0.2]),
}


def validate_configs():
    assert set(STAGES) <= set(task_registry.task_classes)
    for task, (latency_range, randomize_restitution, restitution_range) in STAGES.items():
        cfg = task_registry.env_cfgs[task]
        assert cfg.env.num_actions == 12
        assert len(cfg.init_state.default_joint_angles) == 12
        assert os.path.isfile(cfg.asset.file), cfg.asset.file
        assert cfg.domain_rand.observation_latency_range == latency_range
        assert cfg.domain_rand.randomize_restitution is randomize_restitution
        assert cfg.domain_rand.restitution_range == restitution_range
        assert cfg.domain_rand.per_joint_randomization
        assert cfg.domain_rand.joint_friction_range == [0.0, 0.02]
        assert cfg.domain_rand.joint_damping_range == [0.0, 0.01]
        assert cfg.control.use_torque_speed_envelope
        assert cfg.control.joint_gear_ratios == {
            "hip": 1.0,
            "thigh": 1.0,
            "calf": 2.0,
        }
        assert len(cfg.control.torque_speed_curve_rpm) == len(
            cfg.control.torque_speed_curve_nm
        )
        assert cfg.control.torque_speed_curve_rpm[-1] == 420.0
        assert cfg.control.torque_speed_curve_nm[-1] == 0.0


def validate_runtime():
    args = get_args()
    args.task = "solo12_v3_1_r6"
    args.headless = True
    args.num_envs = 1
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.max_init_terrain_level = 0
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)

    assert env.num_dof == 12, env.num_dof
    assert len(env.dof_names) == 12, env.dof_names
    assert env.feet_indices.numel() == 4, env.feet_indices
    assert env.Kp_factors.shape == (1, 12)
    assert env.Kd_factors.shape == (1, 12)
    assert env.motor_strength_factors.shape == (1, 12)
    assert env.raw_actor_obs_history.shape == (1, 3, 45), env.raw_actor_obs_history.shape
    expected_gears = torch.tensor(
        [1.0, 1.0, 2.0] * 4, device=env.device
    )
    expected_velocity_limits = torch.tensor(
        [44.0, 44.0, 22.0] * 4, device=env.device
    )
    assert torch.equal(env.joint_gear_ratios, expected_gears)
    assert torch.equal(env.dof_vel_limits, expected_velocity_limits)

    obs, privileged_obs = env.reset()
    assert obs.shape == (1, 270)
    assert privileged_obs.shape == (1, 238)
    assert torch.isfinite(obs).all()
    assert torch.isfinite(privileged_obs).all()

    env.dof_vel[:] = 0.0
    zero_speed_limits = env._compute_torque_speed_limits()
    assert torch.equal(zero_speed_limits[0], env.torque_limits)
    env.dof_vel[:] = env.dof_vel_limits
    assert torch.allclose(
        env._compute_torque_speed_limits(),
        torch.zeros_like(env.dynamic_torque_limits),
        atol=1e-4,
    )
    env.dof_vel[:] = -env.dof_vel_limits
    assert torch.allclose(
        env._compute_torque_speed_limits(),
        torch.zeros_like(env.dynamic_torque_limits),
        atol=1e-4,
    )
    env.dof_vel[:] = 0.0
    for _ in range(5):
        obs, privileged_obs, rewards, _, _, _, _ = env.step(
            torch.zeros((1, 12), device=env.device)
        )
        assert torch.isfinite(obs).all()
        assert torch.isfinite(privileged_obs).all()
        assert torch.isfinite(rewards).all()
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    validate_configs()
    validate_runtime()
    print("Solo12 HIMLoco validation passed:", sorted(STAGES))
