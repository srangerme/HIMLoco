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

PAYLOAD_STAGES = {
    "solo12_v3_1_payload_2_5": (2.5, 1.0, 6.0),
    "solo12_v3_1_payload_5": (5.0, 1.0, 6.0),
    "solo12_v3_1_payload_7_5": (7.5, 1.0, 6.0),
    "solo12_v3_1_payload_10": (10.0, 1.0, 6.0),
    "solo12_v3_1_payload_10_torque_recovery": (10.0, 1.0, 6.0),
    "solo12_v3_1_payload_10_envelope125": (10.0, 1.25, 7.5),
    "solo12_v3_1_payload_10_envelope150": (10.0, 1.5, 9.0),
    "solo12_v3_1_payload_10_envelope200": (10.0, 2.0, 12.0),
}

RS06_STAGES = {
    "solo12_v3_1_rs06_adapt": 0.0,
    "solo12_v3_1_rs06_warmup": 0.0,
    "solo12_v3_1_rs06_intermediate": 0.0,
    "solo12_v3_1_rs06_payload_2_5": 2.5,
    "solo12_v3_1_rs06_payload_5": 5.0,
    "solo12_v3_1_rs06_payload_7_5": 7.5,
    "solo12_v3_1_rs06_payload_10": 10.0,
    "solo12_v3_1_rs06_payload_10_torque_recovery": 10.0,
}

RS03_THIGH_STAGES = {
    "solo12_v3_1_rs03_thigh_payload_10_obstacle_adapt": [0.0, 1.0],
    "solo12_v3_1_rs03_thigh_payload_10_speed20_adapt": [1.0, 2.0],
    "solo12_v3_1_rs03_thigh_payload_10_speed_limit_polish": [1.0, 2.0],
    "solo12_v3_1_rs03_thigh_payload_10_slope_warmup": [0.25, 0.55],
    "solo12_v3_1_rs03_thigh_payload_10_slope_bridge": [0.35, 0.65],
    "solo12_v3_1_rs03_thigh_payload_10_slope_adapt": [0.45, 0.7],
}

DIRECT_CALF_STAGES = {
    "solo12_v3_1_rs03_calf1_payload_10_obstacle_adapt": (195.0, 60.0, 20.0, 13.0, {"hip": 0.520, "thigh": 0.520}),
    "solo12_v3_1_rs03_calf1_payload_10_speed_adapt": (195.0, 60.0, 20.0, 13.0, {"hip": 0.520, "thigh": 0.520}),
    "solo12_v3_1_rs03_calf1_payload_10_slope_warmup": (195.0, 60.0, 20.0, 13.0, {"hip": 0.520, "thigh": 0.520}),
    "solo12_v3_1_rs03_calf1_payload_10_slope_bridge": (195.0, 60.0, 20.0, 13.0, {"hip": 0.520, "thigh": 0.520}),
    "solo12_v3_1_rs03_calf1_payload_10_slope_adapt": (195.0, 60.0, 20.0, 13.0, {"hip": 0.520, "thigh": 0.520}),
    "solo12_v3_1_rs04_calf1_payload_10_obstacle_adapt": (200.0, 120.0, 40.0, 28.5, {"hip": 0.520, "thigh": 1.040}),
    "solo12_v3_1_rs04_calf1_payload_10_speed_adapt": (200.0, 120.0, 40.0, 28.5, {"hip": 0.520, "thigh": 1.040}),
    "solo12_v3_1_rs04_calf1_payload_10_slope_warmup": (200.0, 120.0, 40.0, 28.5, {"hip": 0.520, "thigh": 1.040}),
    "solo12_v3_1_rs04_calf1_payload_10_slope_bridge": (200.0, 120.0, 40.0, 28.5, {"hip": 0.520, "thigh": 1.040}),
    "solo12_v3_1_rs04_calf1_payload_10_slope_adapt": (200.0, 120.0, 40.0, 28.5, {"hip": 0.520, "thigh": 1.040}),
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
    assert set(PAYLOAD_STAGES) <= set(task_registry.task_classes)
    for task, (payload_max, capability_scale, rated_nm) in PAYLOAD_STAGES.items():
        cfg = task_registry.env_cfgs[task]
        assert cfg.domain_rand.payload_mass_range == [0.0, payload_max]
        assert cfg.domain_rand.payload_min_endpoint_probability == 0.15
        assert cfg.domain_rand.payload_max_endpoint_probability == 0.15
        assert cfg.control.torque_speed_envelope_scale == capability_scale
        assert cfg.asset.dof_torque_limit_scale == capability_scale
        assert cfg.control.rated_motor_torque_nm == rated_nm
    assert set(RS06_STAGES) <= set(task_registry.task_classes)
    for task, payload_max in RS06_STAGES.items():
        cfg = task_registry.env_cfgs[task]
        assert cfg.domain_rand.payload_mass_range == [0.0, payload_max]
        assert cfg.control.joint_gear_ratios["calf"] == 3.0
        assert cfg.control.rated_motor_torques_nm["calf"] == 11.0
        assert cfg.control.torque_speed_curves["calf"]["rpm"][-1] == 480.0
        assert cfg.control.torque_speed_curves["calf"]["nm"][-1] == 0.0
        assert cfg.asset.dof_torque_limits["calf"] == 108.0
        assert cfg.asset.body_mass_additions == {"thigh": 0.241}
    assert set(RS03_THIGH_STAGES) <= set(task_registry.task_classes)
    for task, command_range in RS03_THIGH_STAGES.items():
        cfg = task_registry.env_cfgs[task]
        assert cfg.domain_rand.payload_mass_range == [0.0, 10.0]
        assert cfg.commands.ranges.lin_vel_x == command_range
        assert cfg.control.joint_gear_ratios == {
            "hip": 1.0,
            "thigh": 1.0,
            "calf": 3.0,
        }
        assert cfg.control.torque_speed_curves["thigh"]["rpm"][-1] == 195.0
        assert cfg.control.torque_speed_curves["thigh"]["nm"][0] == 60.0
        assert cfg.control.torque_speed_curves["thigh"]["nm"][-1] == 0.0
        assert cfg.control.rated_motor_torques_nm["thigh"] == 20.0
        assert cfg.asset.dof_velocity_limits["thigh"] == 20.420352248333657
        assert cfg.asset.dof_torque_limits["thigh"] == 60.0
        assert cfg.asset.body_mass_additions == {"hip": 0.520, "thigh": 0.241}
        assert cfg.asset.base_mass_addition_kg == 0.0
        assert cfg.domain_rand.armature_ranges["thigh"] == [0.0, 0.00504]
    assert set(DIRECT_CALF_STAGES) <= set(task_registry.task_classes)
    for task, (max_rpm, peak_nm, rated_nm, stall_nm, mass_additions) in DIRECT_CALF_STAGES.items():
        cfg = task_registry.env_cfgs[task]
        assert cfg.control.joint_gear_ratios == {
            "hip": 1.0,
            "thigh": 1.0,
            "calf": 1.0,
        }
        assert cfg.control.torque_speed_curves["calf"]["rpm"][-1] == max_rpm
        assert cfg.control.torque_speed_curves["calf"]["nm"][0] == peak_nm
        assert cfg.control.torque_speed_curves["calf"]["nm"][-1] == 0.0
        assert cfg.control.rated_motor_torques_nm["calf"] == rated_nm
        assert cfg.control.stall_rated_motor_torques_nm["calf"] == stall_nm
        assert cfg.asset.dof_torque_limits["calf"] == peak_nm
        assert cfg.asset.body_mass_additions == mass_additions
        assert cfg.asset.base_mass_addition_kg == 0.0


def validate_runtime():
    args = get_args()
    args.task = "solo12_v3_1_rs03_thigh_payload_10_obstacle_adapt"
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
    assert env.raw_actor_obs_history.shape[0] == 1
    assert env.raw_actor_obs_history.shape[2] == 45
    expected_gears = torch.tensor(
        [1.0, 1.0, 3.0] * 4, device=env.device
    )
    expected_velocity_limits = torch.tensor(
        [43.9823, 20.420352248333657, 16.755160819145562] * 4,
        device=env.device,
    )
    assert torch.allclose(env.joint_gear_ratios, expected_gears)
    assert torch.allclose(
        env.dof_vel_limits, expected_velocity_limits
    ), (env.dof_names, env.dof_vel_limits, expected_velocity_limits)
    assert torch.allclose(
        env.rated_motor_torques_nm,
        torch.tensor([6.0, 20.0, 11.0] * 4, device=env.device),
    )
    assert torch.allclose(
        env.torque_limits,
        torch.tensor([17.0, 60.0, 108.0] * 4, device=env.device),
    )

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
