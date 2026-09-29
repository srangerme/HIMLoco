# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import subprocess
import tempfile

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0):
    if args.command_switch_time_s is not None and args.command_switch_time_s < 0.0:
        raise ValueError("--command_switch_time_s must be non-negative")
    if args.obstacle_entry_height_m <= 0.0:
        raise ValueError("--obstacle_entry_height_m must be positive")
    if args.post_entry_seconds is not None and args.post_entry_seconds <= 0.0:
        raise ValueError("--post_entry_seconds must be positive")
    if (
        args.minimum_post_entry_progress_m is not None
        and args.minimum_post_entry_progress_m <= 0.0
    ):
        raise ValueError("--minimum_post_entry_progress_m must be positive")
    obstacle_gate_requested = (
        args.require_obstacle_entry
        or args.command_switch_on_obstacle_entry
        or args.analysis_start_after_entry
        or args.post_entry_seconds is not None
        or args.minimum_post_entry_progress_m is not None
    )
    if obstacle_gate_requested and args.play_terrain not in ("stairs", "slope"):
        raise ValueError("Obstacle-entry options require --play_terrain stairs or slope")
    after_command = np.asarray(
        [
            x_vel if args.command_x_after is None else args.command_x_after,
            y_vel if args.command_y_after is None else args.command_y_after,
            yaw_vel if args.command_yaw_after is None else args.command_yaw_after,
        ],
        dtype=np.float32,
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    initial_yaw_rad = np.deg2rad(args.initial_yaw_deg)
    env_cfg.init_state.rot = [
        0.0,
        0.0,
        float(np.sin(0.5 * initial_yaw_rad)),
        float(np.cos(0.5 * initial_yaw_rad)),
    ]
    # override some parameters for testing
    env_cfg.env.num_envs = 1 if args.video_path else (args.num_envs or min(env_cfg.env.num_envs, 50))
    env_cfg.env.enable_camera_sensors = bool(args.video_path)
    if args.acceptance_trace or args.video_path:
        requested_duration = (
            args.video_seconds if args.video_path else args.rollout_seconds
        )
        env_cfg.env.episode_length_s = max(
            env_cfg.env.episode_length_s, requested_duration + 1.0
        )
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = (
        9 if args.max_terrain_level is None else args.max_terrain_level
    )
    if args.terrain_level is not None:
        if args.terrain_level < 0 or args.terrain_level >= env_cfg.terrain.num_rows:
            raise ValueError("--terrain_level must be within the configured terrain rows")
        env_cfg.terrain.fixed_init_level = args.terrain_level
        env_cfg.terrain.max_init_terrain_level = args.terrain_level
    if not args.acceptance_randomization:
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_restitution = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.disturbance = False
        env_cfg.domain_rand.randomize_payload_mass = False
        env_cfg.domain_rand.randomize_com_displacement = False
        env_cfg.domain_rand.randomize_link_mass = False
        env_cfg.domain_rand.randomize_motor_strength = False
        env_cfg.domain_rand.randomize_kp = False
        env_cfg.domain_rand.randomize_kd = False
        env_cfg.domain_rand.randomize_initial_joint_pos = False
        env_cfg.domain_rand.randomize_joint_friction = False
        env_cfg.domain_rand.randomize_joint_damping = False
        env_cfg.domain_rand.observation_latency = False
    if args.acceptance_latency_ms is not None:
        env_cfg.domain_rand.observation_latency = args.acceptance_latency_ms > 0.0
        env_cfg.domain_rand.observation_latency_range = [
            args.acceptance_latency_ms,
            args.acceptance_latency_ms,
        ]
        env_cfg.domain_rand.observation_latency_min_probability = 0.0
        env_cfg.domain_rand.observation_latency_max_probability = 0.0
    if args.acceptance_restitution is not None:
        env_cfg.domain_rand.randomize_restitution = True
        env_cfg.domain_rand.restitution_range = [
            args.acceptance_restitution,
            args.acceptance_restitution,
        ]
    if args.acceptance_friction is not None:
        if args.acceptance_friction <= 0.0:
            raise ValueError("--acceptance_friction must be positive")
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.friction_range = [
            args.acceptance_friction,
            args.acceptance_friction,
        ]
    if args.acceptance_payload_kg is not None:
        if args.acceptance_payload_kg < 0.0:
            raise ValueError("--acceptance_payload_kg must be non-negative")
        env_cfg.domain_rand.randomize_payload_mass = True
        env_cfg.domain_rand.payload_mass_range = [
            args.acceptance_payload_kg,
            args.acceptance_payload_kg,
        ]
        env_cfg.domain_rand.payload_min_endpoint_probability = 0.0
        env_cfg.domain_rand.payload_max_endpoint_probability = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.asset.dof_velocity_limit_scale = args.velocity_limit_scale
    if args.play_terrain == "plane":
        env_cfg.terrain.mesh_type = "plane"
    elif args.play_terrain == "stairs":
        # Explicit acceptance terrain must bypass the training replay mix.
        # Otherwise column 0 is consumed by mixed_plane_probability first.
        env_cfg.terrain.mixed_plane_probability = 0.0
        env_cfg.terrain.mixed_stair_probability = 0.0
        env_cfg.terrain.mixed_slope_probability = 0.0
        env_cfg.terrain.mixed_continuous_uneven_probability = 0.0
        env_cfg.terrain.up_down_stairs = False
        env_cfg.terrain.continuous_up_stairs = True
        env_cfg.terrain.continuous_uneven = False
        env_cfg.terrain.up_down_stair_height = args.stair_step_height
        env_cfg.terrain.spawn_offset_x = 1.5
        env_cfg.terrain.terrain_length = max(
            env_cfg.terrain.terrain_length,
            30.0,
        )
        env_cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        env_cfg.terrain.num_cols = 1
        env_cfg.viewer.pos = [-1.20, -1.80, 0.90]
        env_cfg.viewer.lookat = [0.60, 0.0, 0.35]
    elif args.play_terrain == "uneven":
        # Acceptance-only continuous random height field.  The standard
        # discrete-obstacle generator leaves a 3 m flat spawn platform, which
        # can make a short rollout look like an uneven-ground test even though
        # most of the motion happened on the platform.
        env_cfg.terrain.continuous_uneven = True
        env_cfg.terrain.mixed_plane_probability = 0.0
        env_cfg.terrain.mixed_stair_probability = 0.0
        env_cfg.terrain.mixed_slope_probability = 0.0
        env_cfg.terrain.mixed_continuous_uneven_probability = 0.0
        env_cfg.terrain.spawn_offset_x = 2.0
        env_cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        env_cfg.terrain.num_cols = 1
        env_cfg.viewer.pos = [0.0, -1.40, 0.65]
        env_cfg.viewer.lookat = [0.25, 0.0, 0.20]
    elif args.play_terrain == "slope":
        if not 0.0 < args.slope_angle_deg < 90.0:
            raise ValueError("--slope_angle_deg must be between 0 and 90")
        env_cfg.terrain.mixed_plane_probability = 0.0
        env_cfg.terrain.mixed_stair_probability = 0.0
        env_cfg.terrain.mixed_slope_probability = 0.0
        env_cfg.terrain.mixed_continuous_uneven_probability = 0.0
        env_cfg.terrain.up_down_stairs = False
        env_cfg.terrain.continuous_uneven = False
        env_cfg.terrain.fixed_slope_angle_deg = args.slope_angle_deg
        env_cfg.terrain.continuous_up_stairs = False
        env_cfg.terrain.slope_ramp_start_distance = 2.5
        env_cfg.terrain.spawn_offset_x = 1.0
        env_cfg.terrain.terrain_length = max(
            env_cfg.terrain.terrain_length,
            30.0,
        )
        env_cfg.terrain.num_cols = 1
        # Preserve a 45-degree surface during heightfield-to-mesh conversion.
        env_cfg.terrain.slope_treshold = max(
            env_cfg.terrain.slope_treshold,
            np.tan(np.deg2rad(args.slope_angle_deg)) + 0.05,
        )
        env_cfg.viewer.pos = [-1.50, -1.80, 1.00]
        env_cfg.viewer.lookat = [0.60, 0.0, 0.40]
    elif args.play_terrain != "rough":
        raise ValueError(
            "--play_terrain must be 'rough', 'stairs', 'uneven', 'slope', or 'plane'"
        )
    if env_cfg.terrain.mesh_type not in ("plane", "none"):
        travel_seconds = args.video_seconds if args.video_path else args.rollout_seconds
        world_x_vel = x_vel * np.cos(initial_yaw_rad) - y_vel * np.sin(initial_yaw_rad)
        world_y_vel = x_vel * np.sin(initial_yaw_rad) + y_vel * np.cos(initial_yaw_rad)
        # Stair recordings need more boundary clearance: long oblique runs
        # previously reached the edge before completing the down flight.
        boundary_margin = 8.0 if args.play_terrain == "stairs" else 4.0
        required_length = abs(world_x_vel) * travel_seconds + boundary_margin
        # Terrain origins are centered in Y, so diagonal validation needs
        # clearance on both sides of the center line.
        required_width = 2.0 * abs(world_y_vel) * travel_seconds + boundary_margin
        env_cfg.terrain.terrain_length = max(
            env_cfg.terrain.terrain_length,
            required_length,
        )
        env_cfg.terrain.terrain_width = max(
            env_cfg.terrain.terrain_width,
            required_width,
        )
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    obs = env.get_observations()

    # Runner construction resets the environment while loading a checkpoint,
    # so acceptance geometry must be anchored only after that final reset.
    initial_root_x = env.root_states[:, 0].clone()
    initial_root_z = env.root_states[:, 2].clone()
    terrain_length = float(env_cfg.terrain.terrain_length)
    tile_start_x = torch.floor(initial_root_x / terrain_length) * terrain_length
    if args.play_terrain == "stairs":
        obstacle_start_x = tile_start_x + 3.5
    elif args.play_terrain == "slope":
        obstacle_start_x = tile_start_x + 2.5
    else:
        obstacle_start_x = torch.full(
            (env.num_envs,), float("inf"), device=env.device
        )
    obstacle_entered = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    obstacle_entry_root_x = torch.full_like(initial_root_x, float("nan"))
    failed_after_entry = torch.zeros_like(obstacle_entered)
    first_entry_step = None
    all_entry_step = None


    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    acceptance_trace = None
    if args.acceptance_trace:
        acceptance_trace = {
            "joint_torque_nm": [],
            "commanded_joint_torque_nm": [],
            "joint_velocity_rad_s": [],
            "dynamic_torque_limit_nm": [],
            "base_height_m": [],
            "base_lin_vel_m_s": [],
            "base_ang_vel_rad_s": [],
            "base_quaternion_xyzw": [],
            "root_position_m": [],
            "command_velocity": [],
            "dof_position_rad": [],
            "action": [],
            "foot_position_m": [],
            "foot_velocity_m_s": [],
            "foot_height_m": [],
            "foot_contact": [],
            "terrain_level": [],
            "terrain_type": [],
            "done": [],
            "payload_mass_kg": [],
            "obstacle_entered": [],
        }

    video_process = None
    video_camera = None
    video_path = None
    video_subtitle_file = None
    video_width = 1280
    video_height = 720
    video_stride = max(1, round(1.0 / (args.video_fps * env.dt)))
    if args.video_path:
        video_camera = env.camera_handle
        video_path = os.path.abspath(args.video_path)
        video_dir = os.path.dirname(video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        ffmpeg_env = os.environ.copy()
        ffmpeg_env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
        if args.play_terrain == "slope":
            scene_line = (
                f"Scene  continuous uphill slope {args.slope_angle_deg:g} deg"
            )
        elif args.play_terrain == "stairs":
            scene_line = (
                "Scene  continuous uphill stairs "
                f"{args.stair_step_height * 100.0:g} cm"
            )
        elif args.play_terrain == "uneven":
            level = args.terrain_level if args.terrain_level is not None else 9
            scene_line = f"Scene  continuous uneven level {level}"
        else:
            scene_line = f"Scene  {args.play_terrain}"
        if args.command_switch_on_obstacle_entry:
            hold_s = args.post_entry_seconds
            hold_text = "after entry" if hold_s is None else f"for {hold_s:g} s"
            command_lines = (
                f"Approach  {x_vel:g} m/s until obstacle entry\n"
                f"Hold  {after_command[0]:g} m/s {hold_text}"
            )
        elif args.command_switch_time_s is None:
            command_lines = f"Command  {x_vel:g} m/s"
        else:
            hold_s = max(args.video_seconds - args.command_switch_time_s, 0.0)
            command_lines = (
                f"Approach  {x_vel:g} m/s for {args.command_switch_time_s:g} s\n"
                f"Hold  {after_command[0]:g} m/s for {hold_s:g} s"
            )
        checkpoint_label = "latest" if args.checkpoint is None else str(args.checkpoint)
        payload_label = (
            "training default"
            if args.acceptance_payload_kg is None
            else f"{args.acceptance_payload_kg:g} kg"
        )
        friction_label = (
            "asset default"
            if args.acceptance_friction is None
            else f"{args.acceptance_friction:g}"
        )
        subtitle_text = (
            f"Solo12 v3.1  model {checkpoint_label}  payload {payload_label}\n"
            "Hip RS02 1-to-1 | Thigh RS03 1-to-1 | Calf RS06 3-to-1\n"
            f"{scene_line} | friction {friction_label} | seed {args.seed}\n"
            f"{command_lines}"
        )
        video_subtitle_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        )
        video_subtitle_file.write(subtitle_text)
        video_subtitle_file.close()
        subtitle_filter = (
            "drawtext="
            "fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            f"textfile={video_subtitle_file.name}:"
            "x=24:y=24:fontsize=24:fontcolor=white:line_spacing=6:"
            "box=1:boxcolor=black@0.58:boxborderw=12"
        )
        video_process = subprocess.Popen(
            [
                "/usr/bin/ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "rgba",
                "-video_size", f"{video_width}x{video_height}",
                "-framerate", str(args.video_fps), "-i", "-",
                "-an", "-vf", subtitle_filter,
                "-c:v", "libx264", "-preset", "medium",
                "-crf", "20", "-pix_fmt", "yuv420p", video_path,
            ],
            stdin=subprocess.PIPE,
            env=ffmpeg_env,
        )

    if args.video_path:
        num_steps = int(np.ceil(args.video_seconds / env.dt))
    elif args.acceptance_trace:
        num_steps = int(np.ceil(args.rollout_seconds / env.dt))
    else:
        num_steps = 10 * int(env.max_episode_length)
    for i in range(num_steps):
        if (
            args.command_switch_time_s is not None
            and i * env.dt >= args.command_switch_time_s
        ):
            active_command = after_command
        else:
            active_command = (x_vel, y_vel, yaw_vel)
        env.commands[:, 0] = float(active_command[0])
        env.commands[:, 1] = float(active_command[1])
        env.commands[:, 2] = float(active_command[2])
        if args.command_switch_on_obstacle_entry and obstacle_entered.any():
            env.commands[obstacle_entered, 0] = float(after_command[0])
            env.commands[obstacle_entered, 1] = float(after_command[1])
            env.commands[obstacle_entered, 2] = float(after_command[2])
        actions = policy(obs.detach())
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())
        entered_before_step = obstacle_entered.clone()
        entered_this_step = (
            (env.root_states[:, 0] >= obstacle_start_x)
            & (env.root_states[:, 2] - initial_root_z >= args.obstacle_entry_height_m)
        )
        newly_entered = entered_this_step & ~obstacle_entered
        obstacle_entry_root_x[newly_entered] = env.root_states[newly_entered, 0]
        obstacle_entered |= entered_this_step
        if first_entry_step is None and obstacle_entered.any():
            first_entry_step = i + 1
            print(
                "Verified obstacle entry at "
                f"{first_entry_step * env.dt:.3f} s "
                f"(base-height rise >= {args.obstacle_entry_height_m:.3f} m)"
            )
        if all_entry_step is None and obstacle_entered.all():
            all_entry_step = i + 1
            print(
                "Verified all environments entered the obstacle by "
                f"{all_entry_step * env.dt:.3f} s"
            )
        failed_after_entry |= dones & entered_before_step

        if video_process is not None and i % video_stride == 0:
            env.gym.fetch_results(env.sim, True)
            # Camera sensors use the simulation frame here. Rough-terrain
            # actors are placed at non-zero terrain origins, so track the
            # absolute root pose rather than terrain-relative coordinates.
            robot_xyz = env.root_states[0, :3].detach().cpu().numpy()
            camera_position = np.asarray(env_cfg.viewer.pos).copy()
            camera_target = np.asarray(env_cfg.viewer.lookat).copy()
            camera_position[:2] += robot_xyz[:2]
            camera_target[:2] += robot_xyz[:2]
            camera_height_delta = robot_xyz[2] - env_cfg.rewards.base_height_target
            camera_position[2] += camera_height_delta
            camera_target[2] += camera_height_delta
            env.gym.step_graphics(env.sim)
            env.gym.set_camera_location(
                video_camera,
                env.envs[0],
                gymapi.Vec3(*camera_position),
                gymapi.Vec3(*camera_target),
            )
            env.gym.render_all_camera_sensors(env.sim)
            frame = env.gym.get_camera_image(
                env.sim, env.envs[0], video_camera, gymapi.IMAGE_COLOR
            )
            frame = np.asarray(frame, dtype=np.uint8).reshape(
                video_height, video_width, 4
            )
            video_process.stdin.write(frame.tobytes())

        if acceptance_trace is not None:
            acceptance_trace["joint_torque_nm"].append(
                env.actuator_torques.detach().cpu().numpy().copy()
            )
            acceptance_trace["commanded_joint_torque_nm"].append(
                env.commanded_actuator_torques.detach().cpu().numpy().copy()
            )
            acceptance_trace["joint_velocity_rad_s"].append(
                env.dof_vel.detach().cpu().numpy().copy()
            )
            acceptance_trace["dynamic_torque_limit_nm"].append(
                env.dynamic_torque_limits.detach().cpu().numpy().copy()
            )
            acceptance_trace["base_height_m"].append(
                env._get_base_heights().detach().cpu().numpy().copy()
            )
            acceptance_trace["base_lin_vel_m_s"].append(
                env.base_lin_vel.detach().cpu().numpy().copy()
            )
            acceptance_trace["base_ang_vel_rad_s"].append(
                env.base_ang_vel.detach().cpu().numpy().copy()
            )
            acceptance_trace["base_quaternion_xyzw"].append(
                env.base_quat.detach().cpu().numpy().copy()
            )
            acceptance_trace["root_position_m"].append(
                env.root_states[:, :3].detach().cpu().numpy().copy()
            )
            acceptance_trace["command_velocity"].append(
                env.commands[:, :3].detach().cpu().numpy().copy()
            )
            acceptance_trace["dof_position_rad"].append(
                env.dof_pos.detach().cpu().numpy().copy()
            )
            acceptance_trace["action"].append(
                env.actions.detach().cpu().numpy().copy()
            )
            acceptance_trace["foot_position_m"].append(
                env.feet_pos.detach().cpu().numpy().copy()
            )
            acceptance_trace["foot_velocity_m_s"].append(
                env.feet_vel.detach().cpu().numpy().copy()
            )
            acceptance_trace["foot_height_m"].append(
                env._get_feet_heights().detach().cpu().numpy().copy()
            )
            acceptance_trace["foot_contact"].append(
                (env.contact_forces[:, env.feet_indices, 2] > 1.0)
                .detach().cpu().numpy().copy()
            )
            if hasattr(env, "terrain_levels"):
                terrain_level = env.terrain_levels
            else:
                terrain_level = torch.full(
                    (env.num_envs,), -1, dtype=torch.long, device=env.device
                )
            acceptance_trace["terrain_level"].append(
                terrain_level.detach().cpu().numpy().copy()
            )
            if hasattr(env, "terrain_types"):
                terrain_type = env.terrain_types
            else:
                terrain_type = torch.full(
                    (env.num_envs,), -1, device=env.device, dtype=torch.long
                )
            acceptance_trace["terrain_type"].append(
                terrain_type.detach().cpu().numpy().copy()
            )
            acceptance_trace["done"].append(dones.detach().cpu().numpy().copy())
            acceptance_trace["payload_mass_kg"].append(
                env.payload.detach().cpu().numpy().copy()
            )
            acceptance_trace["obstacle_entered"].append(
                obstacle_entered.detach().cpu().numpy().copy()
            )

        if (
            all_entry_step is not None
            and args.post_entry_seconds is not None
            and (i + 1 - all_entry_step) * env.dt >= args.post_entry_seconds
        ):
            break

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if not args.video_path and acceptance_trace is None and i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif not args.video_path and acceptance_trace is None and i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

    if acceptance_trace is not None:
        trace_path = os.path.abspath(args.acceptance_trace)
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        stall_motor_torques_nm = np.full(env.num_dof, np.nan, dtype=float)
        stall_motor_torque_sources = np.full(env.num_dof, "", dtype="U96")
        stall_rated_by_family = env_cfg.control.stall_rated_motor_torques_nm
        for dof_id, dof_name in enumerate(env.dof_names):
            for joint_family, stall_rated_nm in stall_rated_by_family.items():
                if joint_family == "joint" or joint_family in dof_name:
                    stall_motor_torques_nm[dof_id] = float(stall_rated_nm)
                    stall_motor_torque_sources[dof_id] = (
                        f"{joint_family} official manual stall-overload rated row"
                    )
                    break
        if np.isnan(stall_motor_torques_nm).any():
            missing = [
                env.dof_names[i]
                for i in np.flatnonzero(np.isnan(stall_motor_torques_nm))
            ]
            raise ValueError(
                "Missing manual stall-rated torque for joints: " + ", ".join(missing)
            )
        np.savez_compressed(
            trace_path,
            joint_torque_nm=np.asarray(acceptance_trace["joint_torque_nm"]),
            commanded_joint_torque_nm=np.asarray(
                acceptance_trace["commanded_joint_torque_nm"]
            ),
            joint_velocity_rad_s=np.asarray(acceptance_trace["joint_velocity_rad_s"]),
            dynamic_torque_limit_nm=np.asarray(acceptance_trace["dynamic_torque_limit_nm"]),
            base_height_m=np.asarray(acceptance_trace["base_height_m"]),
            base_lin_vel_m_s=np.asarray(acceptance_trace["base_lin_vel_m_s"]),
            base_ang_vel_rad_s=np.asarray(acceptance_trace["base_ang_vel_rad_s"]),
            base_quaternion_xyzw=np.asarray(acceptance_trace["base_quaternion_xyzw"]),
            root_position_m=np.asarray(acceptance_trace["root_position_m"]),
            command_velocity=np.asarray(acceptance_trace["command_velocity"]),
            dof_position_rad=np.asarray(acceptance_trace["dof_position_rad"]),
            action=np.asarray(acceptance_trace["action"]),
            foot_position_m=np.asarray(acceptance_trace["foot_position_m"]),
            foot_velocity_m_s=np.asarray(acceptance_trace["foot_velocity_m_s"]),
            foot_height_m=np.asarray(acceptance_trace["foot_height_m"]),
            foot_contact=np.asarray(acceptance_trace["foot_contact"]),
            terrain_level=np.asarray(acceptance_trace["terrain_level"]),
            terrain_type=np.asarray(acceptance_trace["terrain_type"]),
            done=np.asarray(acceptance_trace["done"]),
            payload_mass_kg=np.asarray(acceptance_trace["payload_mass_kg"]),
            obstacle_entered=np.asarray(acceptance_trace["obstacle_entered"]),
            obstacle_entry_time_s=np.asarray(
                -1.0 if first_entry_step is None else first_entry_step * env.dt
            ),
            obstacle_all_entry_time_s=np.asarray(
                -1.0 if all_entry_step is None else all_entry_step * env.dt
            ),
            obstacle_start_world_x_m=obstacle_start_x.detach().cpu().numpy(),
            obstacle_entry_height_m=np.asarray(args.obstacle_entry_height_m),
            obstacle_entry_required=np.asarray(args.require_obstacle_entry),
            obstacle_entry_root_x_m=obstacle_entry_root_x.detach().cpu().numpy(),
            minimum_post_entry_progress_m=np.asarray(
                -1.0
                if args.minimum_post_entry_progress_m is None
                else args.minimum_post_entry_progress_m
            ),
            dt_s=np.asarray(env.dt),
            dof_names=np.asarray(env.dof_names),
            foot_names=np.asarray(env.feet_names),
            velocity_limit_scale=np.asarray(args.velocity_limit_scale),
            torque_speed_envelope_scale=np.asarray(
                env_cfg.control.torque_speed_envelope_scale
            ),
            rated_motor_torque_nm=np.asarray(
                env_cfg.control.rated_motor_torque_nm
            ),
            rated_motor_torques_nm=env.rated_motor_torques_nm.detach().cpu().numpy(),
            joint_gear_ratios=env.joint_gear_ratios.detach().cpu().numpy(),
            hard_joint_speed_rad_s=env.dof_vel_limits.detach().cpu().numpy(),
            stall_motor_torques_nm=stall_motor_torques_nm,
            stall_motor_torque_sources=stall_motor_torque_sources,
            base_height_target_m=np.asarray(env_cfg.rewards.base_height_target),
            acceptance_randomization=np.asarray(args.acceptance_randomization),
            acceptance_payload_kg=np.asarray(
                -1.0 if args.acceptance_payload_kg is None else args.acceptance_payload_kg
            ),
            acceptance_latency_ms=np.asarray(
                -1.0 if args.acceptance_latency_ms is None else args.acceptance_latency_ms
            ),
            acceptance_restitution=np.asarray(
                -1.0 if args.acceptance_restitution is None else args.acceptance_restitution
            ),
            acceptance_friction=np.asarray(
                -1.0 if args.acceptance_friction is None else args.acceptance_friction
            ),
            actual_robot_friction=np.asarray(
                env.friction_coeffs.detach().cpu().numpy()
            ),
            play_terrain=np.asarray(args.play_terrain),
            slope_angle_deg=np.asarray(args.slope_angle_deg),
            command_switch_time_s=np.asarray(
                -1.0 if args.command_switch_time_s is None else args.command_switch_time_s
            ),
            command_after=np.asarray(after_command),
        )
        print("Saved actuator acceptance trace:", trace_path)
        requested_plots = args.motor_torque_plot or args.motor_speed_plot
        if requested_plots:
            if not args.motor_torque_plot or not args.motor_speed_plot:
                raise ValueError(
                    "--motor_torque_plot and --motor_speed_plot must be provided together"
                )
            plot_command = [
                os.environ.get("PYTHON", "python"),
                os.path.join(os.path.dirname(__file__), "plot_solo12_motor_acceptance.py"),
                trace_path,
                "--torque-output", os.path.abspath(args.motor_torque_plot),
                "--speed-output", os.path.abspath(args.motor_speed_plot),
                "--comparison", args.torque_comparison,
                "--start-seconds", str(
                    all_entry_step * env.dt
                    if args.analysis_start_after_entry and all_entry_step is not None
                    else args.analysis_start_seconds
                ),
                "--title", args.acceptance_title,
            ]
            if args.motor_metrics:
                plot_command.extend(
                    ["--metrics-output", os.path.abspath(args.motor_metrics)]
                )
            subprocess.run(plot_command, check=True)

    if video_process is not None:
        video_process.stdin.close()
        return_code = video_process.wait()
        if video_subtitle_file is not None:
            os.unlink(video_subtitle_file.name)
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
        print("Saved play video:", video_path)

    if args.require_obstacle_entry:
        entry_fraction = obstacle_entered.float().mean().item()
        print(f"Obstacle entry gate: {entry_fraction:.1%} of environments entered")
        if not obstacle_entered.all():
            raise RuntimeError(
                "Obstacle entry gate failed: at least one environment never climbed "
                f"{args.obstacle_entry_height_m:.3f} m above its spawn height"
            )
        if failed_after_entry.any():
            raise RuntimeError("Obstacle entry gate failed: robot reset after entering")
    if args.minimum_post_entry_progress_m is not None:
        post_entry_progress = env.root_states[:, 0] - obstacle_entry_root_x
        print(
            "Post-entry progress gate: minimum "
            f"{post_entry_progress.min().item():.3f} m, required "
            f"{args.minimum_post_entry_progress_m:.3f} m"
        )
        if (post_entry_progress < args.minimum_post_entry_progress_m).any():
            raise RuntimeError(
                "Post-entry progress gate failed: at least one environment did not "
                f"advance {args.minimum_post_entry_progress_m:.3f} m after entry"
            )

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=args.command_x, y_vel=args.command_y, yaw_vel=args.command_yaw)
