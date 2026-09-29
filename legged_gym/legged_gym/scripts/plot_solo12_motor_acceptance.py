"""Plot multi-environment Solo12 motor-side torque and speed statistics."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _load_rollout(trace, key):
    values = np.asarray(trace[key], dtype=float)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError(f"{key} must have shape [time, env, joint]")
    return values


def _statistics(values):
    absolute = np.abs(values).reshape(-1, values.shape[-1])
    return {
        "median": np.percentile(absolute, 50, axis=0),
        "p95": np.percentile(absolute, 95, axis=0),
        "max": np.max(absolute, axis=0),
    }


def _time_series_plot(
    names,
    values,
    limit,
    dt_s,
    ylabel,
    title,
    output,
    limit_label,
    unit,
):
    """Render one time-series panel per joint, matching acceptance reports."""
    absolute = np.abs(values)
    per_time = {
        "max": np.max(absolute, axis=1),
        "p95": np.percentile(absolute, 95, axis=1),
        "median": np.percentile(absolute, 50, axis=1),
    }
    within = absolute <= limit[None, None, :]
    per_joint_within = np.mean(within, axis=(0, 1)) * 100.0
    overall_within = float(np.mean(within) * 100.0)
    peak = float(np.max(absolute))
    time_s = np.arange(absolute.shape[0]) * dt_s
    env_count = absolute.shape[1]

    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.reshape(-1)
    colors = {"max": "#cbd5e1", "p95": "#e85d32", "median": "#1f77b4"}
    for joint_id, (axis, name) in enumerate(zip(axes, names)):
        axis.plot(
            time_s,
            per_time["max"][:, joint_id],
            color=colors["max"],
            linewidth=0.7,
            alpha=0.75,
        )
        axis.plot(
            time_s,
            per_time["p95"][:, joint_id],
            color=colors["p95"],
            linewidth=1.0,
        )
        axis.plot(
            time_s,
            per_time["median"][:, joint_id],
            color=colors["median"],
            linewidth=1.1,
        )
        axis.axhline(
            limit[joint_id],
            color="#262626",
            linestyle="--",
            linewidth=1.0,
        )
        axis.set_title(
            f"{name}   <{limit[joint_id]:g} {unit}: "
            f"{per_joint_within[joint_id]:.1f}%",
            fontsize=9,
        )
        axis.grid(alpha=0.22)

    for row in range(4):
        axes[row * 3].set_ylabel(ylabel)
    for axis in axes[-3:]:
        axis.set_xlabel("Time (s)")

    subtitle = (
        f"overall below per-joint {limit_label}: {overall_within:.1f}%; "
        f"peak: {peak:.2f} {unit}"
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=14, y=0.985)
    legend_labels = [
        f"max across {env_count} env{'s' if env_count != 1 else ''}",
        f"P95 across {env_count} env{'s' if env_count != 1 else ''}",
        f"median across {env_count} env{'s' if env_count != 1 else ''}",
        limit_label,
    ]
    fig.legend(
        axes[0].lines[:4],
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.025, 0.035, 0.995, 0.90))
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--torque-output", required=True)
    parser.add_argument("--speed-output", required=True)
    parser.add_argument("--metrics-output")
    parser.add_argument("--comparison", choices=("rated", "stall"), default="rated")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--title", default="Solo12 motor acceptance")
    args = parser.parse_args()

    with np.load(args.trace, allow_pickle=False) as trace:
        joint_torque = _load_rollout(trace, "joint_torque_nm")
        joint_velocity = _load_rollout(trace, "joint_velocity_rad_s")
        dt_s = float(np.asarray(trace["dt_s"]).reshape(()))
        names = [str(value) for value in np.asarray(trace["dof_names"]).tolist()]
        ratios = np.asarray(trace["joint_gear_ratios"], dtype=float).reshape(-1)
        rated = np.asarray(trace["rated_motor_torques_nm"], dtype=float).reshape(-1)
        hard_joint_speed = np.asarray(
            trace["hard_joint_speed_rad_s"], dtype=float
        ).reshape(-1)
        if "stall_motor_torques_nm" in trace:
            stall = np.asarray(trace["stall_motor_torques_nm"], dtype=float).reshape(-1)
        else:
            dynamic_limit = _load_rollout(trace, "dynamic_torque_limit_nm")
            stall = np.max(dynamic_limit, axis=(0, 1)) / ratios
        done = np.asarray(trace["done"]).astype(bool)
        payload = np.asarray(trace["payload_mass_kg"], dtype=float)
        base_velocity = np.asarray(trace["base_lin_vel_m_s"], dtype=float)
        command = np.asarray(trace["command_velocity"], dtype=float)
        root_position = np.asarray(trace["root_position_m"], dtype=float)
        foot_velocity = np.asarray(trace["foot_velocity_m_s"], dtype=float)
        foot_contact = np.asarray(trace["foot_contact"]).astype(bool)
        acceptance_friction = (
            float(np.asarray(trace["acceptance_friction"]).reshape(()))
            if "acceptance_friction" in trace
            else -1.0
        )
        actual_robot_friction = (
            np.asarray(trace["actual_robot_friction"], dtype=float).reshape(-1)
            if "actual_robot_friction" in trace
            else np.asarray([], dtype=float)
        )
        play_terrain = (
            str(np.asarray(trace["play_terrain"]).reshape(()))
            if "play_terrain" in trace
            else "unknown"
        )
        slope_angle_deg = (
            float(np.asarray(trace["slope_angle_deg"]).reshape(()))
            if "slope_angle_deg" in trace
            else 0.0
        )
        obstacle_entry = (
            np.asarray(trace["obstacle_entered"]).astype(bool)
            if "obstacle_entered" in trace
            else None
        )
        obstacle_entry_time_s = (
            float(np.asarray(trace["obstacle_entry_time_s"]).reshape(()))
            if "obstacle_entry_time_s" in trace
            else -1.0
        )
        obstacle_entry_required = (
            bool(np.asarray(trace["obstacle_entry_required"]).reshape(()))
            if "obstacle_entry_required" in trace
            else False
        )

    start_step = int(round(args.start_seconds / dt_s))
    if start_step < 0 or start_step >= joint_torque.shape[0]:
        raise ValueError("--start-seconds is outside the trace duration")
    joint_torque = joint_torque[start_step:]
    joint_velocity = joint_velocity[start_step:]
    done = done[start_step:]
    base_velocity = base_velocity[start_step:]
    command = command[start_step:]
    root_position = root_position[start_step:]
    foot_velocity = foot_velocity[start_step:]
    foot_contact = foot_contact[start_step:]

    motor_torque = joint_torque / ratios[None, None, :]
    motor_speed_rpm = (
        joint_velocity * ratios[None, None, :] * 60.0 / (2.0 * np.pi)
    )
    hard_motor_speed_rpm = hard_joint_speed * ratios * 60.0 / (2.0 * np.pi)
    torque_reference = rated if args.comparison == "rated" else stall
    torque_reference_name = (
        "rated torque"
        if args.comparison == "rated"
        else "manual stall-rated torque"
    )

    torque_stats = _statistics(motor_torque)
    speed_stats = _statistics(motor_speed_rpm)
    progress_x = root_position[-1, :, 0] - root_position[0, :, 0]
    progress_z = root_position[-1, :, 2] - root_position[0, :, 2]
    commanded_distance = np.sum(np.maximum(command[:, :, 0], 0.0), axis=0) * dt_s
    progress_ratio = np.divide(
        progress_x,
        commanded_distance,
        out=np.full_like(progress_x, np.nan),
        where=commanded_distance > 1e-6,
    )
    if play_terrain == "slope":
        slope_rad = np.deg2rad(slope_angle_deg)
        terrain_normal = np.asarray([-np.sin(slope_rad), 0.0, np.cos(slope_rad)])
    else:
        terrain_normal = np.asarray([0.0, 0.0, 1.0])
    normal_speed = np.sum(foot_velocity * terrain_normal, axis=-1)
    tangential_speed = np.sqrt(
        np.maximum(np.sum(foot_velocity ** 2, axis=-1) - normal_speed ** 2, 0.0)
    )
    contact_slip_speed = tangential_speed[foot_contact]
    _time_series_plot(
        names,
        motor_torque,
        torque_reference,
        dt_s,
        "Motor torque magnitude (Nm)",
        f"{args.title}\nMotor-side torque",
        args.torque_output,
        torque_reference_name,
        "Nm",
    )
    _time_series_plot(
        names,
        motor_speed_rpm,
        hard_motor_speed_rpm,
        dt_s,
        "Motor speed magnitude (rpm)",
        f"{args.title}\nMotor-side speed",
        args.speed_output,
        "motor speed limit",
        "rpm",
    )

    metrics = {
        "source_trace": os.path.abspath(args.trace),
        "analysis_start_s": args.start_seconds,
        "analysis_duration_s": joint_torque.shape[0] * dt_s,
        "environments": joint_torque.shape[1],
        "payload_mass_kg": {"min": float(payload.min()), "max": float(payload.max())},
        "command_mean_m_s": np.mean(command, axis=(0, 1)).tolist(),
        "base_velocity_mean_m_s": np.mean(base_velocity, axis=(0, 1)).tolist(),
        "falls_or_resets": int(done.sum()),
        "torque_comparison": args.comparison,
        "contact_friction": {
            "requested": None if acceptance_friction < 0.0 else acceptance_friction,
            "robot_shape_min": (
                None if actual_robot_friction.size == 0 else float(actual_robot_friction.min())
            ),
            "robot_shape_max": (
                None if actual_robot_friction.size == 0 else float(actual_robot_friction.max())
            ),
        },
        "terrain_motion": {
            "terrain": play_terrain,
            "slope_angle_deg": slope_angle_deg,
            "forward_progress_m": {
                "min": float(progress_x.min()),
                "mean": float(progress_x.mean()),
                "max": float(progress_x.max()),
            },
            "vertical_progress_m": {
                "min": float(progress_z.min()),
                "mean": float(progress_z.mean()),
                "max": float(progress_z.max()),
            },
            "forward_progress_ratio": {
                "min": None if np.all(np.isnan(progress_ratio)) else float(np.nanmin(progress_ratio)),
                "mean": None if np.all(np.isnan(progress_ratio)) else float(np.nanmean(progress_ratio)),
                "max": None if np.all(np.isnan(progress_ratio)) else float(np.nanmax(progress_ratio)),
            },
            "contact_foot_slip_m_s": {
                "samples": int(contact_slip_speed.size),
                "median": None if contact_slip_speed.size == 0 else float(np.percentile(contact_slip_speed, 50)),
                "p95": None if contact_slip_speed.size == 0 else float(np.percentile(contact_slip_speed, 95)),
                "max": None if contact_slip_speed.size == 0 else float(np.max(contact_slip_speed)),
            },
        },
        "joints": {},
    }
    if obstacle_entry is not None:
        metrics["obstacle_entry"] = {
            "required": obstacle_entry_required,
            "first_entry_s": obstacle_entry_time_s,
            "entered_environments": int(obstacle_entry[-1].sum()),
            "entry_fraction": float(obstacle_entry[-1].mean()),
        }
    for joint_id, name in enumerate(names):
        metrics["joints"][name] = {
            "gear_ratio": float(ratios[joint_id]),
            "motor_torque_median_nm": float(torque_stats["median"][joint_id]),
            "motor_torque_p95_nm": float(torque_stats["p95"][joint_id]),
            "motor_torque_max_nm": float(torque_stats["max"][joint_id]),
            "torque_reference_nm": float(torque_reference[joint_id]),
            "motor_speed_median_rpm": float(speed_stats["median"][joint_id]),
            "motor_speed_p95_rpm": float(speed_stats["p95"][joint_id]),
            "motor_speed_max_rpm": float(speed_stats["max"][joint_id]),
            "hard_motor_speed_limit_rpm": float(hard_motor_speed_rpm[joint_id]),
        }
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.metrics_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_output)), exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
