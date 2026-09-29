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


def _grouped_plot(names, stats, limit, ylabel, title, output, limit_label):
    x = np.arange(len(names))
    width = 0.23
    colors = {"median": "#60a5fa", "p95": "#2563eb", "max": "#1e3a8a"}
    fig, axis = plt.subplots(figsize=(16, 7))
    for offset, key in zip((-1, 0, 1), ("median", "p95", "max")):
        axis.bar(x + offset * width, stats[key], width, label=key, color=colors[key])
    axis.scatter(
        x,
        limit,
        marker="_",
        s=650,
        linewidths=2.5,
        color="#dc2626",
        label=limit_label,
        zorder=5,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=35, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4)
    fig.tight_layout()
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

    start_step = int(round(args.start_seconds / dt_s))
    if start_step < 0 or start_step >= joint_torque.shape[0]:
        raise ValueError("--start-seconds is outside the trace duration")
    joint_torque = joint_torque[start_step:]
    joint_velocity = joint_velocity[start_step:]
    done = done[start_step:]
    base_velocity = base_velocity[start_step:]
    command = command[start_step:]

    motor_torque = joint_torque / ratios[None, None, :]
    motor_speed_rpm = (
        joint_velocity * ratios[None, None, :] * 60.0 / (2.0 * np.pi)
    )
    hard_motor_speed_rpm = hard_joint_speed * ratios * 60.0 / (2.0 * np.pi)
    torque_reference = rated if args.comparison == "rated" else stall
    torque_reference_name = "rated torque" if args.comparison == "rated" else "stall torque"

    torque_stats = _statistics(motor_torque)
    speed_stats = _statistics(motor_speed_rpm)
    _grouped_plot(
        names,
        torque_stats,
        torque_reference,
        "Motor torque magnitude (Nm)",
        f"{args.title}: motor torque vs {torque_reference_name}",
        args.torque_output,
        torque_reference_name,
    )
    _grouped_plot(
        names,
        speed_stats,
        hard_motor_speed_rpm,
        "Motor speed magnitude (rpm)",
        f"{args.title}: motor speed",
        args.speed_output,
        "hard speed limit",
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
        "joints": {},
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
