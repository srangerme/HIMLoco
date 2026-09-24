"""Plot motor-side torque for every Solo12 joint from an acceptance trace."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("output")
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("--title", default="Solo12 motor-side torque")
    args = parser.parse_args()

    with np.load(args.trace, allow_pickle=False) as trace:
        joint_torque = np.asarray(trace["joint_torque_nm"])
        if joint_torque.ndim == 2:
            joint_torque = joint_torque[:, None, :]
        dt_s = float(np.asarray(trace["dt_s"]).reshape(()))
        names = [str(value) for value in np.asarray(trace["dof_names"]).tolist()]
        ratios = np.asarray(trace["joint_gear_ratios"], dtype=float).reshape(-1)
        rated = np.asarray(trace["rated_motor_torques_nm"], dtype=float).reshape(-1)

    if args.env < 0 or args.env >= joint_torque.shape[1]:
        raise ValueError("--env is outside the trace environment range")
    motor_torque = joint_torque[:, args.env, :] / ratios[None, :]
    time_s = np.arange(motor_torque.shape[0]) * dt_s

    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    for joint_id, (axis, name) in enumerate(zip(axes.flat, names)):
        axis.plot(time_s, motor_torque[:, joint_id], linewidth=0.75, color="#2563eb")
        axis.axhline(rated[joint_id], color="#dc2626", linestyle="--", linewidth=0.8)
        axis.axhline(-rated[joint_id], color="#dc2626", linestyle="--", linewidth=0.8)
        p95 = np.percentile(np.abs(motor_torque[:, joint_id]), 95)
        below = np.mean(np.abs(motor_torque[:, joint_id]) < rated[joint_id]) * 100.0
        axis.set_title(f"{name}  |τ| p95={p95:.2f} Nm, <rated={below:.1f}%", fontsize=9)
        axis.grid(alpha=0.25)
        axis.set_ylabel("Nm")
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    fig.suptitle(args.title + " (blue: torque, red dashed: ±rated)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
