"""Check progress, gait quality, and stability from a Solo12 rollout trace."""

import argparse
import json

import numpy as np


def _longest_true_s(mask, done, dt_s):
    longest = 0.0
    for env_id in range(mask.shape[1]):
        current = 0.0
        for step in range(mask.shape[0]):
            current = current + dt_s if mask[step, env_id] else 0.0
            longest = max(longest, current)
            if done[step, env_id]:
                current = 0.0
    return longest


def _inverse_rotate_xyzw(quaternion, vector):
    q_xyz = -quaternion[..., :3]
    q_w = quaternion[..., 3:4]
    uv = np.cross(q_xyz, vector)
    uuv = np.cross(q_xyz, uv)
    return vector + 2.0 * (q_w * uv + uuv)


def _swing_peak_heights(foot_height, contact, done, dt_s):
    peaks = [[[] for _ in range(foot_height.shape[2])] for _ in range(foot_height.shape[1])]
    min_steps = max(2, int(round(0.04 / dt_s)))
    for env_id in range(foot_height.shape[1]):
        for foot_id in range(foot_height.shape[2]):
            current = []
            for step in range(foot_height.shape[0]):
                if not contact[step, env_id, foot_id]:
                    current.append(foot_height[step, env_id, foot_id])
                else:
                    if len(current) >= min_steps:
                        peaks[env_id][foot_id].append(float(np.max(current)))
                    current = []
                if done[step, env_id]:
                    current = []
    return peaks


def check_trace(trace, args):
    required = (
        "base_lin_vel_m_s", "base_quaternion_xyzw", "root_position_m",
        "command_velocity", "foot_position_m", "foot_velocity_m_s",
        "foot_height_m", "foot_contact", "done", "dt_s", "foot_names",
    )
    missing = [key for key in required if key not in trace]
    if missing:
        raise ValueError(f"trace is missing fields: {', '.join(missing)}")

    dt_s = float(np.asarray(trace["dt_s"]).reshape(()))
    start = min(int(round(args.warmup_s / dt_s)), len(trace["done"]) - 1)
    velocity = np.asarray(trace["base_lin_vel_m_s"])[start:]
    command = np.asarray(trace["command_velocity"])[start:]
    quaternion = np.asarray(trace["base_quaternion_xyzw"])[start:]
    root_position = np.asarray(trace["root_position_m"])[start:]
    foot_position = np.asarray(trace["foot_position_m"])[start:]
    foot_velocity = np.asarray(trace["foot_velocity_m_s"])[start:]
    foot_height = np.asarray(trace["foot_height_m"])[start:]
    contact = np.asarray(trace["foot_contact"])[start:].astype(bool)
    done = np.asarray(trace["done"])[start:].astype(bool)
    foot_names = [str(name) for name in np.asarray(trace["foot_names"]).tolist()]

    commanded_forward = np.maximum(np.abs(command[..., 0]), 1e-6)
    forward_ratio = velocity[..., 0] / commanded_forward
    moving_command = commanded_forward >= args.min_command_m_s
    stalled = moving_command & (velocity[..., 0] < args.stall_speed_m_s)
    longest_stall_s = _longest_true_s(stalled, done, dt_s)
    env_mean_ratio = np.sum(forward_ratio * moving_command, axis=0) / np.maximum(
        np.sum(moving_command, axis=0), 1
    )
    env_falls = np.sum(done, axis=0)
    env_pass = (
        (env_mean_ratio >= args.min_speed_ratio)
        & (env_falls <= args.max_falls_per_env)
    )

    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch_abs_deg = np.abs(np.rad2deg(pitch))

    swing_peaks = _swing_peak_heights(foot_height, contact, done, dt_s)
    foot_peak_medians = []
    for foot_id in range(len(foot_names)):
        values = [value for env in swing_peaks for value in env[foot_id]]
        foot_peak_medians.append(float(np.median(values)) if values else 0.0)
    left_peaks = [foot_peak_medians[i] for i, name in enumerate(foot_names) if name.startswith("FL") or name.startswith("RL")]
    right_peaks = [foot_peak_medians[i] for i, name in enumerate(foot_names) if name.startswith("FR") or name.startswith("RR")]
    clearance_asymmetry_m = abs(float(np.mean(left_peaks)) - float(np.mean(right_peaks)))
    clearance_spread_m = max(foot_peak_medians) - min(foot_peak_medians)

    duty_factor = np.mean(contact, axis=(0, 1))
    left_duty = [duty_factor[i] for i, name in enumerate(foot_names) if name.startswith("FL") or name.startswith("RL")]
    right_duty = [duty_factor[i] for i, name in enumerate(foot_names) if name.startswith("FR") or name.startswith("RR")]
    duty_asymmetry = abs(float(np.mean(left_duty)) - float(np.mean(right_duty)))

    foot_relative_world = foot_position - root_position[:, :, None, :]
    base_quaternion = quaternion[:, :, None, :]
    foot_body = _inverse_rotate_xyzw(base_quaternion, foot_relative_world)
    expected_side = np.asarray([1.0 if name.startswith("FL") or name.startswith("RL") else -1.0 for name in foot_names])
    signed_lateral = foot_body[..., 1] * expected_side[None, None, :]
    stance_signed_lateral = signed_lateral[contact]
    mean_stance_half_width_m = float(np.mean(stance_signed_lateral))
    stance_half_width_per_foot_m = [
        float(np.mean(signed_lateral[..., foot_id][contact[..., foot_id]]))
        for foot_id in range(len(foot_names))
    ]
    stance_width_spread_m = max(stance_half_width_per_foot_m) - min(
        stance_half_width_per_foot_m
    )
    center_crossing_fraction = float(np.mean(stance_signed_lateral < 0.0))

    stance_planar_speed = np.linalg.norm(foot_velocity[..., :2], axis=-1)[contact]
    mean_stance_slip_m_s = float(np.mean(stance_planar_speed))

    env_pass_fraction = float(np.mean(env_pass))
    checks = {
        "environment_success": env_pass_fraction >= args.min_env_pass_fraction,
        "stall": longest_stall_s <= args.max_stall_s,
        "swing_clearance": min(foot_peak_medians) >= args.min_swing_peak_m,
        "pitch": float(np.mean(pitch_abs_deg)) <= args.max_mean_abs_pitch_deg
        and float(np.percentile(pitch_abs_deg, 95)) <= args.max_p95_abs_pitch_deg,
        "left_right_symmetry": clearance_asymmetry_m <= args.max_clearance_asymmetry_m
        and clearance_spread_m <= args.max_clearance_spread_m
        and duty_asymmetry <= args.max_duty_asymmetry
        and stance_width_spread_m <= args.max_stance_width_spread_m,
        "stance_width": min(stance_half_width_per_foot_m) >= args.min_stance_half_width_m
        and center_crossing_fraction <= args.max_center_crossing_fraction,
        "stance_slip": mean_stance_slip_m_s <= args.max_stance_slip_m_s,
    }
    return {
        "overall_passed": all(checks.values()),
        "checks": checks,
        "environment_pass_fraction": env_pass_fraction,
        "environment_mean_forward_speed_ratio": env_mean_ratio.tolist(),
        "falls_per_environment": env_falls.astype(int).tolist(),
        "stall_fraction": float(np.mean(stalled)),
        "longest_continuous_stall_s": longest_stall_s,
        "mean_abs_pitch_deg": float(np.mean(pitch_abs_deg)),
        "p95_abs_pitch_deg": float(np.percentile(pitch_abs_deg, 95)),
        "foot_names": foot_names,
        "median_swing_peak_height_m": foot_peak_medians,
        "swing_clearance_left_right_asymmetry_m": clearance_asymmetry_m,
        "swing_clearance_per_foot_spread_m": clearance_spread_m,
        "foot_contact_duty_factor": duty_factor.tolist(),
        "contact_duty_left_right_asymmetry": duty_asymmetry,
        "mean_stance_half_width_m": mean_stance_half_width_m,
        "stance_half_width_per_foot_m": stance_half_width_per_foot_m,
        "stance_half_width_spread_m": stance_width_spread_m,
        "center_crossing_fraction": center_crossing_fraction,
        "mean_stance_slip_m_s": mean_stance_slip_m_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--output")
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--min-command-m-s", type=float, default=0.2)
    parser.add_argument("--min-speed-ratio", type=float, default=0.6)
    parser.add_argument("--stall-speed-m-s", type=float, default=0.1)
    parser.add_argument("--max-stall-s", type=float, default=2.0)
    parser.add_argument("--max-falls-per-env", type=int, default=0)
    parser.add_argument("--min-env-pass-fraction", type=float, default=0.9)
    parser.add_argument("--min-swing-peak-m", type=float, default=0.04)
    parser.add_argument("--max-mean-abs-pitch-deg", type=float, default=4.0)
    parser.add_argument("--max-p95-abs-pitch-deg", type=float, default=10.0)
    parser.add_argument("--max-clearance-asymmetry-m", type=float, default=0.02)
    parser.add_argument("--max-clearance-spread-m", type=float, default=0.02)
    parser.add_argument("--max-duty-asymmetry", type=float, default=0.08)
    parser.add_argument("--min-stance-half-width-m", type=float, default=0.08)
    parser.add_argument("--max-stance-width-spread-m", type=float, default=0.04)
    parser.add_argument("--max-center-crossing-fraction", type=float, default=0.01)
    parser.add_argument("--max-stance-slip-m-s", type=float, default=0.15)
    args = parser.parse_args()
    with np.load(args.trace, allow_pickle=False) as trace:
        report = check_trace(trace, args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
