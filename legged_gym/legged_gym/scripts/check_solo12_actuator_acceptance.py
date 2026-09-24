"""Check Solo12 v3.1 actuator limits from a post-training rollout trace."""

import argparse
import json

import numpy as np


# RS02 manual (2026-07-13), stall overload test. Values are actuator-output
# torque, not Solo calf-joint torque. The stall limits are more conservative
# than the rotating test; unlisted values use the next higher torque point.
RS02_OVERLOAD_BANDS = (
    (6.0, 7.0, 867.0),
    (7.0, 9.0, 174.0),
    (9.0, 11.0, 40.0),
    (11.0, 13.0, 22.0),
    (13.0, 15.0, 10.0),
    (15.0, 17.0, 6.0),
)
RS06_OVERLOAD_BANDS = (
    (11.0, 13.0, 1360.0),
    (13.0, 15.0, 280.0),
    (15.0, 18.0, 70.0),
    (18.0, 23.0, 21.0),
    (23.0, 28.0, 11.0),
    (28.0, 32.0, 8.0),
    (32.0, 34.0, 6.0),
    (34.0, 36.0, 5.0),
)


def _as_rollouts(values, name):
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape [time, env, joint] or [time, joint]")
    return values


def _duration_metrics(mask, done, dt_s):
    """Return total, longest continuous, and largest per-episode exposure."""
    total_s = float(mask.sum() * dt_s)
    max_continuous_s = 0.0
    max_episode_s = 0.0
    for env_id in range(mask.shape[1]):
        for joint_id in range(mask.shape[2]):
            continuous_s = 0.0
            episode_s = 0.0
            for step in range(mask.shape[0]):
                if mask[step, env_id, joint_id]:
                    continuous_s += dt_s
                    episode_s += dt_s
                    max_continuous_s = max(max_continuous_s, continuous_s)
                else:
                    continuous_s = 0.0
                if done[step, env_id]:
                    max_episode_s = max(max_episode_s, episode_s)
                    continuous_s = 0.0
                    episode_s = 0.0
            max_episode_s = max(max_episode_s, episode_s)
    return {
        "total_s": total_s,
        "max_continuous_s": max_continuous_s,
        "max_episode_accumulated_s": max_episode_s,
    }


def _task_metrics(trace):
    dt_s = float(np.asarray(trace["dt_s"]).reshape(()))
    start = min(int(round(1.0 / dt_s)), len(trace["done"]) - 1)
    velocity = np.asarray(trace["base_lin_vel_m_s"])[start:]
    command = np.asarray(trace["command_velocity"])[start:]
    error_xy = velocity[..., :2] - command[..., :2]
    tracking = np.exp(-np.sum(np.square(error_xy), axis=-1) / 0.25)
    return {
        "tracking_reward": float(np.mean(tracking)),
        "mean_forward_velocity_m_s": float(np.mean(velocity[..., 0])),
        "falls": int(np.asarray(trace["done"])[start:].astype(bool).sum()),
    }


def check_trace(
    trace,
    hard_hit_fraction,
    hard_hit_duration_s,
    envelope_tolerance_nm,
    raised_limit_trace=None,
    max_tracking_delta=0.05,
    max_forward_velocity_delta=0.05,
    rated_torque_nm=None,
    min_rated_fraction=0.95,
):
    joint_torque = _as_rollouts(trace["joint_torque_nm"], "joint_torque_nm")
    joint_velocity = _as_rollouts(
        trace["joint_velocity_rad_s"], "joint_velocity_rad_s"
    )
    if joint_torque.shape != joint_velocity.shape:
        raise ValueError("torque and velocity shapes differ")

    steps, num_envs, num_joints = joint_torque.shape
    done = np.asarray(trace["done"] if "done" in trace else np.zeros((steps, num_envs)))
    if done.ndim == 1:
        done = done[:, None]
    if done.shape != (steps, num_envs):
        raise ValueError("done must have shape [time, env]")
    done = done.astype(bool)
    dt_s = float(np.asarray(trace["dt_s"]).reshape(()))
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    if "dof_names" in trace:
        dof_names = [str(name) for name in np.asarray(trace["dof_names"]).tolist()]
    else:
        if num_joints != 12:
            raise ValueError("dof_names is required unless the trace has 12 Solo joints")
        dof_names = [f"leg_{joint}" for leg in range(4) for joint in ("hip", "thigh", "calf")]
    if len(dof_names) != num_joints:
        raise ValueError("dof_names length does not match trace")

    gear_ratio = np.asarray(
        trace["joint_gear_ratios"]
        if "joint_gear_ratios" in trace
        else [2.0 if "calf" in name else 1.0 for name in dof_names],
        dtype=float,
    ).reshape(-1)
    if gear_ratio.size != num_joints:
        raise ValueError("joint_gear_ratios length does not match trace")
    motor_torque = np.abs(joint_torque) / gear_ratio[None, None, :]
    motor_speed = np.abs(joint_velocity) * gear_ratio[None, None, :]

    if rated_torque_nm is not None:
        rated_torques_nm = np.full(num_joints, rated_torque_nm, dtype=float)
    elif "rated_motor_torques_nm" in trace:
        rated_torques_nm = np.asarray(trace["rated_motor_torques_nm"], dtype=float).reshape(-1)
    else:
        legacy_rated = float(np.asarray(trace["rated_motor_torque_nm"]).reshape(())) \
            if "rated_motor_torque_nm" in trace else 6.0
        rated_torques_nm = np.full(num_joints, legacy_rated, dtype=float)
    if rated_torques_nm.size != num_joints:
        raise ValueError("rated_motor_torques_nm length does not match trace")
    below_rated = motor_torque < rated_torques_nm[None, None, :]
    per_joint_rated_fraction = np.mean(below_rated, axis=(0, 1))
    rated_passed = bool(
        np.mean(below_rated) >= min_rated_fraction
        and np.all(per_joint_rated_fraction >= min_rated_fraction)
    )
    rated = {
        "rated_motor_torques_nm": {
            name: float(value) for name, value in zip(dof_names, rated_torques_nm)
        },
        "minimum_required_fraction": min_rated_fraction,
        "comparison": "strictly_less_than",
        "overall_fraction": float(np.mean(below_rated)),
        "per_joint_fraction": {
            name: float(fraction)
            for name, fraction in zip(dof_names, per_joint_rated_fraction)
        },
        "passed": rated_passed,
    }

    overload = []
    overload_passed = True
    peak_torques_nm = np.asarray([36.0 if "calf" in name else 17.0 for name in dof_names])
    for joint_id, name in enumerate(dof_names):
        bands = RS06_OVERLOAD_BANDS if "calf" in name else RS02_OVERLOAD_BANDS
        for lower_nm, upper_nm, allowed_s in bands:
            metrics = _duration_metrics(
                ((motor_torque[..., joint_id:joint_id + 1] > lower_nm)
                 & (motor_torque[..., joint_id:joint_id + 1] <= upper_nm)),
                done,
                dt_s,
            )
            metrics.update({
                "joint": name,
                "motor_torque_band_nm": f"({lower_nm}, {upper_nm}]",
                "allowed_s": allowed_s,
                "passed": metrics["max_continuous_s"] <= allowed_s
                and metrics["max_episode_accumulated_s"] <= allowed_s,
            })
            overload_passed &= metrics["passed"]
            overload.append(metrics)

    peak_exceeded = bool(np.any(
        motor_torque > peak_torques_nm[None, None, :] + envelope_tolerance_nm
    ))
    overload_passed &= not peak_exceeded

    if "hard_joint_speed_rad_s" in trace:
        hard_joint_speed = np.asarray(trace["hard_joint_speed_rad_s"], dtype=float).reshape(-1)
    else:
        velocity_limit_scale = float(
            np.asarray(trace["velocity_limit_scale"]).reshape(())
            if "velocity_limit_scale" in trace else 1.0
        )
        hard_joint_speed = 44.0 * velocity_limit_scale / gear_ratio
    if hard_joint_speed.size != num_joints:
        raise ValueError("hard_joint_speed_rad_s length does not match trace")
    normalized_speed = np.abs(joint_velocity) / hard_joint_speed[None, None, :]
    near_hard_limit = normalized_speed >= 0.90
    hard_limit_hit = normalized_speed >= 0.99
    hard_metrics = _duration_metrics(hard_limit_hit, done, dt_s)
    hard_metrics.update(
        {
            "near_90_percent_fraction": float(np.mean(near_hard_limit)),
            "near_99_percent_fraction": float(np.mean(hard_limit_hit)),
        }
    )
    contact_guard_passed = (
        hard_metrics["near_99_percent_fraction"] <= hard_hit_fraction
        and hard_metrics["max_continuous_s"] <= hard_hit_duration_s
    )
    raised_limit_ab = {"available": False, "passed": None}
    hard_passed = contact_guard_passed
    if raised_limit_trace is not None:
        baseline_task = _task_metrics(trace)
        raised_task = _task_metrics(raised_limit_trace)
        tracking_delta = abs(
            raised_task["tracking_reward"] - baseline_task["tracking_reward"]
        )
        forward_velocity_delta = abs(
            raised_task["mean_forward_velocity_m_s"]
            - baseline_task["mean_forward_velocity_m_s"]
        )
        raised_limit_ab = {
            "available": True,
            "baseline": baseline_task,
            "raised_limit": raised_task,
            "tracking_reward_absolute_delta": tracking_delta,
            "mean_forward_velocity_absolute_delta_m_s": forward_velocity_delta,
            "allowed_tracking_delta": max_tracking_delta,
            "allowed_forward_velocity_delta_m_s": max_forward_velocity_delta,
            "passed": tracking_delta <= max_tracking_delta
            and forward_velocity_delta <= max_forward_velocity_delta,
        }
        hard_passed = raised_limit_ab["passed"]

    envelope = {"available": False, "passed": None}
    envelope_passed = True
    if "dynamic_torque_limit_nm" in trace:
        dynamic_limit = _as_rollouts(
            trace["dynamic_torque_limit_nm"], "dynamic_torque_limit_nm"
        )
        if dynamic_limit.shape != joint_torque.shape:
            raise ValueError("dynamic torque limit shape differs from torque shape")
        violation = np.abs(joint_torque) > dynamic_limit + envelope_tolerance_nm
        envelope = {
            "available": True,
            "violation_fraction": float(np.mean(violation)),
            "maximum_excess_nm": float(
                np.maximum(np.abs(joint_torque) - dynamic_limit, 0.0).max()
            ),
            "passed": not bool(np.any(violation)),
        }
        envelope_passed = envelope["passed"]

    return {
        "source_conditions": "RS02 stall and RS06 rotating overload limits from manufacturer manuals",
        "joint_gear_ratios": {
            name: float(value) for name, value in zip(dof_names, gear_ratio)
        },
        "transmission_efficiency": 1.0,
        "rated_torque": rated,
        "overload": {
            "bands": overload,
            "above_scaled_peak": peak_exceeded,
            "peak_motor_torques_nm": {
                name: float(value) for name, value in zip(dof_names, peak_torques_nm)
            },
            "passed": overload_passed,
        },
        "torque_speed_envelope": envelope,
        "hard_speed_limit": {
            **hard_metrics,
            "allowed_hit_fraction": hard_hit_fraction,
            "allowed_continuous_s": hard_hit_duration_s,
            "contact_guard_passed": contact_guard_passed,
            "passed": hard_passed,
            "raised_limit_ab_test": raised_limit_ab,
        },
        "overall_passed": rated_passed and overload_passed and envelope_passed and hard_passed,
        "samples": steps * num_envs * num_joints,
        "motor_peak_speed_rad_s": float(motor_speed.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="NPZ produced by play.py --acceptance_trace")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--max-hard-hit-fraction", type=float, default=0.001)
    parser.add_argument("--max-hard-hit-duration-s", type=float, default=0.02)
    parser.add_argument("--envelope-tolerance-nm", type=float, default=1e-4)
    parser.add_argument("--raised-limit-trace", help="Matching rollout with a raised hard velocity limit")
    parser.add_argument("--max-tracking-delta", type=float, default=0.05)
    parser.add_argument("--max-forward-velocity-delta", type=float, default=0.05)
    parser.add_argument(
        "--rated-torque-nm",
        type=float,
        help="Override the trace's rated torque; defaults to trace metadata or 6 Nm.",
    )
    parser.add_argument("--min-rated-fraction", type=float, default=0.95)
    args = parser.parse_args()

    with np.load(args.trace, allow_pickle=False) as trace:
        if args.raised_limit_trace:
            with np.load(args.raised_limit_trace, allow_pickle=False) as raised_trace:
                report = check_trace(
                    trace,
                    args.max_hard_hit_fraction,
                    args.max_hard_hit_duration_s,
                    args.envelope_tolerance_nm,
                    raised_trace,
                    args.max_tracking_delta,
                    args.max_forward_velocity_delta,
                    args.rated_torque_nm,
                    args.min_rated_fraction,
                )
        else:
            report = check_trace(
                trace,
                args.max_hard_hit_fraction,
                args.max_hard_hit_duration_s,
                args.envelope_tolerance_nm,
                rated_torque_nm=args.rated_torque_nm,
                min_rated_fraction=args.min_rated_fraction,
            )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
