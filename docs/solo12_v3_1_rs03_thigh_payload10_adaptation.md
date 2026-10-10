# Solo12 v3.1 RS03-thigh +10 kg adaptation

## Hardware model

- Hip: RS02 at 48 V, no external reduction (`1:1`).
- Thigh: RS03 at 48 V, no external reduction (`1:1`).
- Calf: RS06 at 48 V with the existing external `3:1` stage.
- Payload randomization: `[0, 10] kg`, with 50% exact replay at `10 kg`.
  This payload is added to the base rigid body at its existing center of mass.

The RS03 model uses the official 2026-07-13 48 V peak torque-speed plot,
`60 Nm` peak torque, a conservative `20 Nm` continuous-duty objective, and a
`195 rpm` zero-torque speed endpoint.  The RS03-minus-RS02 thigh actuator delta
is `0.520 kg` per leg and is added to the hip link that carries the thigh
stator.  The RS06-minus-RS02 calf actuator delta is `0.241 kg` per leg and is
added to the thigh link that carries the calf stator.  No motor-mass delta is
added to the base.  The independent fixed `10 kg` acceptance payload remains
on the base rigid body at its existing center of mass.

Sources:

- https://github.com/RobStride/Product_Information
- https://github.com/RobStride/Product_Information/blob/main/Product%20Literature/RS03/RS03User%20Manual260713.pdf

## Adaptation runs

The finalized predecessor hardware is Hip RS02 `1:1`, Thigh RS02 `3:1`, and
Calf RS06 `3:1`.  Adapt the hardware and payload once with the general obstacle
policy, then use that new RS03 checkpoint as the common parent for the separate
high-speed and ramp capabilities.  This keeps one shared hardware-adaptation
stage without forcing the final capabilities into one policy.

Common RS03 base (`10900` -> `11100`):

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1_rs03_thigh_payload_10_obstacle_adapt \
  --num_envs 4096 --max_iterations 200 --seed 1 \
  --resume \
  --load_run Sep24_06-14-05_rs06_thigh3_obstacle_replay_10800_plus200 \
  --checkpoint 10900 \
  --run_name rs03_obstacle_adapt_10900_to11100
```

High-speed branch from the common RS03 base (`11100` -> `11300`):

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1_rs03_thigh_payload_10_speed20_adapt \
  --num_envs 4096 --max_iterations 200 --seed 1 \
  --resume \
  --load_run Sep29_09-31-42_rs03_obstacle_adapt_10900_to11100 \
  --checkpoint 11100 \
  --run_name rs03_speed20_from_base_11100_to11300
```

The first acceptance pass found contact-driven RS03 thigh overspeed on level-9
uneven terrain.  Continue the speed branch with an 85% soft-speed threshold and
explicit velocity-limit penalty (`11300` -> `11400`):

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1_rs03_thigh_payload_10_speed_limit_polish \
  --num_envs 4096 --max_iterations 100 --seed 1 \
  --resume \
  --load_run Sep29_09-50-19_rs03_speed20_from_base_11100_to11300 \
  --checkpoint 11300 \
  --run_name rs03_speed_limit_polish_11300_to11400
```

Slope branch from the common RS03 base.  Do not jump directly into the final
35--45 degree distribution: the attempted direct run stayed at curriculum
level zero.  Use three stages inside this one capability branch.

Warm up on 20--35 degree ramps (`11100` -> `11200`):

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1_rs03_thigh_payload_10_slope_warmup \
  --num_envs 4096 --max_iterations 100 --seed 1 \
  --resume \
  --load_run Sep29_09-31-42_rs03_obstacle_adapt_10900_to11100 \
  --checkpoint 11100 \
  --run_name rs03_slope_warmup_from_base_11100_to11200
```

Bridge on 28--42 degree ramps (`11200` -> `11300`):

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1_rs03_thigh_payload_10_slope_bridge \
  --num_envs 4096 --max_iterations 100 --seed 1 \
  --resume \
  --load_run Sep29_10-15-31_rs03_slope_warmup_from_base_11100_to11200 \
  --checkpoint 11200 \
  --run_name rs03_slope_bridge_11200_to11300
```

Polish on 35--45 degree ramps (`11300` -> `11400`):

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1_rs03_thigh_payload_10_slope_adapt \
  --num_envs 4096 --max_iterations 100 --seed 1 \
  --resume \
  --load_run Sep29_10-23-16_rs03_slope_bridge_11200_to11300 \
  --checkpoint 11300 \
  --run_name rs03_slope_final_moving_11300_to11400
```

The final slope training command range is `0.45--0.7 m/s`; zero-speed samples
are intentionally excluded because they cannot satisfy the world-X traversal
curriculum.  Static slope acceptance switches to zero only after verified ramp
entry.

The resulting run directories are:

- common base: `Sep29_09-31-42_rs03_obstacle_adapt_10900_to11100`;
- high speed: `Sep29_09-50-19_rs03_speed20_from_base_11100_to11300`;
- speed-limit polish: `Sep29_11-02-51_rs03_speed_limit_polish_11300_to11400`;
- slope warmup: `Sep29_10-15-31_rs03_slope_warmup_from_base_11100_to11200`;
- slope bridge: `Sep29_10-23-16_rs03_slope_bridge_11200_to11300`;
- slope final: `Sep29_10-35-56_rs03_slope_final_moving_11300_to11400`.

All stages preserve the policy observation/action layout and therefore
support checkpoint continuation without network surgery.  Finish and verify
the common base before launching either branch.  The earlier
`11420 -> 11620` experiment used the obsolete RS02-thigh `2:1` parent and is
diagnostic-only; it is not a deployment or acceptance candidate.  The
interrupted direct 35--45 degree run
`Sep29_10-07-44_rs03_slope_from_base_11100_to11300` is also diagnostic-only.

## Historical base-mass redistribution adaptation

After moving the fixed `3.044 kg` actuator delta from the four thigh links to
the base, adapt once with the common obstacle policy and branch from that new
checkpoint.  The objective is a majority-pass capability probe, not a policy
that hides hardware limits by repeatedly optimizing isolated failures.

Final candidate checkpoints:

- common obstacle/base: `Sep29_13-37-06_rs03_body_mass_base_11100_to11200`,
  checkpoint `11200` (100 iterations);
- high-speed/uneven: `Sep29_14-03-54_rs03_body_mass_speed_polish_11300_to11400`,
  checkpoint `11400` (200 branch iterations total);
- slope warmup: `Sep29_14-15-19_rs03_body_mass_slope_warmup_11250_to11300`,
  checkpoint `11300` (100 warmup iterations total);
- slope bridge: `Sep29_14-19-45_rs03_body_mass_slope_bridge_11300_to11400`,
  checkpoint `11400` (100 iterations);
- final 35--45 degree slope: `Sep29_14-27-47_rs03_body_mass_slope_final_11400_to11500`,
  checkpoint `11500` (100 iterations).

The 50-environment stopping criterion is that a clear majority completes each
requested scene.  A few resets, obstacle-entry misses, or impact-driven speed
spikes are retained as actuator capability evidence rather than used to
justify indefinite policy polishing.

## Acceptance gates

After adaptation, repeat fixed `10 kg` rollouts with the matching policy:

- speed policy: plane and continuous uneven level 9 at `1.0`, `1.5`, and
  `2.0 m/s`;
- obstacle policy: 15 cm stairs at `0.7 m/s`, plane stand for 10 s, and
  stairs stand for 10 s after climbing;
- slope policy: 40 and 45 degree slopes at `0.5` and `0.7 m/s`, plus a
  40 degree 10 s stand after climbing.

Obstacle tests must prove entry onto the obstacle before their measurement
window starts.  Flat-approach speed alone is not evidence of stair or ramp
success.

Acceptance traces must report thigh gear ratio `1.0`, thigh hard speed limit
`20.420352248333657 rad/s`, thigh peak torque limit `60 Nm`, and fixed payload
`10.0 kg` on the base rigid body.

## Historical base-mass candidate result

The final 50-environment capability probe uses the common checkpoint `11200`,
speed checkpoint `11400`, and slope checkpoint `11500` listed above.

- Plane `1.0/1.5/2.0 m/s`: zero resets, with mean forward speed
  `0.948/1.381/1.728 m/s`.
- Uneven level 9 `1.0/1.5/2.0 m/s`: `2/6/13` reset environments, leaving
  `48/44/37` environments without a reset; mean speed is
  `0.903/1.265/1.511 m/s`.
- Continuous 15 cm stairs at `0.7 m/s`: all 50 environments enter with zero
  resets and `0.649 m/s` mean speed.
- Plane stand for 10 s: zero resets.
- Stair stand for 10 s: all 50 environments enter with zero resets.
- 40 degree slope at `0.5/0.7 m/s`: all 50 environments enter with zero
  resets; mean speed is `0.562/0.673 m/s`.
- 45 degree slope at `0.5/0.7 m/s`: 49 of 50 environments enter, with
  three reset environments in each scene (four reset events); mean speed is
  `0.367/0.401 m/s`.

All final traces remain inside the modeled dynamic torque-speed torque
envelope, and every motor family's P95 torque remains below its rated torque.
Speed is the hardware boundary: RS03 thigh P95 reaches `194.95 rpm` against a
`195 rpm` endpoint on uneven `2.0 m/s`, with impact/backdrive peaks above the
endpoint on uneven terrain, stairs, and steep slopes.  RS06 calf P95 reaches
about `478.6 rpm` against its `480 rpm` limit on slopes.  The configuration is
therefore majority-capable in simulation, but has essentially no speed margin
for high-speed uneven terrain or steep-slope impacts; that limitation is kept
as a module requirement rather than polished away with further training.

The two historical sections above kept all actuator deltas on the base and are
retained only for comparison.  They are superseded by the link-mounted result
below.

## Link-mounted motor-mass re-adaptation (canonical)

The corrected mass model adds the `0.520 kg` RS03 thigh delta to each hip link
and the `0.241 kg` RS06 calf delta to each thigh link.  Motor delta on the base
is zero; the independent acceptance payload remains `10 kg` on the base.

All stages used 4096 environments.  The accepted runs are:

- base: `Oct10_08-42-15_rs03_rs06_linkmass_base_11200_to11400`, model 11400;
- speed: `Oct10_09-21-47_rs03_rs06_linkmass_speed_11400_to11700`, model 11700;
- slope final: `Oct10_10-59-57_rs03_rs06_linkmass_slope_final_11600_to11800`,
  model 11800.

The 50-environment capability result is:

- plane 1.0/1.5/2.0 m/s: 50/50 without reset, mean speed
  `0.995/1.424/1.785 m/s`;
- level-9 uneven 1.0/1.5/2.0 m/s: `50/48/41` environments without reset,
  mean speed `0.941/1.313/1.557 m/s`;
- continuous 15 cm stairs: 50/50 enter and advance at least 1 m;
- 40 degree slopes: 50/50 at both speeds;
- 45 degree slopes: 49/50 advance at least 1 m at both speeds;
- plane stand: 50/50 without reset; stair stand: 49/50 without reset.

Every selected sample is inside the configured torque-speed envelope.  The
lowest hard-speed compliance is `99.426%` on uneven 2.0 m/s and the lowest
rated/stall-reference compliance is `98.652%` on the same scene.  The
canonical dataset is under
`/tmp/solo12_hardware_test_data_20260924/link_mass_corrected_20261010/03_thigh_rs03_ratio1_calf_rs06_ratio3/10kg`.
