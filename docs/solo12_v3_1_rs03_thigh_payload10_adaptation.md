# Solo12 v3.1 RS03-thigh +10 kg adaptation

## Hardware model

- Hip: RS02 at 48 V, no external reduction (`1:1`).
- Thigh: RS03 at 48 V, no external reduction (`1:1`).
- Calf: RS06 at 48 V with the existing external `3:1` stage.
- Payload randomization: `[0, 10] kg`, with 50% exact replay at `10 kg`.

The RS03 model uses the official 2026-07-13 48 V peak torque-speed plot,
`60 Nm` peak torque, a conservative `20 Nm` continuous-duty objective, and a
`195 rpm` zero-torque speed endpoint.  The product mass is `0.900 kg`; relative
to the original `0.380 kg` RS02, `0.520 kg` is added to every thigh link.  The
existing `0.241 kg` RS06-for-RS02 calf actuator delta is mounted on that link as
well, so the total fixed thigh-link addition is `0.761 kg`.

Sources:

- https://github.com/RobStride/Product_Information
- https://github.com/RobStride/Product_Information/blob/main/Product%20Literature/RS03/RS03User%20Manual260713.pdf

## Adaptation runs

The finalized predecessor hardware is Hip RS02 `1:1`, Thigh RS02 `3:1`, and
Calf RS06 `3:1`.  Its obstacle, high-speed, and ramp capabilities live in three
separate policies, so the RS03 handoff preserves the same split instead of
forcing one policy to cover every scene.

Obstacle chain (`10900` -> `11100`):

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1_rs03_thigh_payload_10_obstacle_adapt \
  --num_envs 4096 --max_iterations 200 --seed 1 \
  --resume \
  --load_run Sep24_06-14-05_rs06_thigh3_obstacle_replay_10800_plus200 \
  --checkpoint 10900 \
  --run_name rs03_obstacle_adapt_10900_to11100
```

High-speed chain (`11200` -> `11400`):

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1_rs03_thigh_payload_10_speed20_adapt \
  --num_envs 4096 --max_iterations 200 --seed 1 \
  --resume \
  --load_run Sep24_07-33-27_rs06_thigh3_speed20_11100_plus100 \
  --checkpoint 11200 \
  --run_name rs03_speed20_adapt_11200_to11400
```

Slope chain (`11300` -> `11500`):

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1_rs03_thigh_payload_10_slope_adapt \
  --num_envs 4096 --max_iterations 200 --seed 1 \
  --resume \
  --load_run Sep24_06-25-20_rs06_thigh3_slope_adapt_11200_plus100 \
  --checkpoint 11300 \
  --run_name rs03_slope_adapt_11300_to11500
```

All three tasks preserve the policy observation/action layout and therefore
support checkpoint continuation without network surgery.  The earlier
`11420 -> 11620` experiment used the obsolete RS02-thigh `2:1` parent and is
diagnostic-only; it is not a deployment or acceptance candidate.

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
`10.0 kg`.
