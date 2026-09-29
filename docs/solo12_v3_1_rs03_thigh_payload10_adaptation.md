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

## Adaptation run

Resume the accepted +10 kg RS02-thigh-2:1 checkpoint `11420` and adapt only the
hardware dynamics while retaining the balanced plane/stair/slope/uneven replay
distribution:

```bash
python3 -m trains.cli train himloco-ada \
  --task solo12_v3_1_rs03_thigh_payload_10_adapt \
  --num_envs 4096 --max_iterations 200 --seed 1 \
  --resume \
  --load_run Sep24_03-05-29_rs06_thigh2_torque_incidence_11320_to11420 \
  --checkpoint 11420 \
  --run_name rs03_thigh_payload10_adapt_11420_to11620
```

The task preserves the policy observation/action layout and therefore supports
checkpoint continuation without network surgery.  Checkpoints are expected at
20-iteration intervals through `11620`.

## Acceptance gates

After adaptation, repeat fixed `10 kg` rollouts for:

- flat ground at `1.0 m/s`;
- continuous uneven level 9 at `1.0 m/s`;
- 15 cm stairs at `0.7 m/s`;
- 45 degree slope at `0.7 m/s`.

Acceptance traces must report thigh gear ratio `1.0`, thigh hard speed limit
`20.420352248333657 rad/s`, thigh peak torque limit `60 Nm`, and fixed payload
`10.0 kg`.
