# Solo12 v3.1 direct-calf capability study

## Purpose

Train and evaluate two no-external-reduction calf variants under a fixed 10 kg
acceptance payload.  The target is a majority-pass policy that exposes motor
capability requirements in the requested scenes, not a broadly robust policy
or repeated optimization of isolated failures.

Both variants retain RS02 hip actuators at 1:1 and RS03 thigh actuators at
1:1.  The calf variants are:

- RS03 at 1:1: 60 Nm peak, 195 rpm endpoint, 20 Nm moving-rated reference,
  and 13 Nm stall-rated reference.
- RS04 at 1:1: 120 Nm peak, 200 rpm endpoint, 40 Nm moving-rated reference,
  and 28.5 Nm stall-rated reference.

The external ratio is 1.0 in both variants.  The torque-speed envelopes are
digitized from the official 2026-07-13 RobStride manuals.

## Mass model

The URDF is the all-RS02 mass baseline.  Actuator mass deltas follow the link
that carries each stator: the thigh actuator delta is added to the hip link,
and the calf actuator delta is added to the thigh link.  Values below are per
leg:

- RS03-thigh + RS03-calf: `hip +0.520 kg`, `thigh +0.520 kg`.
- RS03-thigh + RS04-calf: `hip +0.520 kg`, `thigh +1.040 kg`.

The fixed motor-mass addition on the base is zero.  Acceptance adds a separate
fixed `10.0 kg` payload at the existing base center of mass.

Because output inertia is not published, the direct-calf armature ranges are
estimated by removing the former external 3:1 reflection (`/ 9`) from the
RS06 range and scaling by actuator mass:

- RS03 calf: `[0.00725, 0.01530] kg m^2`.
- RS04 calf: `[0.01140, 0.02410] kg m^2`.

This assumption must be retained with the final capability data.

## Training structure

Each motor starts independently from the accepted RS03-thigh + RS06-calf
common checkpoint `Sep29_13-37-06_rs03_body_mass_base_11100_to11200`, model
`11200`.

For each motor:

1. common obstacle/base adaptation: 200 iterations;
2. high-speed plane/uneven branch: 200 iterations from the common checkpoint;
3. slope warmup 20--35 degrees: 100 iterations from the common checkpoint;
4. slope bridge 28--42 degrees: 100 iterations;
5. final slope branch 35--45 degrees: 100 iterations.

This fixed budget is intentionally modest.  Once the requested scenes pass in
a clear majority, failures remain in the dataset as capability evidence.

## Acceptance matrix

Every scene uses 50 environments for the NPZ trace and plots, and one
environment for the subtitled video.  Friction is fixed at 1.2 and payload is
fixed at 10 kg.

The 13 scenes are plane and level-9 uneven terrain at 1.0/1.5/2.0 m/s,
continuous 15 cm stairs at 0.7 m/s, continuous 40 and 45 degree slopes at
0.5/0.7 m/s, plane stand for 10 s, and 15 cm stair stand for 10 s after entry.

Moving torque plots use per-motor rotating-rated references.  Static torque
plots use the manual's stall-rated references.  All scenes include motor-side
speed plots and preserve raw 50-environment traces.

## Historical base-mounted-delta run and result record

RS03 runs:

- base: `Oct10_03-28-09_rs03_calf1_base_11200_to11400`, model 11400;
- speed: `Oct10_03-47-23_rs03_calf1_speed_11400_to11600`, model 11600;
- first slope final: `Oct10_04-33-17_rs03_calf1_slope_final_11600_to11700`;
- accepted slope: `Oct10_04-56-18_rs03_calf1_slope_extra_11700_to11900`, model 11900.

The first RS03 slope model passed only 20/50 environments on the 45 degree,
0.7 m/s case.  One bounded 200-iteration continuation raised the four slope
cases to 50/50, 50/50, 50/50, and 49/50 for 40 degree 0.5/0.7 m/s and 45
degree 0.5/0.7 m/s respectively.  No further optimization was performed.

RS04 runs:

- base: `Oct10_05-15-37_rs04_calf1_base_11200_to11400`, model 11400;
- speed: `Oct10_05-31-56_rs04_calf1_speed_11400_to11600`, model 11600;
- first slope final: `Oct10_06-04-44_rs04_calf1_slope_final_11600_to11700`;
- accepted slope: `Oct10_06-26-07_rs04_calf1_slope_extra_11700_to11900`, model 11900.

The first RS04 slope model passed 18/50 and 6/50 environments on the 45
degree 0.5 and 0.7 m/s cases.  One bounded 200-iteration continuation raised
the four slope cases to 50/50, 50/50, 50/50, and 49/50.  No further
optimization was performed.

Both final configurations meet the requested majority-pass criterion in all
13 scenes.  RS03 has materially tighter continuous and stall-rated torque
margin; RS04 improves calf torque margin but adds 2.080 kg more base mass and
increases thigh demand.  Actual actuator output remained inside the configured
torque-speed envelope in all selected traces.  Rare hard-speed-limit samples
remain, with the largest incidence on level-9 uneven terrain at 2.0 m/s.

The canonical archive is
`/tmp/solo12_hardware_test_data_20260924`, under configurations
`04_thigh_rs03_ratio1_calf_rs03_ratio1` and
`05_thigh_rs03_ratio1_calf_rs04_ratio1`.  Each scene contains the 50-environment
trace, metrics, torque and speed plots, plus a separately recorded one-
environment subtitled video.

The section above retains the earlier result where actuator deltas were kept
on the base.  It is superseded by the corrected link-mounted result below.

## Link-mounted motor-mass re-adaptation (canonical)

The corrected configurations keep the independent 10 kg payload on the base
while placing each thigh actuator delta on its hip link and each calf actuator
delta on its thigh link.  All training stages used 4096 environments.

RS03-calf accepted runs:

- base: `Oct10_12-19-47_rs03_rs03_linkmass_base_11400_to11600`, model 11600;
- speed: `Oct10_12-59-20_rs03_rs03_linkmass_speed_11600_to11900`, model 11900;
- slope final: `Oct10_14-37-36_rs03_rs03_linkmass_slope_final_11800_to12100`,
  model 12100.

RS03-calf 50-environment result:

- plane 1.0/1.5/2.0 m/s: 50/50 without reset;
- level-9 uneven 1.0/1.5/2.0 m/s: `48/48/45` without reset;
- stairs and both 40 degree cases: 50/50;
- 45 degree 0.5/0.7 m/s: `50/49` advance at least 1 m;
- both static cases: 50/50 without reset.

RS04-calf accepted runs:

- base: `Oct10_15-55-31_rs03_rs04_linkmass_base_11400_to11600`, model 11600;
- speed: `Oct10_16-34-47_rs03_rs04_linkmass_speed_11600_to11900`, model 11900;
- slope final: `Oct10_18-11-53_rs03_rs04_linkmass_slope_final_11800_to12100`,
  model 12100.

RS04-calf 50-environment result:

- plane 1.0/1.5/2.0 m/s: 50/50 without reset;
- level-9 uneven 1.0/1.5/2.0 m/s: `49/45/43` without reset;
- stairs: 49/50 advance at least 1 m;
- all four slope cases: 50/50 advance at least 1 m;
- both static cases: 50/50 without reset.

All selected RS03- and RS04-calf samples are inside their configured dynamic
torque-speed envelopes.  The corrected archive is
`/tmp/solo12_hardware_test_data_20260924/link_mass_corrected_20261010`, with
13 traces, 13 videos, 13 metrics files, and 26 plots per configuration.
