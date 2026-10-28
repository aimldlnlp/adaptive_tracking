# Control design and evaluation

## Tracking loop

The Panda receives seven bounded joint torque commands: 87 Nm for joints 1–4 and 12 Nm for joints 5–7. Cartesian position gains are 450 N/m and 45 Ns/m; orientation gains are 35 and 8. A kinematic Jacobian maps task forces to joint torques. Nominal gravity compensation and a posture term complete the command. The controller does not implement full operational-space inertia compensation or acceleration feedforward.

Simulation steps are 2 ms; control steps are 20 ms. References include circle, figure-eight, and tilted-ellipse motion with fixed orientation. Payload changes modify the plant's mass and inertia while preserving state. External pushes are applied to the end-effector. Neither event truth nor changed payload mass enters the estimator.

## Compensation

Integral compensation uses gain 2560 and anti-windup. The residual observer filters a projected nominal inverse-dynamics residual at 8 Hz. These settings were selected on six calibration cases before the earlier thirty-case benchmark, then frozen for the robustness study.

The GRU has 11,427 parameters and 48 hidden units. Twelve feature samples at 50 Hz span 220 ms between the oldest and newest observations. Training uses 48 nominal-controller episodes, with 12 validation episodes and up to 35 epochs. Each inference call uses a fresh recent-history window. Predictions are scaled to force, clipped at ±35 N per axis, and smoothed as `0.65 * previous + 0.35 * prediction`.

Its supervised target is payload gravity minus applied push, not complete arm-dynamics identification. Weights remain fixed during deployment. Training on nominal-controller data can create a deployment-distribution gap when the compensator changes the robot's motion.

## Paired robustness study

Eighteen development cases balance three disturbance families and three trajectories, with two cases per combination. Each case runs all four frozen controllers under:

| Condition | Joint-velocity noise standard deviation | Actuator delay |
|---|---:|---:|
| Nominal | 0.002 rad/s | 0 ms |
| Increased noise | 0.010 rad/s | 0 ms |
| Delay 20 ms | 0.002 rad/s | 20 ms |
| Delay 40 ms | 0.002 rad/s | 40 ms |

Delay is a control-step FIFO after torque clipping, initialized with zero torque at startup. The estimator sees issued commands, not delayed actuator truth. These tests include startup effects and do not isolate mid-episode actuator degradation. Noise and delay levels are experimental stress conditions, not measured Panda hardware specifications.

Position RMSE is computed after the first active disturbance, then averaged across cases. Torque-rate RMS uses the norm across joints of finite-differenced applied torque divided by 0.02 s, then averages per-episode RMS values. Torque amplitudes are bounded; torque slew is not constrained.

Tracking loss means exceeding 15 mm after the event. Final settling requires a final contiguous in-threshold segment lasting at least 0.4 seconds. A transient recovery followed by another loss does not count as final settling. Undefined recovery times remain undefined. Numerical failures retain partial traces and receive no full-episode performance score.

All 288 reported rollouts completed numerically. At 40 ms, all 72 controller/case combinations failed settling. Positive sampled joint-limit margins and zero sampled contacts do not establish continuous-time safety or acceptable oscillation.

## Evidence and limits

`results/robustness/` contains the frozen experiment manifest, case definitions, episode metrics, summary, and paired comparisons. Bootstrap intervals are exploratory and conditional on one checkpoint and the selected development cases. They do not measure training-seed variability.

Cleanup of reporting and media tools changes script hashes relative to the historical manifest; controller physics and archived measurements are preserved. New runs save their own source hashes. Raw traces can be regenerated with the supplied checkpoint and config, but exact floating-point parity across platforms is not guaranteed.

Recorded end-effector positions come from cached MuJoCo state, while saved joint poses follow the integration update. At the 12-second snapshot of demonstration case 00, forward kinematics of the saved pose differs from the recorded position by 0.124 mm without delay and 2.188 mm at 40 ms delay. Historical metrics remain internally consistent with the recorded positions. Correct this sampling issue and rerun the benchmark before making finer accuracy claims.

The video replays case 00, a figure-eight with a 1.286 kg added payload at 7.533 s and no external push. Its two chapters compare 0 ms and 40 ms delay as separate runs, not one continuously changing episode. A four-second opening is followed by two 24-second runs at approximately real-time speed and an eight-second aggregate results card. The complete four-condition benchmark remains in the accompanying figure and measurements.

Media composition tools and large generated runs are not part of the public source. The repository reproduces simulation, training, evaluation, and interactive state playback; it does not reproduce the exact edited video's layout.
