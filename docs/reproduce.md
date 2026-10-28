# Reproduce the experiments

Start with the installation and pinned model checkout in the root README. Python 3.12, MuJoCo 3.6.0, PyTorch 2.2.0 and NumPy 1.26.x are the reference versions. A captured Linux CPU environment is available in `requirements-panda-pilot.txt`; use it in a separate environment when matching the original dependency set.

## Quick checks

```bash
python -m unittest discover -s tests -v
python scripts/run_panda.py --model-dir third_party/menagerie/franka_emika_panda --output outputs/panda/smoke --smoke
```

The smoke run trains a small temporary estimator and evaluates a six-second case. It is a pipeline check, not a reproduction of the reported scores. Physics tests require the pinned assets and MuJoCo version; read the test summary for skips.

## Train or reuse the Panda estimator

```bash
# Fresh estimator, with paired nominal/GRU evaluation
python scripts/run_panda.py --model-dir third_party/menagerie/franka_emika_panda --output outputs/panda/trained

# Frozen estimator used in the reported experiments
python scripts/run_panda.py --model-dir third_party/menagerie/franka_emika_panda --reuse-checkpoint checkpoints/panda_gru.pt --output outputs/panda/frozen

# Replay either saved trace at its recorded simulation rate
python scripts/view_panda.py --model-dir third_party/menagerie/franka_emika_panda --trace outputs/panda/frozen/baseline_rollout.npz
python scripts/view_panda.py --model-dir third_party/menagerie/franka_emika_panda --trace outputs/panda/frozen/adaptive_gru_rollout.npz
```

The viewer requires a graphical session. It uses recorded poses and traces, not a second dynamics simulation. Use the matching config when viewing custom runs.

## Four-controller benchmarks

```bash
# Calibrate integral and residual-observer settings, then evaluate 30 cases
python scripts/benchmark_panda.py --model-dir third_party/menagerie/franka_emika_panda --checkpoint checkpoints/panda_gru.pt --output outputs/panda/benchmark

# Frozen-controller robustness study: 288 rollouts
python scripts/benchmark_panda_robustness.py --model-dir third_party/menagerie/franka_emika_panda --checkpoint checkpoints/panda_gru.pt --output outputs/panda/robustness

# Verify saved traces, pairing, FIFO behavior, torque bounds and reported metrics
python scripts/verify_panda_robustness.py outputs/panda/robustness
```

Use a fresh output directory for every run. The robustness runner saves attempts, configuration, scenarios, versions, source/checkpoint hashes, traces, episode metrics, summaries, and paired comparisons. No graphics context is required for headless evaluation. It does not retrain or retune the controllers.

## Original 2D experiments

The NumPy point-mass benchmark remains independently runnable:

```bash
python scripts/smoke_p0.py
python scripts/run_multiseed.py --seeds 123 124 125 --device cpu --output-subdir research/multiseed
python scripts/report_multiseed.py outputs/research/multiseed
```

The multi-seed runner shares dataset directories using symbolic links. Windows may require Developer Mode or suitable permissions for this command. Panda execution does not require that path.
