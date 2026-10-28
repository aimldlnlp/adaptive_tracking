# Panda GRU checkpoint

`panda_gru.pt` is the frozen estimator used by the reported Panda benchmarks.
It has 11,427 parameters, a 48-unit GRU, twelve input features, and a
12-sample history at 50 Hz (220 ms between the oldest and newest samples).
Training used 48 nominal-controller episodes and 12 validation episodes;
weights were selected by validation force error and remain fixed during control.

SHA-256: `6e498db1ebfbe33b33761dd490c90b2523fbc9421da30531f9f714d891e616d1`.

The bundle includes normalization statistics and model metadata. Only load
trusted PyTorch checkpoints. A fresh training run is supported, but its
scores need not exactly match this checkpoint.
