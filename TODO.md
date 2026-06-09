# TODO

- Add cirq Device constraint checking
- Add dynamical cooling protocol (Lianghong) — `protocols/dynamical_pc.py` is currently an empty placeholder
- Write Schedule class properly — informed by profiling results (see `claudespace/profiling_results.json`, `claudespace/scaling_results.json`)
  - `fname()` abstract on base; `name` property; `reseed(seed)` interface; `cache_size` not list-dependent; `sim_options`/`resample_trajectories` moved to `Randomized`
- Add `name` to Protocol base class
- Add tests
- Complete example notebook
- Cluster setup for `run_parallel` (SLURM array job pattern)
- Profiling scripts: clean up and document in `claudespace/`
- **[HIGH PRIORITY]** Gate-based noise model — implement `NoiseModel` via `cirq_google.NoiseModelFromNoiseProperties` or manual Kraus channels; integrate with `cirq.Simulator(noise=...)`
- **[HIGH PRIORITY]** Replace bath evolution with system Kraus channel — precompute `K_i = <i|_bath U_SB |0>_bath`, eliminating bath qubits from state vector (~16× speedup for Nb=2)
- Short depth initial state preparation
- Benchmark Kagome lattice cooling (see Halimeh et al. https://arxiv.org/abs/2605.26245)
- GPU measurement: use qsim `simulate_expectation_values()` at measurement steps
  instead of extracting state vector + cirq PauliSum on CPU. Needs `measure_from_circuit(circuit, state)`
  interface on Measurement. Required before GPU cluster runs are measurement-bound (crossover ~5×4+).
