# TODO

- Add cirq Device constraint checking
- Add dynamical cooling protocol (Lianghong) — `protocols/dynamical_pc.py` is currently an empty placeholder
- Write Schedule class properly — informed by profiling results (see `claudespace/profiling_results.json`, `claudespace/scaling_results.json`)
  - `fname()` abstract on base; `name` property; `reseed(seed)` interface; `cache_size` not list-dependent; `sim_options`/`resample_trajectories` moved to `Randomized`
- Add `name` to Protocol base class
- Add tests
- Complete example notebook
- ~~Cluster setup for `run_parallel` (SLURM array job pattern)~~ — done, see
  `google/scripts/run_*.sh` and `QuEra/scripts/run_xy_*.sh`
- Profiling scripts: clean up and document in `claudespace/`
- ~~**[HIGH PRIORITY]** Gate-based noise model~~ — done via
  `cooling/noise/basic.py` (`BasicNoiseModel`, commit 7606287)
- Update `QuEra/scripts/job1_heis_noiseless.py` — still uses deprecated
  `gamma=` Protocol kwarg, replace with `noise_model=`
- Short depth initial state preparation
- Benchmark Kagome lattice cooling (see Halimeh et al. https://arxiv.org/abs/2605.26245)
- GPU measurement: use qsim `simulate_expectation_values()` at measurement steps
  instead of extracting state vector + cirq PauliSum on CPU. Needs `measure_from_circuit(circuit, state)`
  interface on Measurement. Required before GPU cluster runs are measurement-bound (crossover ~5×4+).
