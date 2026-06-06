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
