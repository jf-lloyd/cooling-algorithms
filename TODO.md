# TODO

- Add cirq Device constraint checking
- Consider defining YXPowGate and ZXPowGate to replace current single-qubit rotation structure in DetailedBalanceProtocol
- Add XY model
- Add dynamical cooling protocol (Lianghong) — `protocols/dynamical_pc.py` is currently an empty placeholder
- Write Schedule class properly — informed by profiling results (see `examples/profiling_results.json`, `examples/scaling_results.json`)
- Validation notebook: compare new protocol circuits/dynamics against old code
