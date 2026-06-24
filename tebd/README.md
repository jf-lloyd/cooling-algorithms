# MPS-TEBD repeated-interaction cooling

Author: **Yuxuan Zhang** &lt;yuxuanzhang@utexas.edu&gt;

Matrix-product-state (MPS) trajectory simulator for repeated-interaction (RI)
thermal-state preparation of the 1D transverse-field Ising model, with noise.
It scales to system sizes beyond exact state-vector simulation (entanglement-
limited rather than qubit-limited).

## Convention matches the reference state-vector simulation

`ri_mps_cli.py` runs the **exact** RI cooling circuits of the `cooling-algorithms`
package (`DetailedBalanceProtocol`, `BasicNoiseModel`, `Randomized` schedule) on an
MPS backend, so its protocol / Trotter / noise convention is identical to the
reference state-vector code — the MPS is used only as a scalable backend. Gate
application is faithful to floating point (`fidelity = 1.000` vs `cirq.Simulator`
on a single cooling cycle).

**Validation** (L=10, Nb=10, critical `J=g=1`, `beta=1`, full noise
`p1=1e-4, p2=1e-3`, second-half time average, 24 trajectories):

| | steady-state error vs OBC ED |
| --- | --- |
| reference state-vector (`cooling-algorithms`) | 8.8% |
| **`ri_mps_cli.py` (this code)** | **8.8% ± 0.7%** |

i.e. the MPS backend reproduces the reference convention exactly, at scale.

## Files

- `ri_mps_cli.py` — runs the exact RI cooling circuit on an MPS (trajectory
  unravelling of the depolarizing channels and bath resets). Works for **any 1D
  model** defined in `cooling-algorithms` (`--model {ising, heisenberg, xy}`); the
  energy is the model's own Hamiltonian `<model.hamiltonian>` (a `cirq.PauliSum`)
  measured generically term-by-term on the MPS, so it matches `cooling`'s own
  `DefaultMeasurement1` H0. OBC.
- `quimb_traj.py` — standalone MPS-TEBD trajectory RI-cooling simulator (MPI), self-
  contained (no cooling-algorithms dependency). Supports any built-in 1D spin model via
  `--model {ising, xy, heisenberg}` (native Trotter gates + generic <H_sys> energy),
  `--trotter_order {1,2}`, and `--init {cold, random}`.

## Dependencies

`numpy`, `scipy`, `quimb`, `cirq`, and the `cooling-algorithms` package
(J. Lloyd), which provides the reference RI cooling protocol. By default
`ri_mps_cli.py` looks for it at `../cooling-algorithms`; otherwise set
`COOLING_ALGS=/path/to/cooling-algorithms`.

## Usage

```bash
python ri_mps_cli.py --L 10 --NB 10 --J 1.0 --g 1.0 --beta 1.0 \
    --p1 1e-4 --p2 1e-3 --ncycles 500 --sample_every 1 \
    --chi 256 --cutoff 1e-9 --seed 1 --out run0
```

Other models: `--model heisenberg --Jxx 1 --Jyy 1 --Jzz 1` or `--model xy --Jxx 1 --Jyy 1`.
Trotter order: `--trotter_order {1,2}` (default 2 = second-order Strang; 1 = first-order).

Writes `run0.csv` with columns `cyc, E, E/L` (column 2 is energy per site).
Average the second half of the samples over several seeds for the steady-state
estimate. See `example.sh`.
