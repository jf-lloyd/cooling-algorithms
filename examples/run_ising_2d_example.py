"""
Example: 2D Ising cooling simulation on a square lattice.

Pipeline:
  1. Define lattice geometry
  2. Build Hamiltonian + noise model
  3. Build cooling protocol + randomised schedule
  4. Run trajectories          <-- replace this block with GPU routine
  5. Load and inspect results
"""

import os
import cooling
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Set this to wherever you want output parquet files written
# ---------------------------------------------------------------------------
save_path = "./data"

# ---------------------------------------------------------------------------
# 1. Lattice
# ---------------------------------------------------------------------------
# SquareLattice2D(Lx, Ly, pbc_x, pbc_y) — periodic boundary conditions on both axes
lattice = cooling.SquareLattice2D(Lx=3, Ly=3, pbc_x=True, pbc_y=True)

# Attach Nb bath qubits to the system
Nb = 3
device = cooling.CoolingDevice.from_lattice(lattice, Nb=Nb)

# ---------------------------------------------------------------------------
# 2. Hamiltonian  H = -J Σ ZZ - g Σ X
# ---------------------------------------------------------------------------
model = cooling.IsingModel(device, {"J": 0.2, "g": 1.0, "gx": 0.0})

# Gate-level depolarising noise: p2 on 2-qubit gates, p1 = p2/10 on single-qubit
p2 = 1e-3
noise = cooling.BasicNoiseModel(device, {
    "p1": p2 / 10, "p2": p2, "p_reset": 0.0,
    "sys": True, "bath": True, "coupling": True,
})

# ---------------------------------------------------------------------------
# 3. Cooling protocol + randomised schedule
# ---------------------------------------------------------------------------
# DetailedBalanceProtocol builds Trotter circuits that satisfy detailed balance
# at inverse temperature beta via a Gaussian filter function.
protocol = cooling.DetailedBalanceProtocol(
    device, model,
    params={"beta": 1.0, "delta": 0.25, "h": 4.0, "theta": 1.0, "NT": 2},
    function="gaussian",
    noise_model=noise,
    trotter_order=2,
)

# Randomized draws from a cache of pre-compiled circuits at each step.
# allowed_ops controls which Pauli axes the bath couples through.
sched = cooling.Randomized(protocol, n_cache=50, seed=1, allowed_ops=["X", "Y", "Z"])

sim = cooling.Simulation(protocol)

# ---------------------------------------------------------------------------
# 4. Run trajectories
#    R: steps per trajectory   K: number of trajectories
#
#    *** REPLACE THIS BLOCK WITH GPU ROUTINE ***
#    The GPU routine should produce the same output as sim.run_parallel():
#    a parquet file at save_path with columns [repeat, t, H0, total_Z, total_X, total_Y, total_S2]
#    where t is the step index and H0 is the system energy at that step.
#
#    There is a default "measure" routine which is called in run, which can be reused. 
# ---------------------------------------------------------------------------
os.makedirs(save_path, exist_ok=True)
tag = f"Nb{Nb}_p2{p2:.0e}" #fname tag

R, K = 200, 10 # number of resets, number of trajectories
sim.run(sched, R=R, K=K, seed=1, tag=tag, save_path=save_path)

# ---------------------------------------------------------------------------
# 5. Load and inspect results
# ---------------------------------------------------------------------------
# Filename is determined by model.lattice.name and the protocol/noise params.
# Use cooling.fname or construct manually:
fname = (
    f"IsingModel_{lattice.name}"
    f"_J{0.2:.3f}_g{1.0:.3f}_gx0.000"
    f"_Nb{Nb}_b{1.0:.2f}_d{0.25:.4f}_h{4.0:.2f}_th{1.0:.3f}"
    f"_gaussian_o2_rand_R{R}K{K}_{tag}.parquet"
)
record = pd.read_parquet(os.path.join(save_path, fname))

# Compute steady-state energy (average over last half of trajectory)
mean_E = record.groupby("t")["H0"].mean()
e_ss = mean_E.loc[mean_E.index >= mean_E.index.max() // 2].mean()

# Compare to exact thermal energy via ED (requires quspin2 conda env)
_, e_thermal = cooling.ThermalEnergy(model, beta=1.0, k=500, save=True, verbal=False)

rel_err = abs(e_ss - e_thermal) / abs(e_thermal)
print(f"Steady-state energy: {e_ss:.4f}")
print(f"Thermal energy (ED): {e_thermal:.4f}")
print(f"Relative error:      {rel_err:.4f}")
