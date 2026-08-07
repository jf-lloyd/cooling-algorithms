# Compressing the repeated-interaction cooling cycle

Circuit compression for **repeated-interaction (RI) thermal-state preparation** of the transverse-field
Ising model. One cooling cycle is a deep circuit (702 two-qubit gates for 3×4+1 at β=1) and a protocol
needs hundreds of cycles, so per-cycle depth is the hardware bottleneck.

**Start here:** [`examples/compression_walkthrough.ipynb`](examples/compression_walkthrough.ipynb) —
runs in seconds from pre-computed data, no GPU.

---

## Headline results

| | 2q-gates/cycle to reach 1% infidelity (3×4+1, β=1) | vs native |
|---|---|---|
| native cycle | 702 | 1× |
| 1st-order Trotter | never reaches 1% | — |
| 2nd-order Trotter | **259** | 2.7× |
| variational (tuned angles) | **190** | 3.7× |

**And the result that matters most:** cooling depends on the cycle's **channel**, not its unitary.
Because the cycle ends in a bath reset + trace, many unitaries realise the same channel. Optimising the
*channel* gives a compressed cycle whose **unitary fidelity is 0.005** (essentially a random circuit)
that nonetheless **cools to the Gibbs state at fidelity 0.98**, with **5.25× fewer** two-qubit gates.
Unitary fidelity is the wrong objective for a dissipative protocol.

Two further findings:

- **2nd order is free for Ising.** The 2-body terms all commute, so the symmetric split only duplicates
  the single-qubit field. General rule for `C` non-commuting 2-body colour classes:
  overhead = `max(1, (2C-2)/C)` — Ising `C=1` free, 1D `C=2` free, 2D Heisenberg `C=4` → 1.5×
  (measured 1.6× by exact ED).
- **Variational compression pays off at shallow depth; past k~13 the optimiser stops improving.**
  With Adam warm-started from the coarse Trotter angles, every observed gain is at k <= 13 layers
  (2.3-10.8x better than Trotter at the same depth) and every stall at k >= 14 (~250 gates, ~56 free
  parameters), where the run returns its initialisation *unchanged* -- never worse than Trotter by
  construction, but no gain. The stalls happen to be the beta >= 2 runs, but beta <= 1 was never tested
  above k=13, so **this data cannot separate a depth/parameter-count limit from a temperature limit**
  (test pending: beta=1 at k=16,20). Incremental-identity initialisation is untested in either case.

---

## Layout

```
ricc/            importable package (plotting / analysis of the results)
examples/        compression_walkthrough.ipynb   <- the notebook
data/            pre-computed result JSONs (dense/ED); the notebook reads only these
scripts/         drivers that regenerate data/  (need numpy/scipy; torch for the variational ones)
protocol/        cirq/qsim ground-truth definition of the cooling cycle (reference only)
figures/         exported figures (pdf + png)
```

## Install & run

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib is enough for the notebook
jupyter lab examples/compression_walkthrough.ipynb
```

## Regenerating the data

```bash
cd scripts
python cc_trot_sweep.py 3 4 1.0    # 1st/2nd-order Trotter sweep   -> cc_trot_3x4.json
python cc_cycle_adam.py 3 4 1.0    # variational compression       -> cc_cycle_adam_3x4.json
python cc_heisenberg.py            # Heisenberg commutation check  -> cc_heisenberg.json
```
`cc_cycle_adam.py` honours `KS`, `NIT`, `NREST` env vars (depths, Adam iterations, restarts) and saves
after every depth. Copy fresh JSONs into `data/` and re-run the notebook.

## Scope and caveats

- Compression shrinks **per-cycle depth**, not the **number of cycles** needed to reach the Gibbs state.
- Variational results are dense state-vector/ED → 3×4+1 (13q) and 4×4+1 (17q). Past ~17–21q the dense
  optimiser is too heavy; scaling needs the sampled-cost method (arXiv:2409.16346) or Pauli propagation.
- The 5.25× channel-level number is a **different objective on a different system** (1D, channel-match)
  from the 3.7× unitary result (2D, unitary-match). Do not multiply them together.

## References

- Protocol: arXiv:2506.21318 (RI / detailed-balance cooling)
- Compression method: arXiv:2409.16346 (sampled Hilbert-Schmidt cost, incremental-identity init)
