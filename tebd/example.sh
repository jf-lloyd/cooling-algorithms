#!/usr/bin/env bash
# Example: critical Ising (J=g=1, beta=1), L=10 system + Nb=10 bath, full noise.
# Runs a few trajectories of the exact RI cooling on the MPS backend and
# reports the second-half-averaged steady-state energy. It should land near the
# OBC ED thermal value E/L = -1.07744 (~8.8% averaged, the validated match to
# the reference state-vector code).
set -e
: "${COOLING_ALGS:=../cooling-algorithms}"; export COOLING_ALGS
PY=${PY:-python}
for s in 1 2 3 4; do
  "$PY" ri_mps_cli.py --L 10 --NB 10 --J 1.0 --g 1.0 --beta 1.0 \
    --p1 1e-4 --p2 1e-3 --ncycles 300 --sample_every 1 \
    --chi 256 --cutoff 1e-9 --seed "$s" --out "ex_s$s" &
done
wait
"$PY" - <<'PYEOF'
import csv, glob, numpy as np
ED = -1.07744  # OBC ED thermal E/L, critical J=g=1, beta=1
vals = []
for f in sorted(glob.glob("ex_s*.csv")):
    eps = [float(r[2]) for r in csv.reader(open(f)) if r and not r[0].startswith('#')]
    if len(eps) >= 3:
        vals.append(np.mean(eps[len(eps) // 2:]))
e = np.mean(vals)
print(f"MPS (reference convention) steady E/L = {e:.4f}  ->  "
      f"{100 * abs(e - ED) / abs(ED):.1f}% vs ED  (n={len(vals)} seeds)")
PYEOF
