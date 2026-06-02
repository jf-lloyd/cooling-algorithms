# cooling-algorithms
Cooling algorithm simulation code, using Google's cirq and qsim circuit emulator. 

Theory behind the algorithms can be found in (https://arxiv.org/abs/2506.21318, https://arxiv.org/abs/2404.12175). 
Documentation on cirq (https://quantumai.google/cirq) and qsim (https://quantumai.google/qsim). 

## Code structure 

The code has several main modules:
- device.py (lattice geometry and cirq device)
- model.py (Hamiltonian e.g. Ising model, and Floquet versions (system unitary used in algorithm))
- protocol.py (different protocol choice e.g. thermal detailed balance protocol)
- simulation.py (different simulation modes and data saving)
- measure.py (defines what observables to record during simulation)
  
and currently 'helper' modules:
- ed.py (exact diagonalisation for different models, using QuSpin -- should probably be merged with model.py)
- fermion.py (free fermion simulation)

## Installation 
**todo**

## Minimal example
**todo**
