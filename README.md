# cooling-algorithms
Cooling algorithms simulation code, using Google's cirq and qsim circuit emulator. 

Theory behind the algorithms can be found in (https://arxiv.org/abs/2506.21318, https://arxiv.org/abs/2404.12175). 
Documentation on cirq (https://quantumai.google/cirq) and qsim (https://quantumai.google/qsim). 

## Installation 

Requires **Python ≥ 3.10**. 

# 1. create and activate an environment
conda create -n cooling python=3.10
conda activate cooling

# 2. cirq, qsim, pandas
conda install -c conda-forge cirq=1.3
pip install qsimcirq==0.21.0 pandas

# 3. (optional for ED) QuSpin 
# see installation docs https://quspin.github.io/QuSpin/installation/installation.html
pip install quspin 

# 4. install this package in editable mode (registers `import cooling`)
pip install -e .

## Minimal example
See examples/example.ipynb notebook.

## Code structure 

The code has several different modules and classes, which may be extended based on the existing code templates:

- lattices: defines lattice geometry for model (e.g. 2D square lattice)
- device.py: defines cirq device for system and bath qubits (from lattice or cirq device)
- models: defines model (e.g. quantum Ising model) as well as circuit versions (e.g. Floquet unitary)
- protocols: defines different cooling protocols/reset channels (e.g. detailed balance protocol)
- noise: noise model applied to simulation circuit
- schedules: finer-grained time-control over cooling schedule, e.g. randomized cooling channels
- measurements: define what to measure during cooling simulation
- simulation.py: simulator (wraps qsim.simulator)
- ed.py: exact diagonalisation using QuSpin, obtains thermal and ground state energies of model

Happy cooling :) 


