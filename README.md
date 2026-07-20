# cooling-algorithms

Author: **Jerome Lloyd** <jerome.lloyd@unige.ch>.

Cooling algorithms simulation code, using Google's cirq and qsim circuit emulator. 

Theory behind the algorithms can be found in (https://arxiv.org/abs/2506.21318, https://arxiv.org/abs/2404.12175). 
Documentation on cirq (https://quantumai.google/cirq) and qsim (https://quantumai.google/qsim). 

Simulation based on time-evolving block decimation (TEBD) due to **Yuxuan Zhang** is found in the tebd folder.

## Installation 

#### 0. clone the repo
git clone https://github.com/jf-lloyd/cooling-algorithms.git  

cd cooling-algorithms  

#### 1. create and activate an environment
conda create -n cooling -c conda-forge python=3.12 cirq qsimcirq numba pandas pyarrow

conda activate cooling

#### 2. (optional for ED) QuSpin 
#### see installation docs https://quspin.github.io/QuSpin/installation/installation.html 
#### the install run into issues on mac
pip install quspin          # exact diagonalisation (ed.py); omit if you don't need ED

#### 3. install this package in editable mode (registers `import cooling`)
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


