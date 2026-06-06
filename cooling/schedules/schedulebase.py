"""
Schedule base class for cooling protocols.

Created by Jerome Lloyd on 5th June 2026.
"""

from abc import ABC, abstractmethod
import numpy as np
import cirq


class Schedule(ABC):
    """
    Abstract base class for cooling schedules. 

    A Schedule allows for fine-grained control of time-dependence at the per-reset level 
    i.e. different channels (or the same channel with different parameters) can be applied 
    at different reset steps. Schematically, S = C1C2...Ct

    It also allows caching of circuits, which is useful for optimizing the simulation 
    (same circuits can be reused)

    Parameters
    ----------
    protocol : Protocol
    """

    def __init__(self, protocol, params: dict):
        self.protocol = protocol
        self.params   = params
        self._cache   = []

    @property
    def sim_options(self) -> dict:
        return {'n_cache': self.cache_size,
                'resample_trajectories': False}

    @abstractmethod
    def circuit_fn(self, t: int) -> cirq.FrozenCircuit:
        """Return the circuit to run at step t."""
        pass

    @property
    def cache_size(self) -> int:
        """Size of the circuit cache — pass this to Simulation as circuit_memoization_size."""
        return len(self._cache)



