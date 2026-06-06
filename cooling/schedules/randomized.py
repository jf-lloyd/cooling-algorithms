import math
from itertools import combinations, product as iproduct
import numpy as np
import cirq
from .schedulebase import Schedule


class Randomized(Schedule):
    """
    Each channel application uses a new random system-bath geometry and randomly selects
    cooling operators from the allowed protocol set. Other protocol parameters are fixed,
    provided by params. The randomization provides ergodic dynamics in practice.

    In practice, drawing a new circuit every step slows down simulation;
    we therefore build a fixed cache of random circuits at the start of the simulation,
    and randomly sample from this pool. The cache can be renewed periodically.

    Parameters
    ----------
    protocol          : Protocol
    params            : dict — channel params (beta, delta, h, theta, NT, ...).
    coupling_geometry : dict {bath_idx: sys_idx} or None. Fixed geometry if provided; random per circuit if None.
    coupling_ops      : dict {bath_idx: op_str} or None. Fixed coupling ops if provided; random per circuit if None.
    allowed_ops       : list [op_str] or None. Draw from a subset of allowed coupling operators if provided;
                        defaults to all ops supported by the protocol.
    n_cache           : number of circuits to pre-build.
    seed              : RNG seed for reproducibility
    resample          : refresh the cache every resample steps (t=resample, 2*resample, ...); no refresh if None
    parameterized     : if False (default), mark cached circuits as not parameterized
                        (_is_parameterized_ -> False) so qsim/cirq skip the sympy
                        scan and avoid copying the circuit (preserves memoization).
                        The protocol builds concrete circuits, so this is safe.
    """

    def __init__(self, protocol, params: dict, coupling_geometry=None, coupling_ops=None,
                 allowed_ops=None, n_cache: int = 50, resample=None,
                 resample_trajectories: bool = False,
                 parameterized: bool = False, seed=None):

        super().__init__(protocol, params)
        self.params           = params
        self.coupling_geometry = coupling_geometry
        self.coupling_ops     = coupling_ops
        self.n_cache          = n_cache
        self.parameterized    = parameterized
        if resample is not None and (not isinstance(resample, int) or resample <= 0):
            raise ValueError(f"resample must be a positive integer, got {resample!r}.")
        self.resample              = resample
        self.resample_trajectories = resample_trajectories
        if n_cache > 200:
            print("Warning: large cache size may reduce simulation efficiency! "
                  "Suggested n_cache <= 100 and periodic resampling.")

        # Default allowed_ops to all ops supported by the protocol
        all_ops = list(protocol.allowed_coupling_gates.keys())
        if allowed_ops is None:
            self.allowed_ops = all_ops
        else:
            invalid = set(allowed_ops) - set(all_ops)
            if invalid:
                raise ValueError(
                    f"allowed_ops {invalid} not supported by protocol. "
                    f"Choose from {all_ops}."
                )
            self.allowed_ops = list(allowed_ops)

        rng        = np.random.default_rng(seed)
        self._rng  = np.random.default_rng(rng.integers(2**31))
        self._build_rng = rng

        self.build_cache()

    def build_cache(self, _warn=True):
        """
        Build (or rebuild) the cache with n_cache unique random circuits. If there are
        n_possible < n_cache random circuits, the cache size is reduced to n_possible.

        For n_possible < 1000: enumerates all configs and samples without replacement.
        Otherwise: frozenset rejection sampling (collision rate negligible for n_cache << n_possible).
        """
        Nb, Ns = self.protocol.device.Nb, self.protocol.device.Ns
        rng = self._build_rng

        geom_options = math.comb(Ns, Nb) if self.coupling_geometry is None else 1
        ops_options  = len(self.allowed_ops) ** Nb if self.coupling_ops is None else 1
        n_possible   = geom_options * ops_options

        effective_n = min(self.n_cache, n_possible)
        if _warn and effective_n < self.n_cache:
            print(f"Randomized: cache size reduced from {self.n_cache} to {effective_n} "
                  f"(only {n_possible} unique circuits possible).")

        self._cache = []

        if n_possible < 1000:
            # Enumerate all configs, sample without replacement
            geoms = list(combinations(range(Ns), Nb)) \
                    if self.coupling_geometry is None \
                    else [tuple(self.coupling_geometry[bi] for bi in range(Nb))]
            ops   = list(iproduct(self.allowed_ops, repeat=Nb)) \
                    if self.coupling_ops is None \
                    else [tuple(self.coupling_ops[bi] for bi in range(Nb))]
            all_configs  = list(iproduct(geoms, ops))
            selected_idx = rng.choice(len(all_configs), size=effective_n, replace=False)
            for idx in selected_idx:
                geom_tuple, ops_tuple = all_configs[idx]
                self._cache.append(self.protocol.channel(
                    dict(enumerate(geom_tuple)), dict(enumerate(ops_tuple)), self.params))
        else:
            # Rejection sampling — collisions rare when n_cache << n_possible
            expected_extra = effective_n ** 2 / (2 * n_possible)
            if _warn and expected_extra > 50:
                print(f"Warning: rejection sampling will be inefficient "
                      f"(~{expected_extra:.0f} expected extra draws). "
                      f"Consider reducing n_cache.")
            seen = set()
            while len(self._cache) < effective_n:
                if self.coupling_geometry is None:
                    sys_indices = sorted(rng.choice(Ns, size=Nb, replace=False))
                    geometry    = {bi: int(si) for bi, si in enumerate(sys_indices)}
                else:
                    geometry = self.coupling_geometry
                if self.coupling_ops is None:
                    op_choices   = rng.choice(self.allowed_ops, size=Nb)
                    coupling_ops = {bi: op_choices[bi] for bi in range(Nb)}
                else:
                    coupling_ops = self.coupling_ops
                key = tuple(sorted(
                    (int(sys_idx), coupling_ops[bi])
                    for bi, sys_idx in geometry.items()
                ))
                if key not in seen:
                    seen.add(key)
                    self._cache.append(self.protocol.channel(geometry, coupling_ops, self.params))

        if not self.parameterized:
            # Circuits are concrete (no sympy): tell cirq/qsim so they skip the
            # parameterization scan and don't copy the circuit on resolve.
            for fc in self._cache:
                fc._is_parameterized_ = lambda: False

    @property
    def sim_options(self) -> dict:
        return {'n_cache': self.cache_size,
                'resample_trajectories': self.resample_trajectories}

    def fname(self) -> str:
        return f"{self.protocol.name}_rand"

    def circuit_fn(self, t: int) -> cirq.FrozenCircuit:
        if self.resample is not None and t % self.resample == 0:
            self.build_cache(_warn=False)
        return self._cache[self._rng.integers(len(self._cache))]
