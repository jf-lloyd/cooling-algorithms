import math
from itertools import combinations, product as iproduct
import numpy as np
import cirq
from .schedulebase import Schedule

def _false():
    return False


class Randomized(Schedule):
    """
    Each channel application uses a new random system-bath geometry and randomly selects
    cooling operators from the allowed protocol set. Other protocol parameters are fixed,
    provided by params. The randomization provides ergodic dynamics in practice.
    Bath qubits are treated as indistinguishable for random geometry selection, so
    random geometries are sampled as combinations of system sites.

    In practice, drawing a new circuit every step slows down simulation;
    we therefore build a fixed cache of random circuits at the start of the simulation,
    and randomly sample from this pool. The cache can be renewed periodically.

    Parameters
    ----------
    protocol          : Protocol
    coupling_geometry : dict {bath_idx: sys_idx} or None. Fixed geometry if provided; random per circuit if None.
                        If None and Nb > Ns, bath qubit i is fixed to system qubit i for i < Ns
                        (one dedicated bath per system qubit), and only the remaining Nb - Ns bath
                        qubits are given a random assignment to system qubits (see build_cache).
    coupling_ops      : dict {bath_idx: op_str} or None. Fixed coupling ops if provided; random per circuit if None.
    allowed_ops       : list [op_str] or None. Draw from a subset of allowed coupling operators if provided;
                        defaults to all ops supported by the protocol.
    n_cache           : number of circuits to pre-build.
    seed              : RNG seed for reproducibility
    resample          : refresh the cache every resample steps (t=resample, 2*resample, ...); no refresh if None
    parameterized     : if False (default), mark cached circuits as not parameterized so qsim skip the sympy scan.
    """

    def __init__(self, protocol, coupling_geometry=None, coupling_ops=None,
                 allowed_ops=None, n_cache: int = 50, resample=None,
                 resample_trajectories: bool = False, randomize_sector: bool = False,
                 parameterized: bool = False, seed=None, compile: bool = True):

        super().__init__(protocol)
        self.coupling_geometry = coupling_geometry
        self.coupling_ops     = coupling_ops
        self.n_cache          = n_cache
        self.parameterized    = parameterized
        if resample is not None and (not isinstance(resample, int) or resample <= 0):
            raise ValueError(f"resample must be a positive integer, got {resample!r}.")
        self.resample              = resample
        self.resample_trajectories = resample_trajectories
        # Randomise the frame sign (phi -> -phi) over the two symmetry-broken sectors.
        # phi -> -phi is the sublattice flip, and the resulting channel is the original one
        # conjugated by the Z2 operator P = prod(Z), so the two steady states are rho and
        # P rho P. Averaging them restores the symmetry: every P-even expectation value
        # (energy, all even-point correlators, the ground-manifold weight) is untouched,
        # while P-odd ones (the order parameter <X_i>) average to zero. The point is the
        # connected correlator: in a single sector it is a difference of two large numbers
        # and comes out short-ranged, whereas after randomisation it equals the full
        # correlator and reads out the classical long-range order directly.
        #
        # Sectors ALTERNATE across trajectories rather than being drawn at random. Random
        # draws leave a sector imbalance u with E[u^2] = 1/K, and since the residual <X_i>
        # are perfectly correlated across sites, that imbalance feeds a systematic 1/K bias
        # into the disconnected part and hence into C(d). Alternating is the antithetic
        # version: for an even number of trajectories the imbalance is exactly zero, the
        # order parameter cancels to machine precision and the bias vanishes identically.
        #
        # The sector is fixed ONCE PER TRAJECTORY, not per cycle. Flipping it every cycle
        # would instead prepare the fixed point of the averaged channel (Phi_+ + Phi_-)/2,
        # which is a different and much worse state: the pump keeps reversing and drags the
        # system back and forth (measured at L=6, phi=50 deg: ground-manifold weight 0.61
        # and C(5) = -0.29, against 0.98 and -0.79 for the intended mixture of fixed points).
        self.randomize_sector = randomize_sector
        self._sector = 0
        self._traj_count = 0
        if randomize_sector and abs(float(protocol.params.get("phi", 0.))) < 1e-12:
            raise ValueError(
                "randomize_sector=True requires a non-zero 'phi' in the protocol params; "
                "at phi = 0 the two sectors give the same circuit.")
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

        self.compile    = compile
        rng             = np.random.default_rng(seed)
        self._rng       = np.random.default_rng(rng.integers(2**31))
        self._build_rng = rng

        self.build_cache()

    @property
    def name(self) -> str:
        return "rand"

    def build_cache(self, _warn=True):
        """
        Build (or rebuild) the cache with n_cache unique random circuits. If there are
        n_possible < n_cache random circuits, the cache size is reduced to n_possible.

        For n_possible < 1000: enumerates all configs and samples without replacement.
        Otherwise: frozenset rejection sampling (collision rate negligible for n_cache << n_possible).

        With randomize_sector=True each configuration is cached twice, once per frame
        sign, so the cache holds 2 * effective_n circuits and circuit_fn's uniform draw
        yields a balanced mixture of the two symmetry-broken sectors.

        Random geometry (coupling_geometry=None):
            If Nb <= Ns, geometries are random Nb-subsets of system qubits
            (combinations(range(Ns), Nb)).
            If Nb > Ns, bath qubit i is fixed to system qubit i for i < Ns
            (one dedicated bath per system qubit), and only the remaining
            Nb - Ns bath qubits are given a random (without-replacement)
            assignment to system qubits.
        """
        Nb, Ns = self.protocol.device.Nb, self.protocol.device.Ns
        rng = self._build_rng

        random_geometry = self.coupling_geometry is None
        hybrid_geometry = random_geometry and Nb > Ns
        n_rand_geom = (Nb - Ns) if hybrid_geometry else Nb

        if hybrid_geometry:
            fixed_geom   = {bi: bi for bi in range(Ns)}
            geom_options = math.comb(Ns, n_rand_geom)
        elif random_geometry:
            fixed_geom   = {}
            geom_options = math.comb(Ns, Nb)
        else:
            fixed_geom   = {}
            geom_options = 1

        ops_options = len(self.allowed_ops) ** Nb if self.coupling_ops is None else 1
        n_possible  = geom_options * ops_options

        effective_n = min(self.n_cache, n_possible)
        if _warn and effective_n < self.n_cache:
            print(f"Randomized: cache size reduced from {self.n_cache} to {effective_n} "
                  f"(only {n_possible} unique circuits possible).")

        self._cache = []

        if n_possible < 1000:
            # Enumerate all configs, sample without replacement
            if hybrid_geometry:
                extra_geoms = list(combinations(range(Ns), n_rand_geom))
                fixed_tuple = tuple(fixed_geom[bi] for bi in range(Ns))
                geoms = [fixed_tuple + extra for extra in extra_geoms]
            elif random_geometry:
                geoms = list(combinations(range(Ns), Nb))
            else:
                geoms = [tuple(self.coupling_geometry[bi] for bi in range(Nb))]
            ops   = list(iproduct(self.allowed_ops, repeat=Nb)) \
                    if self.coupling_ops is None \
                    else [tuple(self.coupling_ops[bi] for bi in range(Nb))]
            all_configs  = list(iproduct(geoms, ops))
            selected_idx = rng.choice(len(all_configs), size=effective_n, replace=False)
            for idx in selected_idx:
                geom_tuple, ops_tuple = all_configs[idx]
                for sgn in self._sector_signs():
                    self._cache.append(self.protocol.channel(
                        dict(enumerate(geom_tuple)), dict(enumerate(ops_tuple)),
                        params=sgn, compile=self.compile))
        else:
            # Rejection sampling — collisions rare when n_cache << n_possible
            expected_extra = effective_n ** 2 / (2 * n_possible)
            if _warn and expected_extra > 50:
                print(f"Warning: rejection sampling will be inefficient "
                      f"(~{expected_extra:.0f} expected extra draws). "
                      f"Consider reducing n_cache.")
            seen = set()
            while len(self._cache) < effective_n:
                if hybrid_geometry:
                    extra_indices = sorted(rng.choice(Ns, size=n_rand_geom, replace=False))
                    geometry = dict(fixed_geom)
                    geometry.update({bi: int(si) for bi, si in zip(range(Ns, Nb), extra_indices)})
                elif random_geometry:
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
                    for sgn in self._sector_signs():
                        self._cache.append(self.protocol.channel(
                            geometry, coupling_ops, params=sgn, compile=self.compile))

        if not self.parameterized:
            # Circuits are concrete (no sympy): tell cirq/qsim so they skip the
            # parameterization scan and don't copy the circuit on resolve.
            for fc in self._cache:
                fc._is_parameterized_ = _false

    def reset_sector_counter(self):
        """Restart the trajectory alternation (call before an independent batch)."""
        self._traj_count = 0

    def _sector_signs(self):
        """Params overrides to append per configuration: one entry normally, both frame
        signs when randomize_sector is on (so the cache is exactly balanced even at
        n_cache = 1, and circuit_fn's uniform draw gives a 50/50 sector mixture)."""
        if not self.randomize_sector:
            return [None]
        phi = float(self.protocol.params["phi"])
        return [{"phi": phi}, {"phi": -phi}]

    @property
    def sim_options(self) -> dict:
        return {'n_cache': self.cache_size,
                'resample_trajectories': self.resample_trajectories,
                'randomize_sector': self.randomize_sector}

    @property
    def fname(self) -> str:
        Nb = self.protocol.device.Nb
        tag = "_randsec" if self.randomize_sector else ""
        return f"{self.protocol.model.name}_Nb{Nb}_{self.protocol.name}_rand{tag}"

    def circuit_fn(self, t: int) -> cirq.FrozenCircuit:
        if self.resample is not None and t % self.resample == 0:
            self.build_cache(_warn=False)
        if not self.randomize_sector:
            return self._cache[self._rng.integers(len(self._cache))]
        # Simulation.run restarts the step counter at t = 1 for every trajectory, so that
        # is where a new sector is drawn; it is then held for the rest of the trajectory.
        if t <= 1:
            self._sector = self._traj_count % 2
            self._traj_count += 1
        n_cfg = len(self._cache) // 2
        return self._cache[2 * int(self._rng.integers(n_cfg)) + self._sector]
