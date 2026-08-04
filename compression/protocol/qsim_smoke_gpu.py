#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import resource
import sys
import time

import cirq
import numpy as np
import qsimcirq

import a_device
import b_model
import c_protocol
import e_measure


def boundary_fixed(self):
    boundary_sites = []
    if self.pbc_x is False:
        for y in range(self.Ly):
            boundary_sites.append(self.index(0, y))
            boundary_sites.append(self.index(self.Lx - 1, y))
    if self.pbc_y is False:
        for x in range(self.Lx):
            boundary_sites.append(self.index(x, 0))
            boundary_sites.append(self.index(x, self.Ly - 1))
    return sorted(set(boundary_sites))


a_device.RectLattice2D.boundary = boundary_fixed


def sample_basis_preparation_ops(qubits, rng):
    bits = rng.integers(0, 2, size=len(qubits), endpoint=False)
    return [cirq.X(q) for q, bit in zip(qubits, bits, strict=True) if int(bit) == 1]


def result_state_vector(result) -> np.ndarray:
    if hasattr(result, "state_vector"):
        state = result.state_vector()
    elif hasattr(result, "final_state_vector"):
        state = result.final_state_vector
    else:
        raise TypeError(f"Unsupported qsim result type: {type(result)!r}")
    state = np.asarray(state, dtype=np.complex64)
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"Invalid state-vector norm: {norm}")
    if abs(norm - 1.0) > 1e-12:
        state = np.asarray(state / norm, dtype=np.complex64)
    return state


def is_zero_operator(operator) -> bool:
    return isinstance(operator, (int, float, complex)) and operator == 0


def env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


def build_measurement(device, model, split_components: bool):
    measurement = e_measure.Measurement(device)
    if split_components:
        names = getattr(model, "hamiltonian_component_names", ("HJ", "Hg", "Hgx"))
        for name, operator in zip(names, model.hamiltonian_components, strict=True):
            if not is_zero_operator(operator):
                measurement.add_observable(name, operator)
    else:
        measurement.add_Hamiltonian(model)
    return measurement


def _flip_expectation_xx(state: np.ndarray, ns: int, i: int, j: int) -> float:
    # Cirq flattens the state tensor with qubit axis 0 as the most-significant bit.
    mask = (1 << (ns - 1 - i)) | (1 << (ns - 1 - j))
    indices = np.arange(state.size, dtype=np.int64)
    return float(np.vdot(state, state[indices ^ mask]).real)


def axial_xx_correlations(state: np.ndarray, lattice) -> dict[str, float]:
    ns = lattice.Ns
    out: dict[str, float] = {}
    pair_sum = 0.0
    pair_count = 0

    for dx in range(1, getattr(lattice, "Lx", ns)):
        vals = []
        for y in range(getattr(lattice, "Ly", 1)):
            for x in range(getattr(lattice, "Lx", ns) - dx):
                i = lattice.index(x, y) if lattice.Dim == 2 else lattice.index(x)
                j = lattice.index(x + dx, y) if lattice.Dim == 2 else lattice.index(x + dx)
                vals.append(_flip_expectation_xx(state, ns, i, j))
        if vals:
            mean = float(np.mean(vals))
            out[f"Cxx_dx{dx}"] = mean
            pair_sum += float(np.sum(vals))
            pair_count += len(vals)

    if lattice.Dim == 2:
        for dy in range(1, lattice.Ly):
            vals = []
            for x in range(lattice.Lx):
                for y in range(lattice.Ly - dy):
                    i = lattice.index(x, y)
                    j = lattice.index(x, y + dy)
                    vals.append(_flip_expectation_xx(state, ns, i, j))
            if vals:
                mean = float(np.mean(vals))
                out[f"Cxx_dy{dy}"] = mean
                pair_sum += float(np.sum(vals))
                pair_count += len(vals)

    # Axial-only structure proxy: diagonal self-correlations plus measured axial pairs.
    out["Cxx_axial_pair_count"] = float(pair_count)
    out["Sxx0_axial"] = float((ns + 2.0 * pair_sum) / (ns * ns))
    return out


def measurement_summary(state: np.ndarray, device, model, measurement, split_components: bool, measure_correlations: bool = False) -> dict[str, float]:
    joint_norm = float(np.linalg.norm(state))
    sys_state = np.asarray(state.reshape(2**device.Ns, 2**device.Nb)[:, 0], dtype=np.complex128)
    sys_norm = float(np.linalg.norm(sys_state))
    values: dict[str, float] = {}
    if sys_norm > 0:
        sys_state = sys_state / sys_norm
        sys_state = sys_state / np.linalg.norm(sys_state)
        values = {key: float(val) for key, val in measurement.measure_from_state_vector(sys_state).items()}
        if measure_correlations:
            values.update(axial_xx_correlations(sys_state, model.lattice))
    if split_components:
        names = getattr(model, "hamiltonian_component_names", ("HJ", "Hg", "Hgx"))
        for name in names:
            values.setdefault(name, 0.0)
        values["H0"] = sum(values[name] for name in names)
        g = float(model.params.get("g", 0.0))
        j = float(model.params.get("J", 0.0))
        j2 = float(model.params.get("J2", 0.0))
        if g != 0.0:
            values["Ztot"] = -values["Hg"] / g
            values["Zavg"] = values["Ztot"] / device.Ns
        if j != 0.0:
            values["XXbond_sum"] = values["HJ"] / j
            values["XXbond_avg"] = values["XXbond_sum"] / max(1, len(model.lattice.nearest_neighbour_pairs()))
        if j2 != 0.0:
            values["XXdiag_sum"] = values["HJ2"] / j2
            values["XXdiag_avg"] = values["XXdiag_sum"] / max(1, len(model.next_nearest_neighbour_pairs()))
    else:
        values.setdefault("H0", float("nan"))
    values.update({
        "joint_norm": joint_norm,
        "sys_norm_before_renorm": sys_norm,
    })
    return values


def main() -> None:
    nx = int(os.environ.get("NX", "5"))
    ny = int(os.environ.get("NY", "6"))
    nb = int(os.environ.get("NB", "1"))
    j = float(os.environ.get("J", "-0.5"))
    j2 = float(os.environ.get("J2", "0.0"))
    g = float(os.environ.get("G", "1.0"))
    gx = float(os.environ.get("GX", "0.0"))
    beta = float(os.environ.get("BETA", "0.2"))
    pbc_x = env_bool("PBC_X", "0")
    pbc_y = env_bool("PBC_Y", "0")
    n_resets = int(os.environ.get("N_RESETS", "100"))
    seed = int(os.environ.get("SEED", "42"))
    simulator_backend = os.environ.get("SIM_BACKEND", os.environ.get("SIMULATOR_BACKEND", "qsim")).lower()
    threads = int(os.environ.get("QSIM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    use_gpu = env_bool("QSIM_USE_GPU", "1")
    gpu_mode = int(os.environ.get("QSIM_GPU_MODE", "1"))
    max_fused = int(os.environ.get("QSIM_MAX_FUSED", "5"))
    log_every = int(os.environ.get("LOG_EVERY", "5"))
    measure_every = int(os.environ.get("MEASURE_EVERY", "0"))
    measure_start = int(os.environ.get("MEASURE_START", os.environ.get("BURN_IN", "0")))
    energy_every = int(os.environ.get("ENERGY_EVERY", os.environ.get("MEASURE_ENERGY_EVERY", str(measure_every))))
    corr_every = int(os.environ.get("CORR_EVERY", os.environ.get("MEASURE_CORRELATIONS_EVERY", str(measure_every))))
    skip_energy = env_bool("SKIP_ENERGY", "0")
    split_energy_components = env_bool("SPLIT_ENERGY_COMPONENTS", "0")
    measure_correlations = env_bool("MEASURE_CORRELATIONS", "0")
    theta = float(os.environ.get("THETA", "0.8"))
    default_delta = float(0.1 * np.pi / 2.0)
    delta = float(os.environ.get("DELTA", default_delta * float(os.environ.get("DELTA_SCALE", "1.0"))))
    nt = int(os.environ.get("NT", "5"))
    h_bath = float(os.environ.get("BATH_H", max(2 * abs(g), 4 * abs(j))))
    randomize_couplings = env_bool("RANDOMIZE_COUPLINGS", "1")
    randomization_time = float(os.environ.get("RANDOMIZATION_TIME", "0.0"))
    if measure_every <= 0:
        measure_every = log_every
    if energy_every <= 0:
        energy_every = measure_every
    if corr_every <= 0:
        corr_every = measure_every

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    lattice = a_device.RectLattice2D(nx, ny, pbc_x=pbc_x, pbc_y=pbc_y)
    device = a_device.Device(lattice, nb)
    model = b_model.IsingModel(device, {"J": j, "J2": j2, "g": g, "gx": gx})

    protocol = c_protocol.SimpleThermalProtocol(
        device=device,
        model=model,
        params={
            "beta": beta,
            "theta": theta,
            "h": h_bath,
            "delta": delta,
            "NT": nt,
            "randomize_couplings": randomize_couplings,
            "randomization_time": randomization_time,
        },
        frozen_circuit=False,
    )

    all_qubits = list(device.system_qubits) + list(device.bath_qubits)
    prep_circuit = cirq.Circuit()
    prep = sample_basis_preparation_ops(all_qubits, rng)
    if prep:
        prep_circuit.append(prep)
    prep_circuit.append(protocol.reset_layer)

    measurement = build_measurement(device, model, split_energy_components)

    if simulator_backend == "cirq":
        sim = cirq.Simulator(dtype=np.complex64, seed=seed)
    elif simulator_backend == "qsim":
        options = qsimcirq.QSimOptions(
            use_gpu=use_gpu,
            gpu_mode=gpu_mode,
            cpu_threads=threads,
            max_fused_gate_size=max_fused,
            verbosity=0,
        )
        sim = qsimcirq.QSimSimulator(options, seed=seed)
    else:
        raise ValueError(f"Unsupported SIM_BACKEND={simulator_backend!r}; use 'qsim' or 'cirq'")

    total_sim_runtime = 0.0
    last_measurement: dict[str, float] | None = None

    t0 = time.time()
    prep_result = sim.simulate(prep_circuit, initial_state=0)
    total_sim_runtime += time.time() - t0
    state = result_state_vector(prep_result)

    for cycle in range(1, n_resets + 1):
        cycle_circuit = protocol.cycle_circuit(cycle)
        t1 = time.time()
        result = sim.simulate(cycle_circuit, initial_state=state)
        total_sim_runtime += time.time() - t1
        should_log = cycle % log_every == 0 or cycle == n_resets
        reached_measure_start = cycle >= measure_start
        should_energy_measure = (
            reached_measure_start
            and not skip_energy
            and (cycle % energy_every == 0 or cycle == n_resets)
        )
        should_corr_measure = (
            reached_measure_start
            and measure_correlations
            and not skip_energy
            and (cycle % corr_every == 0 or cycle == n_resets)
        )
        should_state_ready = (
            reached_measure_start
            and skip_energy
            and (cycle % measure_every == 0 or cycle == n_resets)
        )
        should_measure = should_energy_measure or should_corr_measure or should_state_ready
        if should_log:
            print(
                json.dumps(
                    {
                        "cycle": cycle,
                        "n_resets": n_resets,
                        "beta": beta,
                        "runtime_sec": total_sim_runtime,
                        "stage": "simulate_done",
                    }
                ),
                flush=True,
            )
        if should_measure:
            state = result_state_vector(result)
            if skip_energy:
                print(
                    json.dumps(
                        {
                            "cycle": cycle,
                            "n_resets": n_resets,
                            "beta": beta,
                            "runtime_sec": total_sim_runtime,
                            "stage": "state_ready",
                            "energy_skipped": True,
                        }
                    ),
                    flush=True,
                )
            else:
                summary = measurement_summary(state, device, model, measurement, split_energy_components, should_corr_measure)
                last_measurement = summary
                payload = {
                    "cycle": cycle,
                    "n_resets": n_resets,
                    "beta": beta,
                    "energy": summary["H0"],
                    "runtime_sec": total_sim_runtime,
                    "joint_norm": summary["joint_norm"],
                    "sys_norm_before_renorm": summary["sys_norm_before_renorm"],
                    "stage": "measured",
                    "measured_correlations": should_corr_measure,
                }
                if split_energy_components:
                    payload.update(
                        {
                            "energy_HJ": summary["HJ"],
                            "energy_HJ2": summary.get("HJ2"),
                            "energy_Hg": summary["Hg"],
                            "energy_Hgx": summary["Hgx"],
                            "Ztot": summary.get("Ztot"),
                            "Zavg": summary.get("Zavg"),
                            "XXbond_sum": summary.get("XXbond_sum"),
                            "XXbond_avg": summary.get("XXbond_avg"),
                            "XXdiag_sum": summary.get("XXdiag_sum"),
                            "XXdiag_avg": summary.get("XXdiag_avg"),
                        }
                    )
                if should_corr_measure:
                    payload.update({key: val for key, val in summary.items() if key.startswith("Cxx_") or key.startswith("Sxx0_")})
                print(json.dumps(payload), flush=True)
        else:
            state = result_state_vector(result)

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if skip_energy:
        final_summary = {
            "H0": float("nan"),
            "joint_norm": float(np.linalg.norm(state)),
            "sys_norm_before_renorm": float(np.linalg.norm(state.reshape(2**device.Ns, 2**device.Nb)[:, 0])),
        }
    elif last_measurement is not None:
        final_summary = last_measurement
    else:
        final_summary = measurement_summary(state, device, model, measurement, split_energy_components, measure_correlations)
    final_payload = {
        "hostname": os.uname().nodename,
        "nx": nx,
        "ny": ny,
        "nb": nb,
        "beta": beta,
        "pbc_x": pbc_x,
        "pbc_y": pbc_y,
        "J": j,
        "J2": j2,
        "g": g,
        "gx": gx,
        "n_resets": n_resets,
        "seed": seed,
        "randomize_couplings": randomize_couplings,
        "simulator_backend": simulator_backend,
        "theta": theta,
        "bath_h": h_bath,
        "delta": delta,
        "NT": nt,
        "MT": protocol.MT,
        "protocol_T": protocol.T,
        "randomization_time": randomization_time,
        "threads": threads,
        "use_gpu": use_gpu,
        "gpu_mode": gpu_mode,
        "log_every": log_every,
        "measure_every": measure_every,
        "energy_every": energy_every,
        "corr_every": corr_every,
        "measure_start": measure_start,
        "skip_energy": skip_energy,
        "split_energy_components": split_energy_components,
        "measure_correlations": measure_correlations,
        "runtime_sec": total_sim_runtime,
        "H0_final": final_summary["H0"],
        "HJ_final": final_summary.get("HJ"),
        "HJ2_final": final_summary.get("HJ2"),
        "Hg_final": final_summary.get("Hg"),
        "Hgx_final": final_summary.get("Hgx"),
        "Ztot_final": final_summary.get("Ztot"),
        "Zavg_final": final_summary.get("Zavg"),
        "XXbond_sum_final": final_summary.get("XXbond_sum"),
        "XXbond_avg_final": final_summary.get("XXbond_avg"),
        "XXdiag_sum_final": final_summary.get("XXdiag_sum"),
        "XXdiag_avg_final": final_summary.get("XXdiag_avg"),
        "Sxx0_axial_final": final_summary.get("Sxx0_axial"),
        "joint_norm": final_summary["joint_norm"],
        "sys_norm_before_renorm": final_summary["sys_norm_before_renorm"],
        "ru_maxrss_kb": ru,
        "ru_maxrss_mb": ru / 1024.0,
    }
    print(json.dumps(final_payload, indent=2))


if __name__ == "__main__":
    main()
