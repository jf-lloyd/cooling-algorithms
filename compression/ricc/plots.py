"""Plot functions for the RI cooling-cycle compression results.

Every function takes a matplotlib axis and reads from the pre-computed JSONs in
`data/`, so the whole notebook runs in seconds with no GPU and no optimisation.
Re-generate the underlying JSONs with the drivers in `scripts/`.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

C1, C2, CV, CN = "#888888", "#0057b7", "#c1272d", "#111111"  # 1st, 2nd, variational, native


def load(name):
    with open(DATA / name) as f:
        return json.load(f)


def _curve(d, key, Gg):
    """{layers: infidelity} -> (2q-gate count array, infidelity array), sorted."""
    ks = sorted(int(k) for k in d[key])
    return np.array([Gg * k for k in ks]), np.array([d[key][str(k)] for k in ks])


def gates_to_target(d, key, Gg, target=0.01):
    """2q-gates needed to first reach `target` infidelity (log-linear interpolation).

    Returns np.nan if the curve never gets there. Uses the FIRST crossing and
    requires the curve to be descending into it, so a variational curve that
    stalled at its Trotter initialisation cannot report a spuriously good number.
    """
    g, e = _curve(d, key, Gg)
    below = np.where(e <= target)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return float(g[0])
    g0, g1, e0, e1 = g[i - 1], g[i], e[i - 1], e[i]
    if not (e0 > target >= e1):
        return float(g[i])
    f = (np.log(e0) - np.log(target)) / (np.log(e0) - np.log(e1))
    return float(g0 + f * (g1 - g0))


# ---------------------------------------------------------------- plot 1
def plot_gatecount_vs_infidelity(ax, lattice="3x4", target=0.01):
    """Infidelity vs 2q-gate count: 1st order, 2nd order, variational."""
    d = load(f"cc_trot_{lattice}.json")
    v = load("cc_cycle_beta_var.json")["1.0"]
    Gg = d["Gg"]

    for key, c, lab in ((("t1"), C1, "1st-order Trotter"), (("t2"), C2, "2nd-order Trotter")):
        g, e = _curve(d, key, Gg)
        ax.plot(g, e, "o-", color=c, ms=4, lw=1.8, label=lab)
    if lattice == "3x4":
        g, e = _curve(v, "var", Gg)
        ax.plot(g, e, "s-", color=CV, ms=5, lw=2, label="variational (optimised angles)")

    ax.axhline(target, color="0.4", ls=":", lw=1.2)
    ax.text(0.02, target * 1.25, f"{target:.0%} infidelity", transform=ax.get_yaxis_transform(),
            fontsize=8, color="0.35")
    ax.axvline(d["native"], color=CN, ls="--", lw=1.4)
    ax.text(d["native"], 0.5, f" native cycle\n {d['native']} gates", color=CN, fontsize=8, va="top")
    ax.axhline(d["eps_nat"], color=CN, ls="-.", lw=1, alpha=0.6)
    ax.text(0.02, d["eps_nat"] * 0.55, "native cycle's own error (floor)",
            transform=ax.get_yaxis_transform(), fontsize=7.5, color=CN, alpha=0.8)

    ax.set_yscale("log")
    ax.set_xlabel("2-qubit gates per cooling cycle")
    ax.set_ylabel("channel infidelity vs ideal cycle")
    ax.set_title(f"Compressing one cooling cycle ({lattice}+1, "
                 rf"$\beta$=1)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25)

    return {"native": d["native"],
            "t1": gates_to_target(d, "t1", Gg, target),
            "t2": gates_to_target(d, "t2", Gg, target),
            "var": gates_to_target(v, "var", Gg, target) if lattice == "3x4" else np.nan}


# ---------------------------------------------------------------- plot 2
def plot_trotter_order_rule(ax):
    """2nd-order 2q-gate overhead vs number of non-commuting 2-body classes C.

    Rule: overhead = (2C-2)/C  (one class can always merge across the boundary).
    Ising C=1 -> free; 1D Heisenberg C=2 -> free; 2D Heisenberg C=4 -> 1.5x.
    """
    h = load("cc_heisenberg.json")
    meas = []
    for name, mark in (("1D", "o"), ("2D", "s")):
        e = h[name]
        r = e["res"][-1]
        meas.append((e["C"], r["g2"] / r["g1"], f"Heisenberg {name}\n(C={e['C']})", mark))
    meas.append((1, 1.0, "Ising (C=1)\n2nd order free", "^"))

    Cs = np.arange(1, 6)
    ax.plot(Cs, np.maximum((2 * Cs - 2) / Cs, 1.0), "-", color="0.5", lw=1.6,
            label=r"rule  $\max(1,\,(2C-2)/C)$")
    for C, ov, lab, mk in meas:
        ax.plot(C, ov, mk, ms=11, color=C2 if C > 1 else CV, zorder=5)
        ax.annotate(lab, (C, ov), textcoords="offset points", xytext=(9, -4), fontsize=8)

    ax.set_xlabel("C  =  mutually non-commuting 2-body classes")
    ax.set_ylabel("2nd-order 2q-gate overhead")
    ax.set_title("When is 2nd-order Trotter free?", fontsize=10.5)
    ax.set_xticks(Cs); ax.set_ylim(0.9, 1.8); ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left")
    return meas


# ---------------------------------------------------------------- plot 3
def plot_beta_sweep(ax, target=0.01):
    """Gates-to-1% vs beta for 2nd-order and variational.

    At beta >= 2 the optimiser returns its Trotter initialisation unchanged (the recorded
    variational values equal the Trotter ones exactly), so it never beats Trotter there;
    those points are marked rather than plotted as a variational win.
    """
    b = load("cc_cycle_beta_var.json")
    d3 = load("cc_trot_3x4.json")
    Gg = d3["Gg"]
    betas = sorted(b, key=float)
    x = [float(t) for t in betas]
    g_nat = [b[t]["native"] for t in betas]
    g_t2 = [gates_to_target(b[t], "t2", Gg, target) for t in betas]
    g_var = [gates_to_target(b[t], "var", Gg, target) for t in betas]

    ax.plot(x, g_nat, "d--", color=CN, ms=6, lw=1.4, label="native cycle")
    ax.plot(x, g_t2, "o-", color=C2, ms=5, lw=1.8, label="2nd-order Trotter")
    ok = [i for i, v in enumerate(g_var) if np.isfinite(v)]
    ax.plot([x[i] for i in ok], [g_var[i] for i in ok], "s-", color=CV, ms=6, lw=2,
            label="variational")
    bad = [i for i, v in enumerate(g_var) if not np.isfinite(v)]
    for i in bad:
        ax.plot(x[i], g_t2[i], "x", color=CV, ms=11, mew=2.5)
    if bad:
        ax.annotate("variational stalls at its\nTrotter init (no gain)", (x[bad[0]], g_t2[bad[0]]),
                    textcoords="offset points", xytext=(-12, 26), fontsize=8, color=CV,
                    ha="center", arrowprops=dict(arrowstyle="->", color=CV, lw=1.2))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"inverse temperature  $\beta$")
    ax.set_ylabel(f"2q-gates to reach {target:.0%} infidelity")
    ax.set_title(r"Compression vs temperature", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.25, which="both")
    return dict(zip(betas, zip(g_nat, g_t2, g_var)))


# ---------------------------------------------------------------- plot 4  (the punchline)
def plot_channel_vs_unitary(axL, axR, series="n3"):
    """THE result: unitary fidelity stays ~0 while the COOLED STATE is reproduced.

    Because the cycle ends in a bath reset+trace, many different unitaries realise
    the same channel -- so a shallow ansatz can match the *channel* (what cooling
    depends on) without matching the *unitary* at all.
    """
    t = load("t1_channel_results.json")
    rows = sorted([v for k, v in t.items() if k.startswith(series) and v.get("F_chan_gibbs")],
                  key=lambda r: r["LV"])
    LV = [r["LV"] for r in rows]
    fg = [r.get("gate_fid") or r.get("Fgate") for r in rows]
    fgibbs = [r["F_chan_gibbs"] for r in rows]
    ratio = [r["n2q_target"] / r["n2q_ansatz"] for r in rows]

    axL.bar(LV, fg, color=C1, width=0.6)
    axL.set_ylim(0, 1.16); axL.set_xticks(LV)
    axL.set_xlabel(r"ansatz depth  $L_V$"); axL.set_ylabel("unitary (gate) fidelity")
    axL.set_title("(a) does the compressed cycle\nmatch the UNITARY?  no", fontsize=10)
    for x, y in zip(LV, fg):
        axL.text(x, y + 0.03, f"{y:.3f}", ha="center", fontsize=8.5, color=C1)
    axL.axhline(1.0, color="0.6", ls=":", lw=1)
    axL.grid(alpha=0.25, axis="y")

    axR.bar(LV, fgibbs, color=CV, width=0.6)
    axR.set_ylim(0, 1.16); axR.set_xticks(LV)
    axR.set_xlabel(r"ansatz depth  $L_V$")
    axR.set_ylabel(r"fidelity of cooled state to $\rho_{Gibbs}$")
    axR.set_title("(b) does it still COOL to the\nright state?  yes", fontsize=10)
    for x, y, r in zip(LV, fgibbs, ratio):
        axR.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=9, color=CV, fontweight="bold")
        axR.text(x, 0.06, f"{r:.1f}x\nfewer", ha="center", fontsize=7.5, color="white")
    axR.axhline(1.0, color="0.6", ls=":", lw=1)
    axR.grid(alpha=0.25, axis="y")
    return list(zip(LV, fg, fgibbs, ratio))


# ---------------------------------------------------------------- plot 5
def plot_resource_projection(ax, target=0.01):
    """Native vs compressed 2q-gates/cycle -- and that the ratio is size-independent."""
    d3, d4 = load("cc_trot_3x4.json"), load("cc_trot_4x4.json")
    v = load("cc_cycle_beta_var.json")["1.0"]
    sizes, nat, t2g, varg = [], [], [], []
    for d, lab in ((d3, "3x4+1\n(13q)"), (d4, "4x4+1\n(17q)")):
        sizes.append(lab); nat.append(d["native"])
        t2g.append(gates_to_target(d, "t2", d["Gg"], target))
        # variational measured at 3x4; the ratio is size-independent (see notebook)
        varg.append(gates_to_target(v, "var", d3["Gg"], target) / d3["native"] * d["native"])

    x = np.arange(len(sizes)); w = 0.27
    ax.bar(x - w, nat, w, color=CN, label="native cycle")
    ax.bar(x, t2g, w, color=C2, label=f"2nd-order @ {target:.0%}")
    ax.bar(x + w, varg, w, color=CV, label=f"variational @ {target:.0%}")
    for xi, (a, b, c) in enumerate(zip(nat, t2g, varg)):
        ax.text(xi - w, a + 12, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(xi, b + 12, f"{b:.0f}\n({a/b:.1f}x)", ha="center", fontsize=8, color=C2)
        ax.text(xi + w, c + 12, f"{c:.0f}\n({a/c:.1f}x)", ha="center", fontsize=8, color=CV)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_ylabel("2-qubit gates per cooling cycle")
    ax.set_title("Resource cost per cycle", fontsize=10.5)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, max(nat) * 1.25)
    return list(zip(sizes, nat, t2g, varg))
