"""
study_thermal.py -- Joule heating, and its effect on the CERTIFIED read-out.

Answers Reviewer 2 comment 3.  The first submission bounded a temperature rise
and stopped there; the reviewer asks for the step that turns a temperature into
something a measurement would show.  That step has also changed meaning, because
the certified read-out in a physiological buffer is now the conductance
(study_edl.py), and conductance and capacitance drift with temperature in
opposite directions and by very different amounts:

    d(eps_r)/dT / eps_r  =  -0.36 %/K   (Malmberg-Maryott, water near 25 C)
    d(sigma)/dT / sigma  =  +2.0  %/K   (ionic mobility of a physiological buffer)

So the read-out the revision certifies is about five and a half times MORE
temperature sensitive than the one the first submission certified.  That is the
opposite of a convenient result and it is reported as such.

What is certified here, and what is not
---------------------------------------
CERTIFIED (level 1, the bracket):
  * the dissipated power, P = G V^2, because G is certified;
  * the thermal conductance from the electrode plane to the substrate sink,
    which is the same elliptic problem with the thermal conductivity as
    coefficient, so the same dual bracket applies.
MODEL (level 2, stated assumption):
  * placing the heat source on the electrode plane.  Joule heat is really
    generated throughout the fluid where the field is strong, and heat released
    higher in the channel sees a LARGER thermal resistance, so this assumption
    UNDER-estimates the rise.  It is a model choice, not a bound, and the paper
    says so rather than presenting the resulting temperature as certified.

Outputs: thermal_data.json and figures/fig_thermal.png
"""

import json

import numpy as np

import device as dv
import figstyle as fs
from certified_core import capacitance_bracket, Bracket, EPS0, HOT, GND
from geometry import Cell, _n

# ---- material data --------------------------------------------------------
K_WATER = 0.60             # W/(m K)
K_GLASS = 1.10             # W/(m K), borosilicate
K_METAL = 300.0            # W/(m K), gold
SIGMA_PBS = 1.6            # S/m
T_AMBIENT = 25.0           # C
T_LIMIT_CELL = 37.0 + 5.0  # C, mammalian cells tolerate a few K above 37
T_LIMIT_PROTEIN = 45.0     # C, onset of denaturation for many proteins

DEPS_DT = -0.0036          # per K, relative
DSIG_DT = +0.020           # per K, relative

H_TH = 100e-9


def thermal_cell(h_sub, h=H_TH):
    """Quarter cell for the heat problem: electrode plane to substrate sink.

    Same voxel geometry as the electrical problem, but the boundary conditions
    are thermal: the metal is the source (it is where the dissipation is
    concentrated) and the bottom of the substrate is the heat sink.  Passing
    k/eps0 as the coefficient makes the solver return a thermal conductance in
    W/(K m).

    h_sub is a SEPARATE parameter from the electrical substrate depth, and this
    is the point of the sweep below.  The electrical model stops 5 um under the
    electrodes because the field does not reach further, but heat does: the sink
    is whatever the chip is mounted on, typically the far face of a 0.5 to 1 mm
    wafer.  Taking the electrical 5 um for the thermal problem would put the sink
    a hundred times too close and under-estimate every temperature.  The depth is
    therefore treated as a bounded datum and swept.
    """
    nx = _n(dv.LAM / 2, h)
    ns = _n(h_sub, h)
    nf = _n(dv.H_CHANNEL, h)
    nm = max(_n(dv.T_METAL, h), 1)
    nw = _n(dv.ETA * dv.LAM / 2, h)
    ny = ns + nf

    k = np.empty((ny, nx))
    k[:ns] = K_GLASS
    k[ns:] = K_WATER
    k[ns:ns + nm, :nw] = K_METAL

    elec = np.zeros((ny, nx), dtype=int)
    elec[ns:ns + nm, :nw] = HOT          # source: the metal
    elec[0, :] = GND                     # sink: the bottom of the substrate

    return Cell(k / EPS0, elec, (h, h), 1.0)


def certified_thermal_conductance(h_sub, h=H_TH):
    """Two-sided enclosure of the electrode-to-sink thermal conductance.

    Returned per PERIOD and per metre of finger: the quarter cell spans lam/2,
    and a period is two of them by mirror symmetry, so the period conductance is
    twice the quarter-cell value.  Unlike the electrical case there is no
    antisymmetry and hence no factor of one half: the source is at one potential
    and the sink at the other, with no intermediate plane.
    """
    c = thermal_cell(h_sub, h)
    br = capacitance_bracket(*c.as_args())
    return Bracket(2 * br.lo, 2 * br.hi)


# Substrate depths to the heat sink, with a mesh coarse enough to keep each
# solve affordable while still resolving the electrode pitch.
SUBSTRATES = [(5e-6, 100e-9), (50e-6, 250e-9), (200e-6, 500e-9),
              (500e-6, 1e-6), (1000e-6, 2e-6)]


def main():
    fs.use_paper_style()
    import matplotlib.pyplot as plt

    out = {"k_water": K_WATER, "k_glass": K_GLASS, "sigma_pbs": SIGMA_PBS,
           "deps_dT": DEPS_DT, "dsig_dT": DSIG_DT, "T_ambient": T_AMBIENT}

    # ---------------------------------------------------------------- power
    G_cell = dv.cell_capacitance(dv.H_FINE, eps_fluid=SIGMA_PBS / EPS0,
                                 eps_sub=dv.SIGMA_GLASS / EPS0)
    G_dev = dv.device_capacitance(G_cell)
    print("certified conductance of the reference chip")
    print(f"  per period : [{G_cell.lo*1e3:.6f}, {G_cell.hi*1e3:.6f}] mS/m")
    print(f"  device     : [{G_dev.lo*1e3:.6f}, {G_dev.hi*1e3:.6f}] mS"
          f"   (+/-{G_dev.relwidth:.4f} %)")
    out["G_device"] = [G_dev.lo, G_dev.hi]

    # ---------------------------------------------------------------- thermal
    print("\ncertified thermal conductance vs depth to the heat sink")
    print(f"  {'h_sub':>8s} {'mesh':>7s} {'K_th device (mW/K)':>26s} "
          f"{'R_th (K/W)':>14s} {'+/- %':>8s}")
    subs = []
    for hs, hm in SUBSTRATES:
        kc = certified_thermal_conductance(hs, hm)
        kd = Bracket(kc.lo * dv.DEPTH * dv.N_PERIODS,
                     kc.hi * dv.DEPTH * dv.N_PERIODS)
        subs.append((hs, kd))
        print(f"  {hs*1e6:7.0f}u {hm*1e9:6.0f}n "
              f"{kd.lo*1e3:12.4f} - {kd.hi*1e3:9.4f} "
              f"{1/kd.hi:6.2f} - {1/kd.lo:5.2f} {kd.relwidth:8.4f}")
    out["substrates"] = [dict(h_sub=hs, Kth_lo=k.lo, Kth_hi=k.hi)
                         for hs, k in subs]

    # The headline case is a real chip: a 500 um wafer.  The 5 um figure is kept
    # only to show what assuming the electrical depth would have done.
    Kth_dev = dict(subs)[500e-6]
    Kth_thin = dict(subs)[5e-6]
    print(f"\n  taking the electrical 5 um depth instead of a 500 um wafer "
          f"would understate every temperature by a factor "
          f"{Kth_thin.lo/Kth_dev.hi:.0f}")
    out["Kth_device"] = [Kth_dev.lo, Kth_dev.hi]
    out["understatement_factor"] = float(Kth_thin.lo / Kth_dev.hi)

    # ---------------------------------------------------------------- rise
    # The drift coefficients are first-order expansions about 25 C, so the sweep
    # stops where the rise is still small enough for them to mean anything.  At
    # 5 V this device would compute a 217 K rise, which says only that the model
    # has left its domain long before: quoting it would be meaningless.
    V = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    DT_MODEL_MAX = 20.0        # K, beyond which the linearised drift is void
    dT_lo = G_dev.lo * V ** 2 / Kth_dev.hi
    dT_hi = G_dev.hi * V ** 2 / Kth_dev.lo
    print("\nsteady temperature rise (source on the electrode plane: a MODEL,")
    print("and one that under-estimates, since heat released higher in the")
    print("channel would see a larger thermal resistance)")
    print(f"  {'V (V)':>7s} {'P (mW)':>12s} {'dT (K)':>22s} {'T (C)':>10s}")
    for i, v in enumerate(V):
        P = 0.5 * (G_dev.lo + G_dev.hi) * v ** 2
        flag = "" if dT_hi[i] <= DT_MODEL_MAX else "  <- drift model void"
        print(f"  {v:7.2f} {P*1e3:12.4f} {dT_lo[i]:10.4f} - {dT_hi[i]:8.4f}"
              f" {T_AMBIENT+dT_hi[i]:10.2f}{flag}")
    out["V"] = V.tolist()
    out["dT_lo"], out["dT_hi"] = dT_lo.tolist(), dT_hi.tolist()

    # safe voltage for each tolerance
    def v_max(dT_allowed):
        return float(np.sqrt(dT_allowed * Kth_dev.lo / G_dev.hi))

    v_cell = v_max(T_LIMIT_CELL - T_AMBIENT)
    v_prot = v_max(T_LIMIT_PROTEIN - T_AMBIENT)
    print(f"\n  safe read-out voltage, cells below {T_LIMIT_CELL:.0f} C : "
          f"{v_cell:.3f} V")
    print(f"  safe read-out voltage, proteins below {T_LIMIT_PROTEIN:.0f} C : "
          f"{v_prot:.3f} V")
    out["v_safe_cell"], out["v_safe_protein"] = v_cell, v_prot

    # ------------------------------------------------------- read-out drift
    print("\ndrift of the two read-outs, and why the channel choice matters")
    print(f"  {'V (V)':>7s} {'dT (K)':>10s} {'dG/G (%)':>12s} {'dC/C (%)':>12s}"
          f" {'ratio':>8s}")
    dG, dC = [], []
    for i, v in enumerate(V):
        t = 0.5 * (dT_lo[i] + dT_hi[i])
        g = 100 * DSIG_DT * t
        c = 100 * DEPS_DT * t
        dG.append(g)
        dC.append(c)
        print(f"  {v:7.2f} {t:10.4f} {g:+12.4f} {c:+12.4f} "
              f"{abs(g/c) if c else float('nan'):8.2f}")
    out["drift_G_pc"], out["drift_C_pc"] = dG, dC

    # at what voltage does the thermal drift exceed the certified width,
    # the fabrication envelope, and a 0.1 % read-out floor?
    width = G_dev.relwidth / 100.0
    thresholds = [("numerical certificate", width),
                  ("0.1 % read-out floor", 1e-3),
                  ("fabrication envelope", 0.0138)]
    print("\n  voltage at which the thermal drift of G reaches ...")
    out["drift_crossings"] = {}
    for name, thr in thresholds:
        # |dsig/dT| * dT(V) = thr  with dT = G V^2 / Kth
        v_star = float(np.sqrt(thr / DSIG_DT * Kth_dev.lo / G_dev.hi))
        out["drift_crossings"][name] = v_star
        print(f"    {name:24s} ({100*thr:6.3f} %) : {v_star:6.3f} V")

    # ---------------------------------------------------------------- figure
    fig, axes = fs.row(3, height=3.05)

    ax = axes[0]
    ax.fill_between(V, dT_lo, dT_hi, color=fs.C_BAND, alpha=0.65, lw=0,
                    label="certified band (narrower\nthan the line width)")
    ax.plot(V, dT_hi, color=fs.C_UP, lw=1.4)
    ax.plot(V, dT_lo, color=fs.C_LO, lw=1.4)
    for lim, lab, col, dy in [(T_LIMIT_CELL - T_AMBIENT, "cells at 42 C",
                               fs.PALETTE[3], -9),
                              (T_LIMIT_PROTEIN - T_AMBIENT, "proteins at 45 C",
                               fs.PALETTE[4], 3)]:
        ax.axhline(lim, color=col, ls="--", lw=1.1)
        ax.annotate(lab, xy=(V[-1], lim), xytext=(-3, dy), ha="right",
                    textcoords="offset points", fontsize=7.2, color=col)
    ax.set_yscale("log")
    ax.set_xlabel("read-out voltage  (V)")
    ax.set_ylabel("steady temperature rise  (K)")
    fs.legend(ax, loc="lower right")

    ax = axes[1]
    ax.plot(V, np.abs(dG), "o-", color=fs.PALETTE[0], lw=1.5, ms=4,
            label="conductance, $+2$ %/K")
    ax.plot(V, np.abs(dC), "s-", color=fs.PALETTE[1], lw=1.5, ms=4,
            label="capacitance, $-0.36$ %/K")
    ax.axhline(100 * width, color="k", ls=":", lw=1.2)
    ax.annotate("numerical certificate", xy=(V[-1], 100 * width),
                xytext=(-3, 3), textcoords="offset points", ha="right",
                fontsize=7.2)
    ax.axhline(1.38, color="0.45", ls="--", lw=1.2)
    ax.annotate("fabrication envelope", xy=(V[-1], 1.38), xytext=(-3, 3),
                textcoords="offset points", ha="right", fontsize=7.2)
    ax.set_yscale("log")
    ax.set_xlabel("read-out voltage  (V)")
    ax.set_ylabel("thermal drift of the read-out  (%)")
    fs.legend(ax, loc="upper left")

    ax = axes[2]
    # A stacked budget would be wrong here: these terms are not additive, and
    # the numerical certificate is invisible next to the others anyway.  What a
    # designer needs is the ceiling each constraint puts on the read-out
    # voltage, so that is what the panel shows.
    limits = [
        ("cells stay below 42 C", out["v_safe_cell"], fs.PALETTE[3]),
        ("proteins below 45 C", out["v_safe_protein"], fs.PALETTE[4]),
        ("drift < fabrication envelope",
         out["drift_crossings"]["fabrication envelope"], "0.55"),
        ("drift < 0.1 % read-out floor",
         out["drift_crossings"]["0.1 % read-out floor"], fs.PALETTE[1]),
        ("drift < numerical certificate",
         out["drift_crossings"]["numerical certificate"], fs.C_LO),
    ]
    ypos = np.arange(len(limits))
    ax.barh(ypos, [v for _, v, _ in limits],
            color=[c for _, _, c in limits], height=0.66, alpha=0.85)
    for i, (_, v, _) in enumerate(limits):
        ax.annotate(f"{v:.2f} V", xy=(v, ypos[i]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels([n for n, _, _ in limits], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.03, 6.0)
    ax.set_xlabel("voltage ceiling  (V)")
    ax.set_title("what limits the read-out", fontsize=8.5, pad=3)

    fs.save(fig, "figures/fig_thermal.png")

    with open("thermal_data.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote thermal_data.json")


if __name__ == "__main__":
    main()
