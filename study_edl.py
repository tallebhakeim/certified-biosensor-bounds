"""
study_edl.py -- which observable does the bracket certify, and in what frequency
window?

Answers Reviewer 2 comments 6 and 7 (electrode polarisation and ionic-strength
drift), and it is also the honest answer to comment 1 ("validated numerical range
versus physical reality").

The physics the first submission left implicit
----------------------------------------------
A buffer is a CONDUCTOR as well as a dielectric.  Its complex permittivity is

    eps* = eps0 eps_r - j sigma / omega ,

so the fluid stops behaving as a dielectric below its dielectric relaxation
frequency

    f_diel = sigma / (2 pi eps0 eps_r) ,

which for phosphate-buffered saline (sigma = 1.6 S/m, eps_r = 78) is 369 MHz.
Below it the current through the channel is conduction, not displacement: a
"capacitance" measured there is not the geometric capacitance of the cell.  At
the other end, the double layer at each electrode adds a large series
capacitance C_dl which short-circuits the geometry below

    f_edl = G / (pi C_dl) .

Between the two the cell is a RESISTOR whose conductance is set by the same
geometry.

What survives, and why the framework is unaffected
--------------------------------------------------
Both observables are the same elliptic problem with a different coefficient:

    C = eps0 eps_r k_C     (Laplace with eps, insulating walls)
    G = sigma     k_G      (Laplace with sigma, sigma_glass ~ 0)

so the dual-TLM bracket certifies the GEOMETRIC FACTOR, and the certified
capacitance and the certified conductance are two readings of one certificate.
Nothing in Sections II-IV changes; what changes is that the paper must name the
observable it certifies and the window in which that observable is the measured
one.  The certified detection theorem then applies to whichever of C or G the
instrument actually reads.

Three regimes, all certified by the same bracket:
    f < f_edl                : double layer masks the geometry -> no claim
    f_edl < f < f_diel       : conductance read-out, certified through k_G
    f > f_diel               : capacitance read-out, certified through k_C

Outputs: edl_data.json and figures/fig_edl.png
"""

import json
import numpy as np

import device as dv
import figstyle as fs
from certified_core import capacitance_bracket, Bracket, EPS0
from geometry import ide_quarter_cell

H = 100e-9

# ---- electrolytes ---------------------------------------------------------
# name, conductivity (S/m), relative permittivity, source of the conductivity
MEDIA = [
    ("PBS",              1.60, 78.0),
    ("blood",            0.69, 78.0),
    ("DEP medium",       0.055, 78.0),
    ("low-sigma buffer", 0.010, 78.0),
]

# double-layer areal capacitance, 10-50 uF/cm2 for a metal in aqueous
# electrolyte; 40 uF/cm2 is the value used by Claudel et al. for Au/Pt IDEs.
C_DL_AREAL = 40e-6 / 1e-4          # F/m2  (40 uF/cm2)
C_DL_RANGE = (10e-6 / 1e-4, 50e-6 / 1e-4)

SIGMA_GLASS_RATIO = 1e-6           # glass against buffer: a true insulator


def geometric_factors(h=H, eta=dv.ETA, eps_fluid=78.0):
    """Certified brackets of the two geometric factors, in metres.

    k_C : electrostatic problem, glass at eps = 4.2, fluid at eps_fluid.
          C = eps0 * k_C  (k_C already carries the permittivities, so it is the
          capacitance in units of eps0 -- reported as C directly below).
    k_G : conduction problem, glass at sigma ~ 0, fluid at 1.
          G = sigma_fluid * k_G.

    Both come from the same solver and the same mesh; only the coefficient
    changes, which is exactly the point being made.
    """
    # capacitance: the physical cell
    brC = dv.cell_capacitance(h, eta=eta, eps_fluid=eps_fluid)

    # conductance: same geometry, sigma_fluid = 1, sigma_glass ~ 0.  The solver
    # is coefficient-agnostic, so we reuse it with sigma in place of eps and
    # divide out eps0 to recover a pure geometric factor in metres.
    cell = ide_quarter_cell(h, lam=dv.LAM, eta=eta, t_metal=dv.T_METAL,
                            h_sub=dv.H_SUB, h_fluid=dv.H_CHANNEL, depth=1.0,
                            eps_fluid=1.0, eps_sub=SIGMA_GLASS_RATIO)
    brG = capacitance_bracket(*cell.as_args())
    kG = Bracket(brG.lo / (2 * EPS0), brG.hi / (2 * EPS0))   # /2: quarter -> cell
    return brC, kG


def main():
    fs.use_paper_style()
    out = {"mesh_m": H, "c_dl_areal_F_per_m2": C_DL_AREAL,
           "c_dl_areal_range": list(C_DL_RANGE)}

    brC, kG = geometric_factors()
    C_dev = dv.device_capacitance(brC)
    kG_dev = Bracket(kG.lo * dv.DEPTH * dv.N_PERIODS,
                     kG.hi * dv.DEPTH * dv.N_PERIODS)

    # electrode area of the whole chip: two fingers per period
    w_finger = dv.ETA * dv.LAM
    A_elec = 2 * w_finger * dv.DEPTH * dv.N_PERIODS
    C_dl = C_DL_AREAL * A_elec
    C_dl_lo = C_DL_RANGE[0] * A_elec
    C_dl_hi = C_DL_RANGE[1] * A_elec

    print("reference chip, certified geometric factors")
    print(f"  C (PBS, electrostatic) : [{C_dev.lo*1e12:.4f}, "
          f"{C_dev.hi*1e12:.4f}] pF   (+/-{C_dev.relwidth:.4f} %)")
    print(f"  k_G (conduction)       : [{kG_dev.lo*1e3:.4f}, "
          f"{kG_dev.hi*1e3:.4f}] mm  (+/-{kG_dev.relwidth:.4f} %)")
    print(f"  electrode area         : {A_elec*1e4:.4f} cm^2")
    print(f"  C_dl at 40 uF/cm2      : {C_dl*1e6:.3f} uF "
          f"(= {C_dl/C_dev.mid:.3e} x C_cell)")
    out["C_device_F"] = [C_dev.lo, C_dev.hi]
    out["kG_device_m"] = [kG_dev.lo, kG_dev.hi]
    out["A_elec_m2"] = A_elec
    out["C_dl_F"] = C_dl

    print("\nfrequency windows per medium (certified through the same bracket)")
    rows = []
    for name, sigma, eps_r in MEDIA:
        G = Bracket(sigma * kG_dev.lo, sigma * kG_dev.hi)
        f_diel = sigma / (2 * np.pi * EPS0 * eps_r)
        f_edl = Bracket(G.lo / (np.pi * C_dl_hi), G.hi / (np.pi * C_dl_lo))
        # is there a window at all where the CAPACITANCE is the observable?
        usable = f_diel > f_edl.hi
        rows.append(dict(name=name, sigma=sigma, eps_r=eps_r,
                         G=[G.lo, G.hi], f_diel=f_diel,
                         f_edl=[f_edl.lo, f_edl.hi],
                         capacitive_window=bool(usable)))
        print(f"  {name:17s} sigma={sigma:6.3f} S/m  "
              f"G in [{G.lo*1e3:8.3f},{G.hi*1e3:8.3f}] mS  "
              f"f_edl in [{f_edl.lo/1e3:8.2f},{f_edl.hi/1e3:8.2f}] kHz  "
              f"f_diel = {f_diel/1e6:8.2f} MHz")
    out["media"] = rows

    # ---- the measured series capacitance across frequency -----------------
    # Z = 2/(j w C_dl) + 1/(G + j w C_cell); the instrument reports
    # C_meas = -1 / (w Im Z), which tends to C_dl/2 at low f and to C_cell at
    # high f.  The certified bracket on (C_cell, G) gives a certified band on
    # C_meas(f), which is what an experiment would actually see.
    freq = np.logspace(2, 10, 400)
    w = 2 * np.pi * freq
    bands = {}
    for name, sigma, eps_r in MEDIA:
        lo_curve, hi_curve = [], []
        for lo_hi in (0, 1):
            Cc = C_dev.lo if lo_hi == 0 else C_dev.hi
            Gg = sigma * (kG_dev.lo if lo_hi == 0 else kG_dev.hi)
            Cd = C_dl_lo if lo_hi == 0 else C_dl_hi
            Z = 2.0 / (1j * w * Cd) + 1.0 / (Gg + 1j * w * Cc)
            Cm = -1.0 / (w * np.imag(Z))
            (lo_curve if lo_hi == 0 else hi_curve).append(Cm)
        a = np.minimum(lo_curve[0], hi_curve[0])
        b = np.maximum(lo_curve[0], hi_curve[0])
        bands[name] = (a, b)
    out["freq_Hz"] = freq.tolist()
    out["Cmeas_bands"] = {k: [v[0].tolist(), v[1].tolist()]
                          for k, v in bands.items()}

    # ---- ionic strength as an uncertain datum ----------------------------
    # sigma is not known exactly (temperature, evaporation, sample dilution).
    # The certified conductance is monotone in sigma, so a box on sigma gives a
    # guaranteed band on G with no sampling -- the same rule as for eps.
    print("\nionic-strength drift: certified conductance over a sigma box")
    drift = []
    for rel in (0.01, 0.02, 0.05, 0.10):
        s_lo, s_hi = 1.6 * (1 - rel), 1.6 * (1 + rel)
        G = Bracket(s_lo * kG_dev.lo, s_hi * kG_dev.hi)
        drift.append(dict(rel=rel, G=[G.lo, G.hi], halfwidth_pc=G.relwidth))
        print(f"  sigma known to +/-{rel*100:4.1f} %  ->  G in "
              f"[{G.lo*1e3:8.3f},{G.hi*1e3:8.3f}] mS  (+/-{G.relwidth:5.2f} %)")
    out["sigma_drift"] = drift

    # ---------------------------------------------------------------- figure
    import matplotlib.pyplot as plt
    fig, axes = fs.row(3, height=3.05)

    ax = axes[0]
    for i, (name, sigma, eps_r) in enumerate(MEDIA):
        a, b = bands[name]
        ax.fill_between(freq, a * 1e12, b * 1e12, alpha=0.55, lw=0,
                        color=fs.PALETTE[i], label=name)
    ax.axhline(C_dev.mid * 1e12, color="k", ls=":", lw=1.2)
    ax.annotate("geometric $C$", xy=(3e9, C_dev.mid * 1e12),
                xytext=(0, 8), textcoords="offset points", fontsize=7.5,
                ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("measured $C$  (pF)")
    fs.legend(ax, loc="lower left")

    ax = axes[1]
    y = np.arange(len(MEDIA))
    for i, r in enumerate(rows):
        ax.barh(i, r["f_diel"] - r["f_edl"][1],
                left=r["f_edl"][1], height=0.55,
                color=fs.PALETTE[i], alpha=0.65)
        ax.plot([r["f_edl"][0], r["f_edl"][1]], [i, i], color="k", lw=2.6)
        ax.plot(r["f_diel"], i, "v", color="k", ms=5)
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows])
    ax.set_xscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_title("conductance window\n(bar), $f_{\\rm edl}$ and "
                 "$f_{\\rm diel}$", fontsize=8.5)

    ax = axes[2]
    rel = np.array([d["rel"] for d in drift]) * 100
    hw = np.array([d["halfwidth_pc"] for d in drift])
    ax.plot(rel, hw, "o-", color=fs.PALETTE[0],
            label="certified $G$")
    ax.axhline(kG_dev.relwidth, color=fs.C_LO, ls="--", lw=1.4,
               label="numerical only")
    ax.set_xlabel("uncertainty on $\\sigma$  (%)")
    ax.set_ylabel("certified half-width on $G$  (%)")
    ax.set_yscale("log")
    fs.legend(ax, loc="lower right")

    fs.save(fig, "figures/fig_edl.png")

    with open("edl_data.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote edl_data.json")


if __name__ == "__main__":
    main()
