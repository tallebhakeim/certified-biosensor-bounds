"""
study_homogenisation.py -- the MODEL error of the concentration read-out.

Answers Reviewer 1 comment 4 and Reviewer 2 comments 1 and 5.  Both ask the same
question: the bracket certifies the field solve, but the quantitative read-out
goes through Maxwell-Garnett (MG), and if that homogenisation is off by 20 % the
certified numerical interval is worthless.

The answer is that the homogenisation error can itself be bounded, so the
concentration interval can be made guaranteed rather than model-dependent.
Three statements are available, in decreasing order of generality:

  (i)   Wiener (arithmetic / harmonic) bounds.  Valid for ANY microstructure,
        isotropic or not, at a given volume fraction.  Weakest and safest.
  (ii)  Hashin-Shtrikman (HS) bounds.  Valid for any STATISTICALLY ISOTROPIC
        two-phase medium at a given volume fraction.  Much tighter.
  (iii) Maxwell-Garnett.  A single value, valid for one specific morphology.

The essential -- and, for the paper, new -- point is (iii) versus (ii):

        MG does not lie inside the HS interval, it IS one of its two edges.

MG is the coated-sphere assemblage that attains the HS bound, so using MG as a
point estimate is not a neutral approximation: it silently selects the extreme
admissible morphology.  For insulating particles in a buffer it is the OPTIMISTIC
edge (highest eps_eff, hence the smallest apparent particle load).  A real
suspension can sit anywhere down to the other edge, and the distance between the
two edges is the model error the reviewers are asking about.

Sections
--------
  1. eps_eff(f): where MG sits relative to HS and Wiener.
  2. A resolved 3-D microstructure test: the certified bracket of a cube holding
     explicit spheres, against the effective-medium prediction.  This checks the
     bound instead of assuming it.
  3. The certified capacitance of the reference chip with the suspension, the
     permittivity ranging over the HS interval.
  4. The inversion: what a measurement of given resolution certifies about the
     concentration, with and without the model error included.

Outputs: homogenisation_data.json and figures/fig_homogenisation.png
"""

import json
import numpy as np

import device as dv
import figstyle as fs
from certified_core import (capacitance_bracket, hashin_shtrikman,
                            maxwell_garnett, wiener_bounds, Bracket, EPS0)
from geometry import suspension_cube_3d

EPS_H = dv.EPS_PBS          # buffer
EPS_P = dv.EPS_CELL         # membrane-insulated cell at 1 MHz
H_DEV = 100e-9              # mesh for the device sweeps
H_CUBE = 250e-9             # mesh for the resolved 3-D microstructure

# instrument resolution: relative resolution of the capacitance read-out.
# 1e-3 is comfortably above the < 1 aF floor of a dedicated CMOS front-end on a
# ~34 pF device, so it is a conservative, achievable figure.
RESOLUTION = 1e-3


# ---------------------------------------------------------------------------
# 1. the three mixing statements
# ---------------------------------------------------------------------------
def mixing_curves(f):
    hs_lo, hs_hi = np.array([hashin_shtrikman(EPS_P, EPS_H, x) for x in f]).T
    w_lo, w_hi = np.array([wiener_bounds(EPS_P, EPS_H, x) for x in f]).T
    mg = np.array([maxwell_garnett(EPS_P, EPS_H, x) for x in f])
    return hs_lo, hs_hi, w_lo, w_hi, mg


# ---------------------------------------------------------------------------
# 2. resolved microstructure versus effective medium
# ---------------------------------------------------------------------------
def microstructure_test(counts=(0, 4, 8, 14, 21), seeds=(1, 2, 3)):
    """Certified bracket of a cube of explicit spheres against the bounds.

    The blank cube is an exact parallel-plate capacitor, so the effective
    permittivity of the realisation can be read straight off the bracket:
        C = eps0 * eps_eff * A / d   =>   eps_eff in [C_lo, C_up] / (eps0 A / d).
    That measured interval is then compared with Wiener and HS at the volume
    fraction actually voxelised.
    """
    out = []
    side = 10e-6
    geom = EPS0 * (side * side) / side          # A/d factor for the cube
    for n_part in counts:
        for seed in seeds:
            cell, f = suspension_cube_3d(H_CUBE, side=side, radius=1.5e-6,
                                         n_part=n_part, seed=seed,
                                         eps_h=EPS_H, eps_p=EPS_P)
            br = capacitance_bracket(*cell.as_args())
            eff = Bracket(br.lo / geom, br.hi / geom)
            w = wiener_bounds(EPS_P, EPS_H, f)
            hs = hashin_shtrikman(EPS_P, EPS_H, f)
            mg = maxwell_garnett(EPS_P, EPS_H, f)
            out.append(dict(n=n_part, seed=seed, f=f,
                            eff=[eff.lo, eff.hi], wiener=list(w),
                            hs=list(hs), mg=mg,
                            in_wiener=bool(eff.lo >= w[0] - 1e-9 and
                                           eff.hi <= w[1] + 1e-9),
                            in_hs=bool(eff.lo >= hs[0] - 1e-9 and
                                       eff.hi <= hs[1] + 1e-9)))
            print(f"  n={n_part:3d} seed={seed}  f={f:6.4f}  "
                  f"eps_eff in [{eff.lo:7.3f},{eff.hi:7.3f}]  "
                  f"HS=[{hs[0]:7.3f},{hs[1]:7.3f}]  MG={mg:7.3f}  "
                  f"Wiener {'ok' if out[-1]['in_wiener'] else 'OUT'}  "
                  f"HS {'ok' if out[-1]['in_hs'] else 'OUT'}")
    return out


# ---------------------------------------------------------------------------
# 3. certified capacitance of the chip versus volume fraction
# ---------------------------------------------------------------------------
def device_curves(f):
    """Certified enclosure of the chip capacitance for each volume fraction.

    The suspension fills the channel and is represented by an effective
    permittivity.  Because the capacitance is monotone in the permittivity
    pointwise, letting eps_eff run over an interval gives a guaranteed enclosure:
        eps_eff in [a, b]  =>  C in [C_lo(a), C_up(b)].
    Two enclosures are built, one from the HS interval (model error included) and
    one from MG alone (model error ignored, i.e. what the first submission did).
    """
    hs_lo, hs_hi, _, _, mg = mixing_curves(f)
    C_hs_lo, C_hs_hi, C_mg_lo, C_mg_hi = [], [], [], []
    for a, b, m in zip(hs_lo, hs_hi, mg):
        C_hs_lo.append(dv.cell_capacitance(H_DEV, eps_fluid=a).lo)
        C_hs_hi.append(dv.cell_capacitance(H_DEV, eps_fluid=b).hi)
        br = dv.cell_capacitance(H_DEV, eps_fluid=m)
        C_mg_lo.append(br.lo)
        C_mg_hi.append(br.hi)
    return (np.array(C_hs_lo), np.array(C_hs_hi),
            np.array(C_mg_lo), np.array(C_mg_hi))


# ---------------------------------------------------------------------------
# 4. the certified inversion
# ---------------------------------------------------------------------------
def _C_of_f(f, which, model):
    """One end of the certified capacitance enclosure at volume fraction f.

    model = "mg"  : the coated-sphere morphology is ASSUMED, so eps_eff is the
                    single MG value and only the numerical error remains.
    model = "hs"  : no morphology is assumed beyond isotropy, so eps_eff ranges
                    over the whole Hashin-Shtrikman interval and the model error
                    is included.
    Both ends are decreasing in f (insulating particles), which is what lets the
    inversion below be a bisection.
    """
    hs_lo, hs_hi = hashin_shtrikman(EPS_P, EPS_H, float(f))
    if model == "mg":
        e = maxwell_garnett(EPS_P, EPS_H, float(f))
        br = dv.cell_capacitance(H_DEV, eps_fluid=e)
        return br.lo if which == "lo" else br.hi
    if which == "lo":
        return dv.cell_capacitance(H_DEV, eps_fluid=hs_lo).lo
    return dv.cell_capacitance(H_DEV, eps_fluid=hs_hi).hi


def invert(f_true, model, resolution=RESOLUTION, f_max=0.6):
    """Volume fractions compatible with a measurement taken at f_true.

    The instrument returns a value known only to lie in [C(1-r), C(1+r)]; no
    distribution is assumed on that error, so this is not a confidence interval.
    A volume fraction is CERTIFIED COMPATIBLE when its own enclosure overlaps the
    measurement interval, and the recovered quantity is the set of all compatible
    f.  Because both ends of the enclosure decrease monotonically with f, that
    set is an interval [f_a, f_b] whose ends solve

        C_up(f_b) = m_lo        and        C_lo(f_a) = m_hi ,

    found here by bisection on the solver itself.  Doing it on a grid instead
    would report the grid step rather than the true width, which is how the
    numerical-only column came out as +/-0 % in the first version of this study.

    The measurement is generated from the MG morphology, i.e. the most
    favourable case, so the comparison below is not rigged against MG.
    """
    c_mid = 0.5 * (_C_of_f(f_true, "lo", "mg") + _C_of_f(f_true, "hi", "mg"))
    m_lo, m_hi = c_mid * (1 - resolution), c_mid * (1 + resolution)

    def bisect(g, target, a, b):
        """Smallest bracket for g(f) = target, g decreasing; None if no root."""
        ga, gb = g(a) - target, g(b) - target
        if ga * gb > 0:
            return a if abs(ga) < abs(gb) else b
        for _ in range(40):
            m = 0.5 * (a + b)
            if (g(m) - target) * ga > 0:
                a = m
            else:
                b = m
            if b - a < 1e-5:
                break
        return 0.5 * (a + b)

    f_b = bisect(lambda x: _C_of_f(x, "hi", model), m_lo, 0.0, f_max)
    f_a = bisect(lambda x: _C_of_f(x, "lo", model), m_hi, 0.0, f_max)
    return min(f_a, f_b), max(f_a, f_b)


def main():
    fs.use_paper_style()

    out = {"eps_h": EPS_H, "eps_p": EPS_P, "resolution": RESOLUTION,
           "mesh_device_m": H_DEV, "mesh_cube_m": H_CUBE}

    # ---------------------------------------------------------------- 1
    f = np.linspace(0.0, 0.45, 46)
    hs_lo, hs_hi, w_lo, w_hi, mg = mixing_curves(f)
    gap = np.where(hs_hi > 0, 100 * (hs_hi - hs_lo) / hs_hi, 0.0)
    print("effective permittivity: MG versus the bounds")
    for x in (0.05, 0.10, 0.20, 0.30, 0.40):
        i = int(np.argmin(np.abs(f - x)))
        print(f"  f={f[i]:4.2f}  Wiener=[{w_lo[i]:6.2f},{w_hi[i]:6.2f}]  "
              f"HS=[{hs_lo[i]:6.2f},{hs_hi[i]:6.2f}]  MG={mg[i]:6.2f}  "
              f"HS width={gap[i]:5.1f} % of HS_up")
    on_edge = float(np.max(np.abs(mg - hs_hi)))
    print(f"  max |MG - HS_upper| over the sweep = {on_edge:.3e}  "
          f"(MG lies ON the upper HS edge)")
    out["mg_on_hs_edge_max_dev"] = on_edge
    out["f_grid"] = f.tolist()
    out["hs_lo"], out["hs_hi"] = hs_lo.tolist(), hs_hi.tolist()
    out["wiener_lo"], out["wiener_hi"] = w_lo.tolist(), w_hi.tolist()
    out["mg"] = mg.tolist()

    # ---------------------------------------------------------------- 2
    print("\nresolved 3-D microstructure versus the effective-medium bounds")
    micro = microstructure_test()
    out["microstructure"] = micro
    out["microstructure_all_in_wiener"] = all(m["in_wiener"] for m in micro)
    out["microstructure_all_in_hs"] = all(m["in_hs"] for m in micro)
    print(f"  all realisations inside Wiener : "
          f"{out['microstructure_all_in_wiener']}")
    print(f"  all realisations inside HS     : "
          f"{out['microstructure_all_in_hs']}")

    # ---------------------------------------------------------------- 3
    print("\ncertified chip capacitance versus volume fraction")
    fd = np.linspace(0.0, 0.30, 31)
    hs_lo_d, hs_hi_d, _, _, mg_d = mixing_curves(fd)
    C_hs_lo, C_hs_hi, C_mg_lo, C_mg_hi = device_curves(fd)
    out["f_device"] = fd.tolist()
    out["C_hs_lo"], out["C_hs_hi"] = C_hs_lo.tolist(), C_hs_hi.tolist()
    out["C_mg_lo"], out["C_mg_hi"] = C_mg_lo.tolist(), C_mg_hi.tolist()
    for x in (0.05, 0.15, 0.30):
        i = int(np.argmin(np.abs(fd - x)))
        print(f"  f={fd[i]:4.2f}  C(MG only)=[{C_mg_lo[i]*1e12:8.3f},"
              f"{C_mg_hi[i]*1e12:8.3f}] pF/m  "
              f"C(model error)=[{C_hs_lo[i]*1e12:8.3f},"
              f"{C_hs_hi[i]*1e12:8.3f}] pF/m")

    # ---------------------------------------------------------------- 4
    print(f"\ncertified concentration interval, resolution {RESOLUTION*100:.1f} %")
    rows = []
    for f_true in (0.02, 0.05, 0.10, 0.20, 0.30):
        a = invert(f_true, "mg")
        b = invert(f_true, "hs")
        wa = 100 * (a[1] - a[0]) / (2 * f_true)
        wb = 100 * (b[1] - b[0]) / (2 * f_true)
        rows.append(dict(f_true=f_true, mg=list(a), hs=list(b),
                         halfwidth_mg_pc=wa, halfwidth_hs_pc=wb))
        print(f"  f={f_true:4.2f}  numerical only: +/-{wa:6.2f} %  "
              f"[{a[0]:.4f},{a[1]:.4f}]   with model error: +/-{wb:6.2f} %  "
              f"[{b[0]:.4f},{b[1]:.4f}]")
    out["inversion"] = rows

    # ---------------------------------------------------------------- figure
    import matplotlib.pyplot as plt
    fig, axes = fs.row(3, height=3.05)

    ax = axes[0]
    ax.fill_between(f, w_lo, w_hi, color="0.82", lw=0,
                    label="Wiener (any morphology)")
    ax.fill_between(f, hs_lo, hs_hi, color=fs.C_BAND, alpha=0.75, lw=0,
                    label="Hashin-Shtrikman (isotropic)")
    ax.plot(f, mg, color=fs.C_LO, lw=2.0, ls="--",
            label="Maxwell-Garnett")
    ax.set_xlabel("particle volume fraction $f$")
    ax.set_ylabel("effective permittivity $\\varepsilon_{\\rm eff}$")
    fs.legend(ax, loc="lower left")
    ax.annotate("MG lies ON the upper edge:\nit is an extremal morphology",
                xy=(0.30, maxwell_garnett(EPS_P, EPS_H, 0.30)),
                xytext=(0.06, 0.60), textcoords="axes fraction", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

    ax = axes[1]
    fm = np.array([m["f"] for m in micro])
    lo = np.array([m["eff"][0] for m in micro])
    hi = np.array([m["eff"][1] for m in micro])
    ax.fill_between(f, hs_lo, hs_hi, color=fs.C_BAND, alpha=0.55, lw=0,
                    label="Hashin-Shtrikman")
    ax.errorbar(fm, 0.5 * (lo + hi), yerr=0.5 * (hi - lo), fmt="o",
                ms=4, lw=1.2, color=fs.C_MC, capsize=2.5,
                label="resolved 3-D spheres\n(certified bracket)")
    ax.set_xlabel("voxelised volume fraction $f$")
    ax.set_ylabel("$\\varepsilon_{\\rm eff}$ from the bracket")
    ax.set_xlim(-0.01, max(0.32, fm.max() * 1.08))
    fs.legend(ax, loc="lower left")
    # The realisations sit ON the upper edge, which is the physical content of
    # the panel: random dispersed spheres ARE close to the coated-sphere
    # morphology, so MG describes them well -- but only because the morphology
    # happens to be that one, and the electrical measurement cannot confirm it.
    ax.annotate("realisations hug the UPPER edge:\ndispersed spheres are the "
                "MG\nmorphology, not a generic one",
                xy=(fm.max(), 0.5 * (lo[-1] + hi[-1])),
                xytext=(0.30, 0.90), textcoords="axes fraction", fontsize=7.5,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

    ax = axes[2]
    ftrue = np.array([r["f_true"] for r in rows])
    wmg = np.array([r["halfwidth_mg_pc"] for r in rows])
    whs = np.array([r["halfwidth_hs_pc"] for r in rows])
    ax.plot(ftrue, wmg, "o-", color=fs.PALETTE[0], label="numerical error only")
    ax.plot(ftrue, whs, "s-", color=fs.PALETTE[1],
            label="numerical + model error")
    ax.set_yscale("log")
    ax.set_xlabel("true volume fraction $f$")
    ax.set_ylabel("certified half-width on $f$  (%)")
    fs.legend(ax, loc="center left", headroom=0.0)
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)

    fs.save(fig, "figures/fig_homogenisation.png")

    with open("homogenisation_data.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote homogenisation_data.json")


if __name__ == "__main__":
    main()
