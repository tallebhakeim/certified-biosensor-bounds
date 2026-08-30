"""
study_single_cell.py -- what ONE cell really does, in three dimensions.

This corrects a claim of the first submission, and the correction is substantial
enough that it belongs in the response letter.

The first submission reported that "a single bioparticle already lowers the
capacitance by several percent, so a single particle is certified detected".  That
number came from a two-dimensional cross-section, where a particle is not a
particle: it is an infinite cylinder running the whole 2 mm finger length.  At a
6 um cell diameter that single 2-D disc stands for about 333 cells in a row, so
the reported single-particle sensitivity was too large by roughly that factor.

What this script does
---------------------
  1. Computes the capacitance change caused by ONE sphere in a genuine 3-D slice
     of the comb, extruded over a length lz along the finger with symmetry faces.
  2. Checks that the change CONVERGES as lz grows.  It must: the perturbation is
     local, so beyond a few particle diameters the extra fluid adds capacitance
     to both the blank and the loaded cell equally and dC saturates.  A converged
     dC is the absolute change caused by one cell anywhere on the chip, and can
     be divided by the whole-chip capacitance directly.
  3. Compares with the 2-D result to quantify the artefact explicitly.
  4. States what the device can and cannot certify: one cell against the
     numerical bracket alone, and one cell against the real uncertainty budget.

Outputs: single_cell_data.json and figures/fig_single_cell.png
"""

import json
import time
import numpy as np

import device as dv
import figstyle as fs
from certified_core import capacitance_bracket, Bracket, EPS0
from geometry import ide_slice_3d

R_CELL = 3.0e-6
EPS_CELL = dv.EPS_CELL
SIG_CELL = 0.01
SIG_PBS = 1.6

H3 = 250e-9                       # 3-D mesh
LZ = [10e-6, 20e-6, 40e-6]        # extrusion lengths to test convergence


def slice_bracket(lz, h=H3, spheres=(), eps_fluid=dv.EPS_PBS, scale=1.0):
    """Certified enclosure of the 3-D slice, in absolute farads (or siemens)."""
    c = ide_slice_3d(h, lam=dv.LAM, eta=dv.ETA, t_metal=dv.T_METAL,
                     h_sub=dv.H_SUB, h_fluid=dv.H_CHANNEL, lz=lz,
                     eps_fluid=eps_fluid, spheres=spheres)
    br = capacitance_bracket(*c.as_args())
    # the quarter cell carries half a finger at half the applied voltage, so the
    # cell value is a quarter of the quarter-cell value (see device.py)
    return Bracket(br.lo * scale / 2, br.hi * scale / 2)


def one_sphere(lz):
    """One cell sitting on the floor, mid-way across the gap, centred in z."""
    return ((dv.LAM / 4, R_CELL * 1.05, lz / 2, R_CELL, EPS_CELL),)


def main():
    fs.use_paper_style()
    out = {"r_cell_m": R_CELL, "mesh_m": H3}

    # ---------------------------------------------------------------- 1, 2
    print("one cell in a 3-D slice: does dC converge with the slice length?")
    rows = []
    for lz in LZ:
        t0 = time.time()
        blank = slice_bracket(lz)
        sph = ((dv.LAM / 4, R_CELL * 1.05, lz / 2, R_CELL, EPS_CELL),)
        load = slice_bracket(lz, spheres=sph)
        dt = time.time() - t0
        # dC is negative; bound it two-sided from the two enclosures
        dC_lo = load.lo - blank.hi
        dC_hi = load.hi - blank.lo
        rows.append(dict(lz=lz, blank=[blank.lo, blank.hi],
                         load=[load.lo, load.hi],
                         dC=[dC_lo, dC_hi]))
        print(f"  lz={lz*1e6:5.1f} um  blank=[{blank.lo*1e15:9.4f},"
              f"{blank.hi*1e15:9.4f}] fF  dC in [{dC_lo*1e18:9.3f},"
              f"{dC_hi*1e18:9.3f}] aF   ({dt:.1f} s)")
    out["convergence"] = rows

    dC_mid = [0.5 * (r["dC"][0] + r["dC"][1]) for r in rows]
    spread = (max(dC_mid) - min(dC_mid)) / abs(np.mean(dC_mid))
    converged = abs(dC_mid[-1] - dC_mid[-2]) / abs(dC_mid[-1])
    print(f"  spread over the three slice lengths : {100*spread:.1f} %")
    print(f"  change between the last two         : {100*converged:.1f} %"
          f"   -> {'converged' if converged < 0.10 else 'NOT converged'}")
    out["spread_pc"] = 100 * spread
    out["last_step_pc"] = 100 * converged
    out["converged"] = bool(converged < 0.10)

    # ---------------------------------------------------------------- 3
    dC = rows[-1]["dC"]
    C_chip = dv.device_capacitance(dv.cell_capacitance(dv.H_FINE))
    rel_lo = 100 * dC[0] / C_chip.hi
    rel_hi = 100 * dC[1] / C_chip.lo
    print(f"\none cell on the whole chip")
    print(f"  chip capacitance      : [{C_chip.lo*1e12:.4f}, "
          f"{C_chip.hi*1e12:.4f}] pF")
    print(f"  dC of one cell        : [{dC[0]*1e18:.3f}, {dC[1]*1e18:.3f}] aF")
    print(f"  relative change       : [{rel_lo:.3e}, {rel_hi:.3e}] %")
    out["chip_C"] = [C_chip.lo, C_chip.hi]
    out["one_cell_dC_F"] = dC
    out["one_cell_rel_pc"] = [rel_lo, rel_hi]

    # the 2-D artefact, quantified
    b2 = dv.cell_capacitance(dv.H_FINE)
    p2 = ((dv.LAM / 4, R_CELL * 1.05, R_CELL, EPS_CELL),)
    l2 = dv.cell_capacitance(dv.H_FINE, particles=p2)
    rel2 = 100 * (l2.mid / b2.mid - 1.0)
    n_equiv = dv.FINGER_LENGTH / (2 * R_CELL)
    print(f"\n  the same particle in 2-D          : {rel2:+.3f} % per period")
    print(f"  a 2-D disc stands for               : {n_equiv:.0f} cells in a row")
    print(f"  overstatement factor of the 2-D claim: "
          f"{abs(rel2) / abs(0.5*(rel_lo+rel_hi)):.0f}x")
    out["two_d_rel_pc"] = rel2
    out["n_equivalent_cells"] = n_equiv
    out["overstatement"] = abs(rel2) / abs(0.5 * (rel_lo + rel_hi))

    # ---------------------------------------------------------------- 4
    # can one cell be certified detected?  Against the numerical bracket alone,
    # and against the budget the revision reports.
    num_halfwidth_pc = 0.5 * (C_chip.hi - C_chip.lo) / C_chip.mid * 100
    print(f"\n  numerical half-width on the chip    : {num_halfwidth_pc:.4f} %")
    print(f"  one cell moves it by                : "
          f"{abs(0.5*(rel_lo+rel_hi)):.3e} %")
    detect_num = abs(0.5 * (rel_lo + rel_hi)) > 2 * num_halfwidth_pc
    print(f"  single cell above the numerical bracket ? "
          f"{'YES' if detect_num else 'NO'}")
    # how many cells to clear the numerical bracket, and the real budget
    per_cell = abs(0.5 * (rel_lo + rel_hi))
    n_num = int(np.ceil(2 * num_halfwidth_pc / per_cell))
    n_fab = int(np.ceil(2 * 1.383 / per_cell))
    n_res = int(np.ceil(2 * 0.1 / per_cell))
    print(f"  cells needed to clear the numerical bracket : {n_num}")
    print(f"  cells needed to clear a 0.1 % read-out floor: {n_res}")
    print(f"  cells needed to clear the 100 nm fabrication envelope: {n_fab}")
    out["single_cell_above_numerical"] = bool(detect_num)
    out["n_cells_numerical"] = n_num
    out["n_cells_resolution"] = n_res
    out["n_cells_fabrication"] = n_fab
    # a differential measurement cancels the fabrication term, so quote both
    print("  note: fabrication is common mode in a differential (same chip)")
    print("        measurement, so the read-out floor is the operative limit")

    # ---------------------------------------------------------------- 5
    # The negative result above is about the BASELINE, not the signal.  One cell
    # moves the capacitance by about 400 aF, which is some 400 times the
    # sub-attofarad absolute floor reported for dedicated CMOS cytometry
    # front-ends (Chien et al., Lab Chip 18:2065, 2018).  What defeats it is the
    # 33.7 pF of parasitic baseline from 50 periods of 2 mm finger.  So the
    # framework can be run backwards: given a relative read-out floor, it returns
    # the largest sensing electrode that still certifies one cell.
    print("\ndesign implication: the baseline budget for one cell")
    ABS_FLOOR = 1e-18                      # 1 aF, measured absolute floor
    dC_abs = abs(0.5 * (dC[0] + dC[1]))
    print(f"  dC of one cell / absolute floor : {dC_abs/ABS_FLOOR:.0f}x "
          f"-> the SIGNAL is not the problem")
    C_per_period = dv.cell_capacitance(dv.H_FINE).mid * dv.FINGER_LENGTH
    rows5 = []
    for res in (1e-2, 1e-3, 1e-4):
        C_max = dC_abs / res               # largest baseline that still resolves
        n_per = C_max / C_per_period
        L_eq = C_max / dv.cell_capacitance(dv.H_FINE).mid
        rows5.append(dict(resolution=res, C_max=C_max, n_periods=n_per,
                          finger_length_m=L_eq))
        print(f"  relative floor {res*100:6.3f} %  ->  baseline <= "
              f"{C_max*1e15:8.3f} fF  =  {n_per:7.3f} periods of 2 mm"
              f"  =  {L_eq*1e6:8.1f} um of single-period electrode")
    out["baseline_budget"] = rows5
    out["abs_floor_margin"] = dC_abs / ABS_FLOOR
    print("  a single electrode pair a few tens of um wide is exactly the")
    print("  geometry used in impedance cytometry, so the certified argument")
    print("  RECOVERS that design choice instead of contradicting it")

    # ---------------------------------------------------------------- figure
    import matplotlib.pyplot as plt
    fig, axes = fs.row(3, height=3.05)

    ax = axes[0]
    lz_um = np.array([r["lz"] for r in rows]) * 1e6
    lo = np.array([r["dC"][0] for r in rows]) * 1e18
    hi = np.array([r["dC"][1] for r in rows]) * 1e18
    ax.errorbar(lz_um, 0.5 * (lo + hi), yerr=0.5 * (hi - lo), fmt="o-",
                color=fs.PALETTE[0], capsize=3, lw=1.4, ms=5,
                label="certified $\\Delta C$ of one cell")
    ax.set_xlabel("slice length along the finger  ($\\mu$m)")
    ax.set_ylabel("$\\Delta C$ of one cell  (aF)")
    ax.set_xlim(lz_um[0] - 6, lz_um[-1] + 14)
    ax.margins(y=0.30)
    fs.legend(ax, loc="upper left")
    # The enclosure widens with the slice because the bigger domain carries more
    # discretisation error, while the midpoint saturates: that is the point.
    ax.annotate("midpoint saturates:\nthe perturbation is local,\nso this is the "
                "absolute\nchange on the chip",
                xy=(lz_um[-1], 0.5 * (lo[-1] + hi[-1])),
                xytext=(0.32, 0.05), textcoords="axes fraction", fontsize=7.5,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

    ax = axes[1]
    names = ["2-D cross-section\n(one infinite cylinder)",
             "3-D, one cell\n(this work)"]
    vals = [abs(rel2), abs(0.5 * (rel_lo + rel_hi))]
    ax.bar(names, vals, color=[fs.C_MC, fs.PALETTE[0]], alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("relative change  (%)")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.margins(y=0.35)
    over = out["overstatement"]
    ax.annotate(f"the 2-D cross-section overstates\none cell by about "
                f"{round(over, -3):.0f} times",
                xy=(0, vals[0]), xytext=(0.03, 0.62),
                textcoords="axes fraction", fontsize=7.5, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

    ax = axes[2]
    labels = ["numerical\nbracket", "0.1 % read-out\nfloor",
              "100 nm fabrication\nenvelope"]
    need = [n_num, n_res, n_fab]
    ax.barh(range(3), need, color=fs.C_BAND, edgecolor=fs.C_LO, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(right=max(need) * 3.2)          # room for the value labels
    for i, v in enumerate(need):
        ax.annotate(f"{v}", xy=(v, i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("cells for certified detection")

    fs.save(fig, "figures/fig_single_cell.png")

    with open("single_cell_data.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote single_cell_data.json")


if __name__ == "__main__":
    main()
