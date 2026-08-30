"""
study_detection.py -- certified detection and the uncertainty budget.

This replaces the detection and uncertainty-quantification sections of the first
submission, recast on the observable the revision actually claims.

Two read-outs, one certificate
------------------------------
The cell capacitance and the cell conductance are the same elliptic problem with
a different coefficient,

    C = eps0 eps_r k        G = sigma k ,

with the SAME geometric factor k.  The solver is linear in the coefficient field,
so passing a conductivity map where the permittivity map is expected and dividing
the result by eps0 returns the conductance exactly.  One bracket therefore
certifies both read-outs, each in its own frequency window (see study_edl.py).

Why this matters for detection.  An intact cell membrane blocks the conduction
current almost completely (sigma_cell about 0.01 S/m against 1.6 S/m for PBS, a
ratio of 160) while it only lowers the permittivity from 78 to 6 (a ratio of 13).
The dielectric contrast the first submission relied on is therefore an order of
magnitude WEAKER than the conductive contrast available in the same device.
Detection is easier in the conductance channel, which is also the channel that is
measurable in a physiological buffer.

Sections
--------
  1. Certified response to the captured cell row density, both channels.
     (A single isolated cell is a 3-D problem: see study_single_cell.py.)
  2. Certified uncertainty: bounded data (permittivity, conductivity) plus the
     instrument resolution, and the certified limit of detection that follows.
  3. The uncertainty budget, ordered by magnitude.  This is the table that makes
     Reviewer 2's point: the numerical term is the smallest one.

Outputs: detection_data.json and figures/fig_detection.png
"""

import json
import numpy as np

import device as dv
import figstyle as fs
from certified_core import Bracket, EPS0

# ---- materials, both channels ---------------------------------------------
EPS_PBS = dv.EPS_PBS            # 78
SIG_PBS = 1.6                   # S/m
EPS_CELL = dv.EPS_CELL          # 6, membrane insulated
SIG_CELL = 0.01                 # S/m, membrane blocks the conduction current

R_CELL = 3.0e-6                 # 6 um diameter cell
H = dv.H_SWEEP                  # 250 nm mesh for the sweeps
H_FINE = dv.H_FINE              # 100 nm for the headline numbers

# ---- bounded data ---------------------------------------------------------
# Permittivity of the buffer drifts with temperature and ionic strength; the
# conductivity drifts more, and in the opposite direction (about +2 %/K).
REL_EPS = 0.04
REL_SIG = 0.02
RESOLUTION = 1e-3               # relative resolution of the read-out


def particle_row(n, y=None):
    """n cell ROWS lying in the sensing window, spread across the quarter cell.

    The quarter cell runs from the finger centre to the gap centre, so its width
    is lam/2 = 20 um.  Cells are placed on a regular row just above the floor,
    which is where a capture assay puts them and where the field is strongest.

    CAREFUL about what n means.  This is a two-dimensional cross-section, so each
    disc is a cylinder running the whole 2 mm finger: n counts ROWS of cells
    along the finger, not individual cells.  One row is about 333 cells at a 6 um
    diameter.  The sweep below therefore measures the response to a LINE DENSITY
    of captured cells, which is the right quantity for a capture assay covering
    the electrode, and it is NOT a single-particle sensitivity.  For one isolated
    cell the calculation has to be three dimensional; see study_single_cell.py,
    which finds 400 aF, four and a half orders of magnitude below what a 2-D disc
    suggests.
    """
    if n == 0:
        return ()
    if y is None:
        y = R_CELL * 1.2
    x0, x1 = R_CELL * 1.1, dv.LAM / 2 - R_CELL * 1.1
    xs = np.linspace(x0, x1, n) if n > 1 else [0.5 * (x0 + x1)]
    return tuple((float(x), float(y), R_CELL, None) for x in xs)


def bracket(n, channel, eps_b=None, sig_b=None, h=H):
    """Certified enclosure of the read-out with n cells in the window.

    channel = "C" returns a capacitance in F per metre of finger.
    channel = "G" returns a conductance in S per metre of finger: the same solve
    with the conductivity map, divided by eps0.
    """
    if channel == "C":
        fluid = EPS_PBS if eps_b is None else eps_b
        part = EPS_CELL
        sub = dv.EPS_GLASS                 # 4.2, borosilicate as a dielectric
        scale = 1.0
    else:
        fluid = SIG_PBS if sig_b is None else sig_b
        part = SIG_CELL
        # The substrate coefficient MUST change with the problem: glass is a
        # middling dielectric but an insulator.  Leaving it at 4.2 would make the
        # substrate more conductive than the buffer and shunt the whole cell.
        sub = dv.SIGMA_GLASS
        scale = 1.0 / EPS0
    parts = tuple((x, y, r, part) for (x, y, r, _) in particle_row(n))
    br = dv.cell_capacitance(h, eps_fluid=fluid, particles=parts, eps_sub=sub)
    return Bracket(br.lo * scale, br.hi * scale)


def envelope(n, channel, rel, h=H):
    """Guaranteed envelope over the bounded-data box, with no sampling.

    The energy is monotone in every coefficient, so its range over a box of
    coefficients is attained at the corners.  Here the box is one-dimensional
    (the buffer coefficient), so the envelope is the pair of corner values.
    """
    if channel == "C":
        lo = bracket(n, "C", eps_b=EPS_PBS * (1 - rel), h=h).lo
        hi = bracket(n, "C", eps_b=EPS_PBS * (1 + rel), h=h).hi
    else:
        lo = bracket(n, "G", sig_b=SIG_PBS * (1 - rel), h=h).lo
        hi = bracket(n, "G", sig_b=SIG_PBS * (1 + rel), h=h).hi
    return Bracket(lo, hi)


def certified_lod(channel, rel=0.0, resolution=0.0, n_max=8, h=H):
    """Smallest number of cells whose enclosure clears the blank envelope.

    The blank envelope is widened by the bounded data (rel) and by the instrument
    resolution, and the sample enclosure is widened the same way; detection is
    certified only when the two are disjoint.  Returning None means the device
    cannot certify detection of any load up to n_max under those hypotheses.
    """
    b = envelope(0, channel, rel, h) if rel > 0 else bracket(0, channel, h=h)
    b = Bracket(b.lo * (1 - resolution), b.hi * (1 + resolution))
    for n in range(1, n_max + 1):
        s = envelope(n, channel, rel, h) if rel > 0 else bracket(n, channel, h=h)
        s = Bracket(s.lo * (1 - resolution), s.hi * (1 + resolution))
        if s.clears(b):
            return n
    return None


def main():
    fs.use_paper_style()
    out = {"r_cell_m": R_CELL, "resolution": RESOLUTION,
           "rel_eps": REL_EPS, "rel_sig": REL_SIG}

    # ---------------------------------------------------------------- 1
    print("certified response to a captured cell ROW density, both channels")
    counts = list(range(0, 7))
    data = {}
    for ch, unit, sc in (("C", "pF/m", 1e12), ("G", "mS/m", 1e3)):
        rows = []
        for n in counts:
            br = bracket(n, ch, h=H_FINE)
            rows.append(br)
        b0 = rows[0]
        rel = [(100 * (r.mid / b0.mid - 1.0)) for r in rows]
        data[ch] = dict(brackets=[[r.lo, r.hi] for r in rows], rel=rel)
        print(f"  channel {ch}: blank = [{b0.lo*sc:.4f}, {b0.hi*sc:.4f}] {unit}"
              f"  (+/-{b0.relwidth:.4f} %)")
        for n, r, x in zip(counts, rows, rel):
            sep = "certified" if not r.clears(b0) is False else ""
            print(f"    n={n}  [{r.lo*sc:10.4f},{r.hi*sc:10.4f}] {unit}  "
                  f"{x:+7.3f} %  {'DISJOINT from blank' if r.clears(b0) else 'overlaps blank'}")
    out["counts"] = counts
    out["channels"] = data

    # contrast of the two channels, which is the point of the reframing
    cC = abs(data["C"]["rel"][1])
    cG = abs(data["G"]["rel"][1])
    print(f"  one cell ROW moves the capacitance by {cC:.3f} % and the "
          f"conductance by {cG:.3f} %  (ratio {cG/max(cC,1e-12):.2f})")
    out["one_cell_pc_C"], out["one_cell_pc_G"] = cC, cG

    # ---------------------------------------------------------------- 2
    print("\ncertified limit of detection under increasing hypotheses")
    lod_rows = []
    for label, ch, rel, res in [
            ("C, numerical only", "C", 0.0, 0.0),
            ("C, + bounded permittivity", "C", REL_EPS, 0.0),
            ("C, + instrument resolution", "C", REL_EPS, RESOLUTION),
            ("G, numerical only", "G", 0.0, 0.0),
            ("G, + bounded conductivity", "G", REL_SIG, 0.0),
            ("G, + instrument resolution", "G", REL_SIG, RESOLUTION)]:
        n = certified_lod(ch, rel, res, n_max=8, h=H)
        lod_rows.append(dict(label=label, channel=ch, rel=rel,
                             resolution=res, lod=n))
        print(f"  {label:32s} LOD = "
              f"{n if n else '> 8 rows (not certified)'}")
    out["lod"] = lod_rows

    # ---------------------------------------------------------------- 3
    # the budget, ordered by magnitude.  Values come from the other studies so
    # that the paper quotes one consistent set of numbers.
    print("\nuncertainty budget of the concentration read-out, by magnitude")
    budget = [
        ("numerical discretisation", 0.020, "this solver, h = 50 nm"),
        ("ionic strength, sigma to 1 %", 1.05, "study_edl.py"),
        ("fabrication, 100 nm tolerance", 1.383, "study_fabrication.py"),
        ("Hele-Shaw sidewall bias", 6.7, "study_validity.py, one sided"),
        ("homogenisation model error", 35.2, "study_homogenisation.py, f = 0.10"),
    ]
    for name, pc, src in budget:
        print(f"  {name:34s} {pc:8.3f} %   ({src})")
    out["budget"] = [dict(name=n, pc=p, source=s) for n, p, s in budget]
    print(f"  ratio largest / smallest = "
          f"{budget[-1][1]/budget[0][1]:.0f}")
    out["budget_ratio"] = budget[-1][1] / budget[0][1]

    # ---------------------------------------------------------------- figure
    import matplotlib.pyplot as plt
    fig, axes = fs.row(3, height=3.05)

    ax = axes[0]
    for ch, col, lab in (("C", fs.PALETTE[0], "capacitance $C$"),
                         ("G", fs.PALETTE[1], "conductance $G$")):
        r = np.array(data[ch]["rel"])
        lo = np.array([b[0] for b in data[ch]["brackets"]])
        hi = np.array([b[1] for b in data[ch]["brackets"]])
        mid0 = 0.5 * (lo[0] + hi[0])
        ax.fill_between(counts, 100 * (lo / mid0 - 1), 100 * (hi / mid0 - 1),
                        color=col, alpha=0.55, lw=0, label=lab)
    ax.axhline(0.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("captured cell rows across the cell\n(1 row $\\approx$ 333 cells along a 2 mm finger)")
    ax.set_ylabel("certified relative change  (%)")
    fs.legend(ax, loc="lower left")

    ax = axes[1]
    labels = [r["label"].replace(", ", "\n") for r in lod_rows]
    vals = [r["lod"] if r["lod"] else 9 for r in lod_rows]
    cols = [fs.PALETTE[0] if r["channel"] == "C" else fs.PALETTE[1]
            for r in lod_rows]
    ax.barh(range(len(vals)), vals, color=cols, alpha=0.8)
    for i, r in enumerate(lod_rows):
        if r["lod"] is None:
            ax.annotate("not certified", xy=(9, i), xytext=(-4, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=7.5, color="white")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("certified LOD  (cell rows)")

    ax = axes[2]
    names = [b[0].replace(", ", "\n") for b in budget]
    pcs = [b[1] for b in budget]
    ax.barh(range(len(pcs)), pcs, color=fs.C_BAND, edgecolor=fs.C_LO, lw=0.8)
    ax.set_yticks(range(len(pcs)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("contribution to the read-out  (%)")
    ax.annotate("the certified numerical term\nis the SMALLEST contribution",
                xy=(pcs[0], 0), xytext=(0.35, 0.30),
                textcoords="axes fraction", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

    fs.save(fig, "figures/fig_detection.png")

    with open("detection_data.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote detection_data.json")


if __name__ == "__main__":
    main()
