"""
test_core.py -- validation of the certified bracket against analytic
references and against the classical mixing bounds.

Run:  python3 test_core.py
Every test prints PASS/FAIL; the script exits non-zero if anything fails.
"""

import numpy as np
from certified_core import (
    EPS0, DIELECTRIC, HOT, GND,
    capacitance_bracket, capacitance_upper, capacitance_lower,
    wiener_bounds, maxwell_garnett, hashin_shtrikman,
)

_fail = 0


def check(name, ok, detail=""):
    global _fail
    print(f"[{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        _fail += 1


def brackets(name, br, exact, tol_rel=1e-9):
    """The enclosure must contain the exact value."""
    ok = br.lo <= exact * (1 + tol_rel) and br.hi >= exact * (1 - tol_rel)
    check(name, ok,
          f"lo={br.lo:.8e} exact={exact:.8e} hi={br.hi:.8e} "
          f"width={br.relwidth:.3g}%")
    return ok


# --------------------------------------------------------------------------
def parallel_plate_2d(n_gap=10, n_wide=6, eps_r=78.0, h=1e-6, depth=1e-3):
    """Uniform slab: both bounds must be exact (the field is 1-D)."""
    ny, nx = n_gap + 2, n_wide
    eps = np.full((ny, nx), eps_r)
    elec = np.zeros((ny, nx), dtype=int)
    elec[0, :] = HOT
    elec[-1, :] = GND
    br = capacitance_bracket(eps, elec, (h, h), depth)
    gap = n_gap * h
    width = n_wide * h
    exact = EPS0 * eps_r * width * depth / gap
    return br, exact


def layered_2d(n1=8, n2=12, e1=78.0, e2=4.0, n_wide=5, h=1e-6, depth=1e-3):
    """Two dielectric layers in series: exact closed form."""
    ny, nx = n1 + n2 + 2, n_wide
    eps = np.empty((ny, nx))
    eps[: n1 + 1, :] = e1
    eps[n1 + 1:, :] = e2
    elec = np.zeros((ny, nx), dtype=int)
    elec[0, :] = HOT
    elec[-1, :] = GND
    br = capacitance_bracket(eps, elec, (h, h), depth)
    width = n_wide * h
    A = width * depth
    exact = 1.0 / ((n1 * h) / (EPS0 * e1 * A) + (n2 * h) / (EPS0 * e2 * A))
    return br, exact


def parallel_plate_3d(n_gap=6, n=4, eps_r=78.0, h=1e-6):
    ny, nx, nz = n_gap + 2, n, n
    eps = np.full((ny, nx, nz), eps_r)
    elec = np.zeros((ny, nx, nz), dtype=int)
    elec[0] = HOT
    elec[-1] = GND
    br = capacitance_bracket(eps, elec, (h, h, h))
    exact = EPS0 * eps_r * (n * h) * (n * h) / (n_gap * h)
    return br, exact


def layered_3d(n1=5, n2=7, e1=78.0, e2=4.0, n=4, h=1e-6):
    ny = n1 + n2 + 2
    eps = np.empty((ny, n, n))
    eps[: n1 + 1] = e1
    eps[n1 + 1:] = e2
    elec = np.zeros((ny, n, n), dtype=int)
    elec[0] = HOT
    elec[-1] = GND
    br = capacitance_bracket(eps, elec, (h, h, h))
    A = (n * h) ** 2
    exact = 1.0 / ((n1 * h) / (EPS0 * e1 * A) + (n2 * h) / (EPS0 * e2 * A))
    return br, exact


# --------------------------------------------------------------------------
def coplanar_cell(h=0.5e-6, particle=None):
    """Quarter period of a coplanar comb, geometry fixed in physical units."""
    from geometry import ide_quarter_cell
    parts = () if particle is None else (particle,)
    return ide_quarter_cell(h, lam=40e-6, eta=0.5, particles=parts,
                            depth=1e-3).as_args()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 74)
    print("Certified bracket -- validation suite")
    print("=" * 74)

    br, ex = parallel_plate_2d()
    brackets("2-D uniform slab encloses the analytic capacitance", br, ex)
    check("2-D uniform slab bracket is tight (<0.01%)", br.relwidth < 1e-2,
          f"width={br.relwidth:.3g}%")

    br, ex = layered_2d()
    brackets("2-D two-layer series capacitor enclosed", br, ex)
    check("2-D layered bracket is tight (<0.01%)", br.relwidth < 1e-2,
          f"width={br.relwidth:.3g}%")

    br, ex = parallel_plate_3d()
    brackets("3-D uniform slab enclosed", br, ex)
    check("3-D uniform slab tight (<0.01%)", br.relwidth < 1e-2,
          f"width={br.relwidth:.3g}%")

    br, ex = layered_3d()
    brackets("3-D two-layer series capacitor enclosed", br, ex)

    # ---- ordering and non-degeneracy on a genuinely 2-D field -------------
    eps, elec, hh, dep = coplanar_cell(h=0.5e-6)
    br = capacitance_bracket(eps, elec, hh, dep)
    check("coplanar comb: lower <= upper", br.lo <= br.hi, repr(br))
    check("coplanar comb: bracket non-degenerate", br.relwidth > 1e-3,
          f"width={br.relwidth:.3g}%")

    # ---- convergence: the certified width must shrink with the mesh ------
    widths, mids = [], []
    for hh_ in (1.0e-6, 0.5e-6, 0.25e-6):
        eps, elec, hh, dep = coplanar_cell(h=hh_)
        b = capacitance_bracket(eps, elec, hh, dep)
        widths.append(b.relwidth)
        mids.append(b.mid)
    ok = widths[0] > widths[1] > widths[2]
    check("coplanar comb: certified width decreases under refinement", ok,
          " -> ".join(f"{w:.3g}%" for w in widths))

    # every finer enclosure must stay compatible with the coarser one
    eps, elec, hh, dep = coplanar_cell(h=0.25e-6)
    bf = capacitance_bracket(eps, elec, hh, dep)
    eps, elec, hh, dep = coplanar_cell(h=1.0e-6)
    bc = capacitance_bracket(eps, elec, hh, dep)
    check("nested enclosures are compatible (they overlap)",
          not bf.clears(bc), f"coarse={bc}  fine={bf}")

    # ---- heterogeneous cell must sit inside the Wiener bounds ------------
    # A checkerboard of BLOCKS, not the stripe pattern one gets from a linear
    # index test: continuous stripes across the gap are a parallel arrangement
    # whose effective permittivity IS the arithmetic mean, i.e. exactly the upper
    # Wiener bound, so the check would sit on the boundary and be decided by
    # round-off.  Blocks put the answer strictly inside, which is what the test
    # is meant to verify.
    n = 40
    eps_h, eps_i = 78.0, 5.0
    ny = nx = n
    epsm = np.full((ny, nx), eps_h)
    yy, xx = np.mgrid[0:ny, 0:nx]
    mask = (yy >= 1) & (yy < ny - 1) & (yy % 4 < 2) & (xx % 4 < 2)
    epsm = np.where(mask, eps_i, epsm)
    f_true = mask[1:-1].mean()
    elec = np.zeros((ny, nx), dtype=int)
    elec[0, :] = HOT
    elec[-1, :] = GND
    h = 1e-6
    br = capacitance_bracket(epsm, elec, (h, h), 1e-3)
    lo_w, up_w = wiener_bounds(eps_i, eps_h, f_true)
    A = nx * h * 1e-3
    gap = (ny - 2) * h
    C_lo_w = EPS0 * lo_w * A / gap
    C_up_w = EPS0 * up_w * A / gap
    check("heterogeneous cell inside the Wiener bounds",
          C_lo_w <= br.lo and br.hi <= C_up_w,
          f"Wiener=[{C_lo_w:.4e},{C_up_w:.4e}] bracket={br}")

    # ---- mixing formulas -------------------------------------------------
    lo, up = hashin_shtrikman(5.0, 78.0, 0.2, d=3)
    mg = maxwell_garnett(5.0, 78.0, 0.2, d=3)
    check("Maxwell-Garnett lies inside Hashin-Shtrikman",
          lo <= mg <= up, f"HS=[{lo:.4f},{up:.4f}] MG={mg:.4f}")
    lo_w, up_w = wiener_bounds(5.0, 78.0, 0.2)
    check("Hashin-Shtrikman is tighter than Wiener",
          lo_w <= lo and up <= up_w,
          f"W=[{lo_w:.4f},{up_w:.4f}] HS=[{lo:.4f},{up:.4f}]")

    # ---- detection: a particle must move the bracket ---------------------
    eps0c, elec0, hh, dep = coplanar_cell(h=0.25e-6)
    b_blank = capacitance_bracket(eps0c, elec0, hh, dep)
    epsp, elecp, hh, dep = coplanar_cell(
        h=0.25e-6, particle=(6e-6, 5e-6, 3e-6, 5.0))
    b_samp = capacitance_bracket(epsp, elecp, hh, dep)
    check("a single insulating particle lowers the capacitance",
          b_samp.mid < b_blank.mid, f"blank={b_blank} sample={b_samp}")
    check("blank and sample enclosures are disjoint (certified detection)",
          b_samp.clears(b_blank),
          f"gap={(b_blank.lo - b_samp.hi) / b_blank.mid * 100:.3g}% of C")

    print("=" * 74)
    print(f"{'ALL TESTS PASSED' if _fail == 0 else str(_fail) + ' TEST(S) FAILED'}")
    raise SystemExit(1 if _fail else 0)
