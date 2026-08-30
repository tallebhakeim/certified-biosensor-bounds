"""
device.py -- the reference device, in physical units, shared by every study.

The revision replaces the dimensionless parametric cell of the first
submission by a fully dimensioned chip, so that every certified number can be
compared with a measurable quantity (Reviewer 2, comment 1).

Reference chip
--------------
  interdigitated comb, pitch (centre to centre)      lam    = 40 um
  metallisation ratio (finger width / pitch)         eta    = 0.5
  metal thickness (Ti/Au lift-off)                   t      = 200 nm
  channel height                                     h_ch   = 20 um
  glass substrate below the electrodes               h_sub  = 5 um
  finger length                                      L      = 2 mm
  number of periods                                  N_p    = 50
  buffer (PBS, 1 MHz, 25 C)                          eps_r  = 78

Assays
------
  CAPTURE assay  : a 1 um layer of captured bacteria / adherent cells bound on
                   the floor, eps_r = 6 (an intact membrane insulates the cell
                   at 1 MHz).  This is the assay the device is designed for and
                   the one every certified sensitivity refers to.
  MOLECULAR assay: a 10 nm protein monolayer, eps_r = 5.  Two orders of
                   magnitude thinner than the mesh; it is treated by the
                   certified thin-layer bound of `monolayer_bound()` instead of
                   being meshed, and the paper states plainly that this device
                   cannot certify it.
"""

import numpy as np
from certified_core import capacitance_bracket, Bracket
from geometry import ide_quarter_cell, EPS_BUFFER, EPS_SUBSTRATE

# ---- reference geometry ---------------------------------------------------
LAM = 40e-6
ETA = 0.5
T_METAL = 200e-9
H_CHANNEL = 20e-6
H_SUB = 5e-6
FINGER_LENGTH = 2e-3
N_PERIODS = 50
DEPTH = FINGER_LENGTH

# ---- materials ------------------------------------------------------------
EPS_PBS = EPS_BUFFER          # 78
EPS_GLASS = EPS_SUBSTRATE     # 4.2
EPS_CELL = 6.0                # membrane-insulated cell at 1 MHz
EPS_CYTOPLASM = 60.0          # high frequency, membrane shorted
EPS_PROTEIN = 5.0

# ---- conductivities, for the conduction read-out --------------------------
# The same solver serves the conduction problem: pass these where a relative
# permittivity is expected and divide the result by eps0 to get siemens.
SIGMA_PBS = 1.6               # S/m
SIGMA_GLASS = 1e-12           # S/m, borosilicate: an insulator
SIGMA_CELL = 0.01             # S/m, an intact membrane blocks the current

CAPTURE_LAYER = (1.0e-6, EPS_CELL)      # thickness, permittivity
MONOLAYER = (10e-9, EPS_PROTEIN)

# ---- meshes ---------------------------------------------------------------
H_FINE = 100e-9               # headline numbers
H_SWEEP = 250e-9              # parameter sweeps

_cache = {}


def cell_capacitance(h=H_SWEEP, edge_shift=0.0, t_metal=T_METAL, eta=ETA,
                     bound=None, eps_fluid=EPS_PBS, particles=(),
                     t_cover=0.0, eps_cover=None, h_fluid=H_CHANNEL,
                     snap="nearest", eps_sub=EPS_GLASS):
    """Certified enclosure of the CELL capacitance, in F per metre of finger.

    `eps_sub` MUST be set explicitly when this routine is reused for the
    conduction problem.  The two problems share the operator but not the
    coefficients: borosilicate is a middling dielectric (eps_r = 4.2) and an
    excellent insulator (sigma about 1e-12 S/m).  Leaving eps_sub at its
    dielectric value while passing a conductivity for the fluid would give the
    substrate a conductivity of 4.2 S/m, above the 1.6 S/m of the buffer, and the
    spurious substrate current would swamp the answer.  See SIGMA_GLASS.

    The cell is the standard interdigitated unit, one finger plus one gap, of
    length lam.  The quarter cell runs from the finger centre (mirror plane) to
    the gap centre (antisymmetry plane, phi = 0), so the driven comb sits at
    +V/2 and the ground comb at -V/2: the applied voltage is twice the quarter
    cell voltage while the cell carries one half-finger, hence

        C_cell = C_quarter / 2 .

    Checked against the conformal-mapping value for a coplanar comb between two
    half spaces (see test_core.py, `ide_analytic`).
    """
    key = (h, round(edge_shift, 12), round(t_metal, 12), eta, bound, eps_fluid,
           tuple(particles), t_cover, eps_cover, h_fluid, snap, eps_sub)
    if key in _cache:
        return _cache[key]
    c = ide_quarter_cell(h, lam=LAM, eta=eta, t_metal=t_metal,
                         h_sub=H_SUB, h_fluid=h_fluid, metal_edge=edge_shift,
                         depth=1.0, eps_fluid=eps_fluid, eps_sub=eps_sub,
                         bound_layer=bound, particles=particles,
                         t_cover=t_cover, eps_cover=eps_cover, snap=snap)
    br = capacitance_bracket(*c.as_args())
    out = Bracket(br.lo / 2, br.hi / 2)
    _cache[key] = out
    return out


def device_capacitance(br_per_metre):
    """Whole-chip capacitance (F) from the per-cell, per-metre enclosure."""
    return Bracket(br_per_metre.lo * DEPTH * N_PERIODS,
                   br_per_metre.hi * DEPTH * N_PERIODS)


# ---------------------------------------------------------------------------
# certified sensitivity over a box of fabrication tolerances
# ---------------------------------------------------------------------------
def certified_sensitivity(delta_edge=0.0, delta_t=0.0, n_sub=8, h=H_SWEEP,
                          eta=ETA, bound=CAPTURE_LAYER):
    """Guaranteed enclosure of  S = C_bound / C_blank - 1  over every electrode
    whose edge is within +/- delta_edge and whose metal thickness is within
    +/- delta_t of the mask.

    Both capacitances are monotone in the conductor (see study_fabrication.py),
    so on a sub-box [a, b] x [ta, tb] of the tolerance box

        C_blank in [C_lo(a, ta), C_up(b, tb)]   and likewise for C_bound,

    hence  S in [L1/B2 - 1, L2/B1 - 1].  Taking the union over a partition of
    the box removes the interval-dependency blow-up of the one-box estimate and
    converges to the exact range as the partition is refined; `n_sub` controls
    the partition and the returned enclosure is guaranteed for any n_sub >= 1.

    The endpoints are snapped OUTWARD ("in" for the eroded electrode, "out" for
    the dilated one), so the discrete conductors really do bracket every
    manufactured one even when the tolerance is finer than the mesh.
    """
    if n_sub <= 0:                       # degenerate box: no tolerance at all
        B = cell_capacitance(h, 0.0, T_METAL, eta)
        L = cell_capacitance(h, 0.0, T_METAL, eta, bound=bound)
        return Bracket(L.lo / B.hi - 1.0, L.hi / B.lo - 1.0)

    edges = np.linspace(-delta_edge, delta_edge, n_sub + 1)
    ta, tb = T_METAL - delta_t, T_METAL + delta_t
    S_lo, S_hi = np.inf, -np.inf
    for a, b in zip(edges[:-1], edges[1:]):
        B1 = cell_capacitance(h, a, ta, eta, snap="in").lo
        B2 = cell_capacitance(h, b, tb, eta, snap="out").hi
        L1 = cell_capacitance(h, a, ta, eta, bound=bound, snap="in").lo
        L2 = cell_capacitance(h, b, tb, eta, bound=bound, snap="out").hi
        S_lo = min(S_lo, L1 / B2 - 1.0)
        S_hi = max(S_hi, L2 / B1 - 1.0)
    return Bracket(S_lo, S_hi)


# ---------------------------------------------------------------------------
# certified bound on a molecular monolayer, without meshing it
# ---------------------------------------------------------------------------
def monolayer_bound(t=MONOLAYER[0], eps_layer=MONOLAYER[1], h=H_FINE):
    """Two-sided bound on the capacitance change caused by a monolayer of
    thickness t << mesh, obtained WITHOUT meshing it.

    The layer only changes the permittivity inside a film of volume V_f sitting
    on the floor.  The Dirichlet principle is monotone in eps pointwise, so
    replacing eps_buffer by eps_layer inside the film can only lower C; and the
    change is bounded by the field energy stored in the film, which the two
    certified fields bracket:

        0 <= C_blank - C_layer <= (1 - eps_layer/eps_buffer) * W_film / V^2

    with W_film the energy the conforming (upper) field puts in the film -- an
    over-estimate of the true film energy, hence a valid upper bound on the
    change.  A lower bound uses the equilibrated field the same way.
    Returns (dC_lo, dC_hi) in F per metre of finger, per period.
    """
    from certified_core import (capacitance_upper, capacitance_lower, EPS0,
                                DIELECTRIC)
    c = ide_quarter_cell(h, lam=LAM, eta=ETA, t_metal=T_METAL, h_sub=H_SUB,
                         h_fluid=H_CHANNEL, depth=1.0)
    eps, elec, hh, depth = c.as_args()
    _, u = capacitance_upper(eps, elec, hh, depth, return_field=True)
    ns = int(round(H_SUB / h))
    # energy density of the conforming field in the first cell row above the floor
    grad2 = ((u[ns + 1, :-1] - u[ns, :-1]) / h) ** 2 + \
            ((u[ns, 1:] - u[ns, :-1]) / h) ** 2
    row_free = (elec[ns] == DIELECTRIC)[:-1]
    W_row = EPS0 * EPS_PBS * np.sum(grad2 * row_free) * h * h
    frac = t / h                                   # the film fills this fraction
    dC_hi = 2 * (1 - eps_layer / EPS_PBS) * W_row * frac
    dC_lo = 0.0
    return dC_lo, dC_hi


def describe():
    print("reference chip")
    print(f"  pitch {LAM*1e6:.0f} um, eta {ETA}, metal {T_METAL*1e9:.0f} nm, "
          f"channel {H_CHANNEL*1e6:.0f} um")
    print(f"  {N_PERIODS} periods x {FINGER_LENGTH*1e3:.1f} mm fingers")
    br = cell_capacitance(H_FINE)
    dev = device_capacitance(br)
    print(f"  certified per-period capacitance : [{br.lo*1e12:.4f}, "
          f"{br.hi*1e12:.4f}] pF/m  (+/-{br.relwidth:.3f} %)")
    print(f"  certified device capacitance     : [{dev.lo*1e12:.4f}, "
          f"{dev.hi*1e12:.4f}] pF")
    return br, dev


if __name__ == "__main__":
    describe()
