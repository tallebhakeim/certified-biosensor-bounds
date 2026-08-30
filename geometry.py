"""
geometry.py -- voxel builders for the capacitive lab-on-chip cells.

Every device is described in PHYSICAL units and then voxelised on a uniform
grid of step h, so that a mesh refinement changes the discretisation only and
never the geometry.  Dirichlet planes that bound the physical domain (the gap
mid-plane of an interdigitated comb, the counter-electrode of a sandwich) are
imposed by padding the array with one extra layer of conductor cells outside
the physical domain: those cells carry no energy, so the physical domain is
untouched, and the electrode plane sits exactly on the physical boundary.

Convention for the interdigitated comb (IDE)
--------------------------------------------
  pitch  lam   : centre-to-centre distance between two adjacent fingers of
                 opposite polarity (the spatial period is 2*lam).
  ratio  eta    : finger width / pitch = w / lam.
  quarter cell  : x from the finger centre (mirror plane, Neumann) to the gap
                 centre (antisymmetry plane, V = 0), width lam/2.
  C_period = 2 * C_quarter   (per unit depth), see the note in the paper.
"""

import numpy as np
from certified_core import DIELECTRIC, HOT, GND

# default material set (relative permittivities)
EPS_BUFFER = 78.0        # phosphate-buffered saline, 1 MHz, 25 C
EPS_SUBSTRATE = 4.2      # borosilicate / thermal oxide
EPS_PARTICLE = 5.0       # membrane-insulated bioparticle
EPS_AIR = 1.0
EPS_PDMS = 2.6


def _n(length, h):
    """Number of cells spanning a physical length, snapped to the grid."""
    n = int(round(length / h))
    return max(n, 1)


class Cell:
    """A voxelised device ready for the certified bracket."""

    def __init__(self, eps, elec, h, depth=1.0, origin=(0.0, 0.0)):
        self.eps = eps
        self.elec = elec
        self.h = h
        self.depth = depth
        self.origin = origin

    @property
    def shape(self):
        return self.eps.shape

    def as_args(self):
        return self.eps, self.elec, self.h, self.depth


# --------------------------------------------------------------------------
# 2-D coplanar interdigitated comb, quarter cell
# --------------------------------------------------------------------------
def ide_quarter_cell(h, lam=40e-6, eta=0.5, t_metal=1e-6, h_sub=5e-6,
                     h_fluid=20e-6, eps_fluid=EPS_BUFFER, eps_sub=EPS_SUBSTRATE,
                     eps_cover=None, t_cover=0.0, depth=1.0,
                     particles=(), bound_layer=None, metal_edge=0.0,
                     eta_effective=None, snap="nearest"):
    """Quarter period of a coplanar comb.

    particles    : iterable of (x, y, radius, eps) with y measured from the
                   substrate floor (the top of the substrate).
    bound_layer  : (thickness, eps) of a bound analyte film covering the whole
                   floor (electrodes and gap), for a surface-binding assay.
    metal_edge   : lateral shift (m) applied to the finger edge; a positive
                   value widens the finger.  Used by the fabrication-tolerance
                   study.
    snap         : how the conductor is snapped to the grid.  "nearest" for the
                   nominal geometry; "in" rounds the conductor DOWN to the
                   largest grid electrode it contains, "out" rounds it UP to the
                   smallest grid electrode containing it.  The tolerance study
                   needs "in"/"out", because only then is the discrete electrode
                   a genuine subset (resp. superset) of the manufactured one and
                   the capacity monotonicity gives a guaranteed bound.  Rounding
                   to the nearest cell would silently lose the guarantee for any
                   tolerance below the mesh step.
    """
    # A tolerance that is an exact multiple of h must not be widened by one
    # cell through floating-point noise, so floor/ceil are given a slack of
    # 1e-6 cell -- negligible against any physical tolerance, and it keeps the
    # snapped conductor a subset (resp. superset) of the manufactured one.
    if snap == "in":
        def _q(x):
            return np.floor(x + 1e-6)
    elif snap == "out":
        def _q(x):
            return np.ceil(x - 1e-6)
    elif snap == "nearest":
        _q = np.round
    else:
        raise ValueError(f"snap must be nearest/in/out, got {snap!r}")
    nx = _n(lam / 2, h)
    ns = _n(h_sub, h)
    nf = _n(h_fluid, h)
    nc = _n(t_cover, h) if (eps_cover is not None and t_cover > 0) else 0
    nm = max(int(_q(t_metal / h)), 1)
    ny = ns + nf + nc

    eps = np.empty((ny, nx))
    eps[:ns] = eps_sub
    eps[ns:ns + nf] = eps_fluid
    if nc:
        eps[ns + nf:] = eps_cover
    elec = np.zeros((ny, nx), dtype=int)

    # bound analyte film on the floor
    if bound_layer is not None:
        t_b, eps_b = bound_layer
        nb = _n(t_b, h)
        eps[ns:ns + nb] = eps_b

    # the finger: metal of physical thickness t_metal sitting on the floor
    w_half = (eta if eta_effective is None else eta_effective) * lam / 2 + metal_edge
    nw = int(_q(w_half / h))
    nw = min(max(nw, 0), nx - 1)
    if nw > 0:
        elec[ns:ns + nm, :nw] = HOT

    # suspended particles
    for (px, py, pr, pe) in particles:
        yy, xx = np.mgrid[0:ny, 0:nx]
        xc = (xx + 0.5) * h
        yc = (yy + 0.5) * h - ns * h
        m = (xc - px) ** 2 + (yc - py) ** 2 < pr ** 2
        eps = np.where(m & (elec == DIELECTRIC), pe, eps)

    # gap mid-plane: antisymmetry -> V = 0, imposed on the right boundary
    pad = np.zeros((ny, 1), dtype=int) + GND
    elec = np.hstack([elec, pad])
    eps = np.hstack([eps, eps[:, -1:]])

    return Cell(eps, elec, (h, h), depth)


# --------------------------------------------------------------------------
# 2-D parallel-plate (sandwich) configuration
# --------------------------------------------------------------------------
def sandwich_cell(h, width=20e-6, h_fluid=20e-6, t_metal=1e-6,
                  eps_fluid=EPS_BUFFER, depth=1.0, particles=(),
                  bound_layer=None):
    """Two facing plates across the channel, one period of width `width`."""
    nx = _n(width, h)
    nf = _n(h_fluid, h)
    nm = _n(t_metal, h)
    ny = nf + 2 * nm
    eps = np.full((ny, nx), eps_fluid)
    elec = np.zeros((ny, nx), dtype=int)
    elec[:nm] = HOT
    elec[-nm:] = GND
    if bound_layer is not None:
        t_b, eps_b = bound_layer
        nb = _n(t_b, h)
        eps[nm:nm + nb] = eps_b
    for (px, py, pr, pe) in particles:
        yy, xx = np.mgrid[0:ny, 0:nx]
        xc = (xx + 0.5) * h
        yc = (yy + 0.5) * h - nm * h
        m = (xc - px) ** 2 + (yc - py) ** 2 < pr ** 2
        eps = np.where(m & (elec == DIELECTRIC), pe, eps)
    return Cell(eps, elec, (h, h), depth)


# --------------------------------------------------------------------------
# 3-D planar electrode patterns on the chip floor
# --------------------------------------------------------------------------
def _pattern_mask(kind, nx, nz, h, lam):
    """Boolean masks (hot, gnd) of a single-mask planar layout, top view."""
    x = (np.arange(nx) + 0.5) * h
    z = (np.arange(nz) + 0.5) * h
    X, Z = np.meshgrid(x, z, indexing="ij")
    Lx, Lz = nx * h, nz * h
    if kind == "comb":                       # P1 interdigitated fingers
        s = np.floor(X / (lam / 2)).astype(int) % 4
        hot, gnd = (s == 0), (s == 2)
    elif kind == "rings":                    # P2 concentric rings
        r = np.hypot(X - Lx / 2, Z - Lz / 2)
        s = np.floor(r / (lam / 2)).astype(int) % 4
        hot, gnd = (s == 0), (s == 2)
    elif kind == "checker":                  # P3 alternating patch array
        ix = np.floor(X / (lam / 2)).astype(int)
        iz = np.floor(Z / (lam / 2)).astype(int)
        par = (ix + iz) % 2
        fill = ((np.floor(X / (lam / 4)).astype(int) % 2 == 0) &
                (np.floor(Z / (lam / 4)).astype(int) % 2 == 0))
        hot, gnd = (par == 0) & fill, (par == 1) & fill
    elif kind == "spiral":                   # P4 double interleaved spiral
        th = np.arctan2(Z - Lz / 2, X - Lx / 2)
        r = np.hypot(X - Lx / 2, Z - Lz / 2)
        u = (r - lam * th / (2 * np.pi)) / (lam / 2)
        s = np.floor(u).astype(int) % 4
        hot, gnd = (s == 0), (s == 2)
    else:
        raise ValueError(kind)
    return hot, gnd


def planar_pattern_3d(h, kind="comb", lam=20e-6, span=80e-6, h_fluid=20e-6,
                      t_metal=1e-6, h_sub=4e-6, eps_fluid=EPS_BUFFER,
                      eps_sub=EPS_SUBSTRATE, bound_layer=None):
    """Fabricable single-mask planar pattern solved in genuine 3-D.

    Axes: (y = height, x, z), the electrodes lying on the chip floor and the
    field arcing up into the fluid.
    """
    nx = _n(span, h)
    nz = _n(span, h)
    ns = _n(h_sub, h)
    nf = _n(h_fluid, h)
    nm = _n(t_metal, h)
    ny = ns + nf

    eps = np.empty((ny, nx, nz))
    eps[:ns] = eps_sub
    eps[ns:] = eps_fluid
    elec = np.zeros((ny, nx, nz), dtype=int)

    if bound_layer is not None:
        t_b, eps_b = bound_layer
        nb = _n(t_b, h)
        eps[ns:ns + nb] = eps_b

    hot, gnd = _pattern_mask(kind, nx, nz, h, lam)
    for j in range(ns, ns + nm):
        elec[j][hot] = HOT
        elec[j][gnd] = GND
    return Cell(eps, elec, (h, h, h), 1.0)


# --------------------------------------------------------------------------
# 3-D slice of the comb, for a genuine SINGLE-particle calculation
# --------------------------------------------------------------------------
def ide_slice_3d(h, lam=40e-6, eta=0.5, t_metal=200e-9, h_sub=5e-6,
                 h_fluid=20e-6, lz=20e-6, eps_fluid=EPS_BUFFER,
                 eps_sub=EPS_SUBSTRATE, spheres=()):
    """Quarter period of the comb, extruded over a length lz along the finger.

    Why this exists.  In a two-dimensional cross-section a "particle" is an
    infinite cylinder running the whole finger length, so a 2-D result describes
    a LINE DENSITY of particles, never an isolated one.  A single-particle claim
    therefore needs a three-dimensional cell, which is what this builds: the same
    quarter cell as ide_quarter_cell, extruded over lz, with Neumann (symmetry)
    faces at both ends of the extrusion.

    Axes are (y = height, x = across the fingers, z = along the finger).
    spheres : iterable of (x, y, z, radius, eps), y measured from the floor.
    """
    nx = _n(lam / 2, h)
    ns = _n(h_sub, h)
    nf = _n(h_fluid, h)
    nz = _n(lz, h)
    nm = max(int(round(t_metal / h)), 1)
    ny = ns + nf

    eps = np.empty((ny, nx, nz))
    eps[:ns] = eps_sub
    eps[ns:] = eps_fluid
    elec = np.zeros((ny, nx, nz), dtype=int)

    nw = int(round(eta * lam / 2 / h))
    nw = min(max(nw, 1), nx - 1)
    elec[ns:ns + nm, :nw, :] = HOT

    for (px, py, pz, pr, pe) in spheres:
        g = (np.arange(max(ny, nx, nz)) + 0.5) * h
        Y, X, Z = np.meshgrid(g[:ny], g[:nx], g[:nz], indexing="ij")
        Y = Y - ns * h
        m = ((X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2) < pr ** 2
        eps = np.where(m & (elec == DIELECTRIC), pe, eps)

    # gap mid-plane: antisymmetry -> V = 0 on the far x face
    pad = np.full((ny, 1, nz), GND, dtype=int)
    elec = np.concatenate([elec, pad], axis=1)
    eps = np.concatenate([eps, eps[:, -1:, :]], axis=1)
    return Cell(eps, elec, (h, h, h), 1.0)


# --------------------------------------------------------------------------
# 3-D suspension between two plates, for the homogenisation check
# --------------------------------------------------------------------------
def suspension_cube_3d(h, side=10e-6, radius=1.5e-6, n_part=0, seed=0,
                       eps_h=EPS_BUFFER, eps_p=EPS_PARTICLE, margin=0):
    """A cube of buffer holding n_part spheres, between two parallel plates.

    Used to test the homogenisation bounds against a resolved microstructure:
    the certified bracket of this cell is compared with the bracket obtained by
    replacing the suspension by an effective permittivity.  The plates are the
    top and bottom faces, so the blank cell is an exact parallel-plate capacitor
    and the reference value is analytic.

    Spheres are placed by random sequential addition without overlap and are
    clipped by the cube faces, so the two-phase medium fills the whole gap and
    no pure-buffer layer is left against the plates (such a layer would be a
    series capacitor, not a homogeneous medium, and would leave the effective
    medium comparison meaningless).  Returns (cell, f) with f the volume
    fraction actually voxelised, which is what the bounds must be evaluated at.
    """
    n = _n(side, h)
    eps = np.full((n, n, n), eps_h)
    rng = np.random.default_rng(seed)
    centres = []
    tries = 0
    while len(centres) < n_part and tries < 20000:
        tries += 1
        c = rng.uniform(-radius, side + radius, size=3)
        if all(np.linalg.norm(c - o) >= 2 * radius for o in centres):
            centres.append(c)
    g = (np.arange(n) + 0.5) * h
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    inside = np.zeros((n, n, n), dtype=bool)
    for c in centres:
        inside |= ((X - c[0]) ** 2 + (Y - c[1]) ** 2 +
                   (Z - c[2]) ** 2) < radius ** 2
    eps[inside] = eps_p
    f = float(inside.mean())

    elec = np.zeros((n, n, n), dtype=int)
    pad_hot = np.full((1, n, n), HOT, dtype=int)
    pad_gnd = np.full((1, n, n), GND, dtype=int)
    elec = np.concatenate([pad_hot, elec, pad_gnd], axis=0)
    eps = np.concatenate([eps[:1], eps, eps[-1:]], axis=0)
    return Cell(eps, elec, (h, h, h), 1.0), f


# --------------------------------------------------------------------------
# serpentine device, sensing zone cross-section
# --------------------------------------------------------------------------
def sensing_zone_cell(h, w_chan=40e-6, h_chan=20e-6, t_metal=1e-6,
                      w_elec=15e-6, eps_fluid=EPS_BUFFER,
                      eps_sub=EPS_SUBSTRATE, depth=1.0, bolus=None):
    """Cross-section of one straight sensing channel read by a coplanar pair.

    bolus : (eps,) of a sample plug filling the channel, or None for the blank.
    """
    nx = _n(w_chan, h)
    ns = _n(4e-6, h)
    nc = _n(h_chan, h)
    nm = _n(t_metal, h)
    ny = ns + nc + _n(4e-6, h)
    eps = np.full((ny, nx), eps_sub)
    eps[ns:ns + nc] = eps_fluid if bolus is None else bolus
    elec = np.zeros((ny, nx), dtype=int)
    ne = _n(w_elec, h)
    elec[ns:ns + nm, :ne] = HOT
    elec[ns:ns + nm, -ne:] = GND
    return Cell(eps, elec, (h, h), depth)


def serpentine_layout(h, w_chan=40e-6, n_turn=4, leg=300e-6, pitch=120e-6,
                      margin=40e-6):
    """Top view of a serpentine micro-mixer: boolean mask of the channel.

    Returns (mask, h, inlet_slice, outlet_slice) for the Hele-Shaw solve.
    """
    height = 2 * margin + (n_turn - 1) * pitch + w_chan
    width = 2 * margin + leg
    nx = _n(width, h)
    ny = _n(height, h)
    mask = np.zeros((ny, nx), dtype=bool)
    nw = _n(w_chan, h)
    nm = _n(margin, h)
    npi = _n(pitch, h)
    for k in range(n_turn):
        y0 = nm + k * npi
        mask[y0:y0 + nw, nm:nx - nm] = True
        if k < n_turn - 1:                      # vertical connector
            if k % 2 == 0:
                mask[y0:y0 + npi + nw, nx - nm - nw:nx - nm] = True
            else:
                mask[y0:y0 + npi + nw, nm:nm + nw] = True
    inlet = (slice(nm, nm + nw), 0)
    outlet = (slice(nm + (n_turn - 1) * npi, nm + (n_turn - 1) * npi + nw), nx - 1)
    return mask, (h, h), inlet, outlet
