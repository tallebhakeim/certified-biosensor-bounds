"""
certified_core.py -- guaranteed two-sided bracket of a capacitance on a
structured (voxel) grid, in 2-D and 3-D.

The bracket realises the complementary (Prager-Synge / hypercircle) pair:

  * UPPER bound  -- Dirichlet (primal) principle.
      C = min  (1/V^2) int eps |grad psi|^2   over psi admissible.
      A conforming Q1 nodal finite-element field is admissible, so its
      energy over-estimates C.   This is the "capacitive / shunt node"
      member of the dual TLM pair.

  * LOWER bound  -- Thomson (complementary) principle.
      1/C = min (1/Q^2) int |D|^2 / eps  over D with div D = 0 carrying
      the electrode charge Q.
      A cell-centred finite-volume (two-point-flux) solve produces face
      fluxes that are conservative to machine precision; their lowest
      order Raviart-Thomas reconstruction is therefore an *admissible*
      equilibrated flux, and its exactly integrated co-energy
      under-estimates C.  This is the "inductive / series node" member.

Both members are computed on the same voxel description of the device, so
the returned interval [C_lo, C_up] is a guaranteed enclosure of the exact
capacitance of the *discretised geometry*: it certifies the numerical
(discretisation) error only.  Model error and experimental error are
handled elsewhere (see model_bounds.py, edl_window.py, noise_budget.py).

Author: H. Talleb -- IEEE Sensors Journal, manuscript Sensors-109944-2026.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS0 = 8.8541878128e-12          # F/m

DIELECTRIC = 0
HOT = 1                          # electrode held at V = 1
GND = 2                          # electrode held at V = 0


# --------------------------------------------------------------------------
# element matrices
# --------------------------------------------------------------------------
DIRECT_MAX = 60000          # unknowns above which the direct solve is hopeless


def _solve_spd(A, b, tol=1e-13):
    """Solve A x = b for a symmetric positive-definite sparse A.

    A sparse direct factorisation of a 3-D Laplacian fills in cubically (a 40^3
    cell mesh already takes over a minute), so large systems go to algebraic
    multigrid accelerated by conjugate gradients instead, which is near linear.

    Returns (x, relres) with relres the relative residual actually achieved.
    The caller must decide what to do with it, and the two bounds differ:

      * the conforming bound is a MINIMISATION, so any admissible potential is
        an upper bound and an unconverged iterate stays valid, merely looser;
      * the equilibrated bound needs the flux to be exactly conservative, so a
        non-zero residual is a spurious charge that would break admissibility.
        `capacitance_lower` therefore checks the residual explicitly rather than
        trusting the solver.
    """
    # A micrometre-scale geometry gives matrix entries around 1e-7, and the
    # coarse-grid pseudo-inverse inside the multigrid hierarchy then overflows.
    # Scaling by the mean diagonal leaves the solution unchanged exactly
    # (A x = b  <=>  (A/s) x = b/s) and puts the spectrum near unity.
    n = A.shape[0]
    s = float(np.mean(np.abs(A.diagonal()))) or 1.0
    As, bs = A / s, b / s
    if n <= DIRECT_MAX:
        x = spla.spsolve(As.tocsc(), bs)
    else:
        import pyamg
        # pyamg builds its coarsest-level solve from a pseudo-inverse, which
        # legitimately divides by zero singular values and zeroes them again; the
        # warning is expected and says nothing about the result.  What actually
        # guards the bound is the finiteness test and the residual returned
        # below, both checked explicitly.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ml = pyamg.smoothed_aggregation_solver(As.tocsr(), max_coarse=500)
            x = ml.solve(bs, tol=tol, accel="cg", maxiter=1000)
    if not np.all(np.isfinite(x)):
        raise RuntimeError("non-finite solution -- singular or ill-scaled system")
    r = float(np.linalg.norm(As @ x - bs) / max(np.linalg.norm(bs), 1e-300))
    return x, r


def _q1_element_stiffness(h):
    """Stiffness matrix of a Q1 brick of size h = (hx, hy[, hz]), eps = 1.

    Exact (2-point Gauss per direction integrates the bilinear/trilinear
    gradients exactly).  Node ordering is the lexicographic corner order
    produced by np.indices, i.e. index = sum_d c_d * stride_d with c_d in
    {0, 1} and the last direction fastest.
    """
    d = len(h)
    g = 1.0 / np.sqrt(3.0)
    gp = np.array([-g, g])
    corners = np.array(np.meshgrid(*[[0, 1]] * d, indexing="ij")).reshape(d, -1).T
    nn = corners.shape[0]
    Ke = np.zeros((nn, nn))
    qpts = np.array(np.meshgrid(*[gp] * d, indexing="ij")).reshape(d, -1).T
    wq = (1.0) ** d                       # weight of each Gauss point in [-1,1]^d
    detJ = np.prod([hi / 2.0 for hi in h])
    for xi in qpts:
        # gradients of the multilinear shape functions on the reference cube
        dN = np.zeros((nn, d))
        for j in range(d):
            col = np.ones(nn)
            for k in range(d):
                if k == j:
                    col *= np.where(corners[:, k] == 0, -0.5, 0.5) * (2.0 / h[k])
                else:
                    col *= np.where(corners[:, k] == 0, (1 - xi[k]) / 2, (1 + xi[k]) / 2)
            dN[:, j] = col
        Ke += wq * detJ * (dN @ dN.T)
    return Ke


# --------------------------------------------------------------------------
# conforming (upper) bound
# --------------------------------------------------------------------------
def capacitance_upper(eps, elec, h, depth=1.0, return_field=False):
    """Conforming Q1 nodal solve -> guaranteed UPPER bound of C.

    Parameters
    ----------
    eps   : ndarray, cell-wise relative permittivity, shape (n0, n1[, n2]).
    elec  : ndarray of int, same shape, values DIELECTRIC / HOT / GND.
    h     : tuple of cell sizes (m), one per axis.
    depth : out-of-plane thickness in 2-D (m).  Ignored in 3-D.

    Returns
    -------
    C_up (F), and optionally the nodal potential array.
    """
    eps = np.asarray(eps, dtype=float)
    elec = np.asarray(elec, dtype=int)
    d = eps.ndim
    dims = eps.shape
    ndims = tuple(n + 1 for n in dims)
    nnod = int(np.prod(ndims))

    Ke = _q1_element_stiffness(h)
    if d == 2:
        Ke = Ke * depth

    # node index of every element corner
    idx = np.arange(nnod).reshape(ndims)
    corners = np.array(np.meshgrid(*[[0, 1]] * d, indexing="ij")).reshape(d, -1).T
    sl0 = tuple(slice(0, n) for n in dims)
    conn = np.stack(
        [idx[tuple(slice(c[k], c[k] + dims[k]) for k in range(d))].ravel() for c in corners],
        axis=1,
    )  # (nel, 2^d)

    active = (elec == DIELECTRIC).ravel()
    epsv = eps.ravel()

    # Dirichlet nodes: every node touching a conductor cell
    fixed = np.zeros(nnod, dtype=int) - 1           # -1 free, else potential index
    val = np.zeros(nnod)
    for code, v in ((HOT, 1.0), (GND, 0.0)):
        cells = np.flatnonzero((elec.ravel() == code))
        if cells.size:
            nodes = np.unique(conn[cells].ravel())
            fixed[nodes] = code
            val[nodes] = v

    ael = np.flatnonzero(active)
    ce = conn[ael]
    data = (epsv[ael][:, None, None] * Ke[None, :, :]).ravel()
    rows = np.repeat(ce, ce.shape[1], axis=1).ravel()
    cols = np.tile(ce, (1, ce.shape[1])).ravel()
    # assembled with the RELATIVE permittivity only: eps0 (1e-11) times a
    # micrometre-scale geometry would otherwise give pivots near 1e-21 and a
    # numerically singular factorisation.  The vacuum constant is restored on
    # the energy at the end.
    K = sp.coo_matrix((data, (rows, cols)), shape=(nnod, nnod)).tocsr()

    free = np.flatnonzero(fixed < 0)
    # nodes not touched by any active element are also removed
    touched = np.zeros(nnod, dtype=bool)
    touched[np.unique(ce.ravel())] = True
    free = free[touched[free]]

    u = val.copy()
    if free.size:
        Kff = K[free][:, free]
        rhs = -K[free] @ val
        # an inexact solve here only loosens the upper bound, it cannot break it
        u[free], _ = _solve_spd(Kff, rhs)

    # The BLAS behind numpy raises spurious FPU flags on this dot product
    # (vectorised kernels on padded tails); the value itself is checked below.
    with np.errstate(all="ignore"):
        C_up = EPS0 * float(u @ (K @ u))           # V = 1
    if not np.isfinite(C_up):
        raise RuntimeError("non-finite conforming energy -- singular system")
    if return_field:
        return C_up, u.reshape(ndims)
    return C_up


# --------------------------------------------------------------------------
# equilibrated (lower) bound
# --------------------------------------------------------------------------
def capacitance_lower(eps, elec, h, depth=1.0, return_flux=False):
    """Cell-centred finite-volume solve -> conservative face fluxes ->
    exact RT0 co-energy -> guaranteed LOWER bound of C.

    The flux field is divergence free in every dielectric cell to machine
    precision and carries the electrode charge, so it is admissible for the
    Thomson principle and C >= Q^2 / (2 W*).
    """
    eps = np.asarray(eps, dtype=float)
    elec = np.asarray(elec, dtype=int)
    d = eps.ndim
    dims = eps.shape
    ncell = int(np.prod(dims))
    cid = np.arange(ncell).reshape(dims)

    vol = np.prod(h) * (depth if d == 2 else 1.0)
    area = [vol / h[k] for k in range(d)]           # face area normal to axis k

    is_cond = elec != DIELECTRIC
    pot = np.where(elec == HOT, 1.0, 0.0)

    cond_flat = is_cond.ravel()
    pot_flat = pot.ravel()
    elec_flat = elec.ravel()

    # every internal face: (axis, low cell, high cell, transmissibility).
    # A face between two conductor cells carries no current: T = 0 there.
    faces = []
    for k in range(d):
        sl_lo = tuple(slice(0, dims[j] - 1) if j == k else slice(None) for j in range(d))
        sl_hi = tuple(slice(1, dims[j]) if j == k else slice(None) for j in range(d))
        e_lo, e_hi = eps[sl_lo], eps[sl_hi]
        eh = 2.0 * e_lo * e_hi / (e_lo + e_hi)          # harmonic mean
        # relative permittivity only; eps0 is restored on the capacitance
        T = (eh * area[k] / h[k]).ravel()
        a, b = cid[sl_lo].ravel(), cid[sl_hi].ravel()
        # a conductor holds its potential on the FACE, not at the cell centre,
        # so a conductor/dielectric face sees only the dielectric half-cell
        ca, cb = cond_flat[a], cond_flat[b]
        half = (area[k] / (0.5 * h[k])) * np.where(ca, eps.ravel()[b], eps.ravel()[a])
        T = np.where(ca ^ cb, half, T)
        T = np.where(ca & cb, 0.0, T)
        faces.append((k, a, b, T))

    # assemble the conservation law  sum_faces T (p_i - p_j) = 0  on free cells
    rows, cols, data = [], [], []
    rhs = np.zeros(ncell)
    for k, a, b, T in faces:
        for i, j in ((a, b), (b, a)):
            m = ~cond_flat[i]                            # equation written on cell i
            rows.append(i[m]); cols.append(i[m]); data.append(T[m])
            mj = m & cond_flat[j]                        # neighbour is a conductor
            np.add.at(rhs, i[mj], T[mj] * pot_flat[j[mj]])
            mo = m & ~cond_flat[j]
            rows.append(i[mo]); cols.append(j[mo]); data.append(-T[mo])

    A = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ncell, ncell),
    ).tocsr()

    p = pot_flat.copy()
    freec = np.flatnonzero(~cond_flat)
    Aff = A[freec][:, freec].tocsc()
    diag = np.asarray(Aff.diagonal()).ravel()
    dead = diag == 0                                     # isolated cells: pin to 0
    if dead.any():
        Aff = Aff + sp.diags(np.where(dead, 1.0, 0.0))
    p[freec], relres = _solve_spd(Aff, rhs[freec])
    # The flux is admissible for the Thomson principle only if it is genuinely
    # conservative, so the algebraic error is not assumed away: a residual in the
    # conservation law is a spurious charge in the cells.  The bound is claimed
    # only when that residual is negligible against the electrode charge.
    if relres > 1e-9:
        raise RuntimeError(
            f"conservation residual {relres:.2e} too large: the reconstructed "
            "flux is not divergence free, so the lower bound is not admissible")

    # face fluxes (positive along +axis) and the charge leaving the HOT electrode
    flux = [np.zeros(tuple(n - 1 if j == k else n for j, n in enumerate(dims)))
            for k in range(d)]
    Q = 0.0
    for k, a, b, T in faces:
        F = T * (p[a] - p[b])
        flux[k] = F.reshape(flux[k].shape)
        Q += float(F[(elec_flat[a] == HOT) & ~cond_flat[b]].sum())
        Q += float((-F)[(elec_flat[b] == HOT) & ~cond_flat[a]].sum())

    # exact co-energy of the RT0 reconstruction, cell by cell
    W2 = 0.0                                            # = 2 * W*
    for k in range(d):
        Fl = np.zeros(dims)
        Fr = np.zeros(dims)
        sl_int_lo = tuple(slice(1, dims[j]) if j == k else slice(None) for j in range(d))
        sl_int_hi = tuple(slice(0, dims[j] - 1) if j == k else slice(None) for j in range(d))
        Fl[sl_int_lo] = flux[k]                          # flux entering through low face
        Fr[sl_int_hi] = flux[k]                          # flux leaving through high face
        # int over the cell of D_k^2 = (h_k/A_k) * (Fl^2 + Fl*Fr + Fr^2)/3
        contrib = (h[k] / area[k]) * (Fl ** 2 + Fl * Fr + Fr ** 2) / 3.0
        W2 += np.sum(np.where(is_cond, 0.0, contrib / eps))

    C_lo = EPS0 * Q ** 2 / W2 if W2 > 0 else 0.0
    if return_flux:
        return C_lo, flux, p.reshape(dims), Q
    return C_lo


# --------------------------------------------------------------------------
# the bracket
# --------------------------------------------------------------------------
class Bracket:
    """A guaranteed enclosure [lo, hi] of a capacitance."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)

    @property
    def mid(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def halfwidth(self):
        return 0.5 * (self.hi - self.lo)

    @property
    def relwidth(self):
        """Half-width relative to the midpoint (per cent)."""
        return 100.0 * self.halfwidth / self.mid if self.mid else np.inf

    def clears(self, other):
        """True when the two enclosures are disjoint (a certified difference)."""
        return self.lo > other.hi or other.lo > self.hi

    def __repr__(self):
        return f"[{self.lo:.6g}, {self.hi:.6g}] (+/-{self.relwidth:.3g}%)"


def capacitance_bracket(eps, elec, h, depth=1.0):
    """Guaranteed two-sided enclosure of the capacitance."""
    hi = capacitance_upper(eps, elec, h, depth)
    lo = capacitance_lower(eps, elec, h, depth)
    if lo > hi:
        # On a cell where the two members are exact (a 1-D field) the ordering
        # can be inverted by round-off only.  Anything larger is a real defect.
        if lo > hi * (1 + 1e-9):
            raise RuntimeError("lower bound above upper bound -- inadmissible pair")
        lo, hi = min(lo, hi), max(lo, hi)
    return Bracket(lo, hi)


# --------------------------------------------------------------------------
# classical mixing bounds, used as an independent sanity check
# --------------------------------------------------------------------------
def wiener_bounds(eps_a, eps_b, f):
    """Arithmetic (upper) and harmonic (lower) mean of a two-phase mixture."""
    up = f * eps_a + (1 - f) * eps_b
    lo = 1.0 / (f / eps_a + (1 - f) / eps_b)
    return lo, up


def hashin_shtrikman(eps_i, eps_h, f, d=3):
    """Hashin-Shtrikman bounds for an isotropic two-phase mixture.

    eps_i : inclusion permittivity, eps_h : host permittivity,
    f     : inclusion volume fraction, d : space dimension.
    Returns (lower, upper); they are the tightest bounds that use only the
    volume fraction and isotropy, and they coincide with Maxwell-Garnett
    when the coated-sphere assemblage is realised.
    """
    def mg(e_in, e_out, frac):
        # Maxwell-Garnett with e_out as host, valid formula in d dimensions
        num = e_in - e_out
        den = e_in + (d - 1) * e_out
        beta = num / den
        return e_out * (1 + (d - 1) * frac * beta) / (1 - frac * beta)

    b1 = mg(eps_i, eps_h, f)          # host = eps_h
    b2 = mg(eps_h, eps_i, 1 - f)      # host = eps_i
    return (min(b1, b2), max(b1, b2))


def maxwell_garnett(eps_i, eps_h, f, d=3):
    """Maxwell-Garnett effective permittivity (inclusions in a host)."""
    beta = (eps_i - eps_h) / (eps_i + (d - 1) * eps_h)
    return eps_h * (1 + (d - 1) * f * beta) / (1 - f * beta)
