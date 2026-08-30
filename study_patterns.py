"""
study_patterns.py -- certified ranking of fabricable planar electrode patterns.

Answers Reviewer 1 comment 5.  The selection criterion, left implicit in the
first submission, is stated here: these are the two-terminal layouts realisable
in a SINGLE lithography step on the channel floor, which is the constraint that
makes a design comparison meaningful for a disposable chip.  Multi-mask,
out-of-plane and three-dimensional electrodes are excluded by that criterion and
not by any claim about their performance.

Everything is solved in genuine three dimensions on a voxel mesh: the electrodes
lie on the floor and the field arcs up into the fluid.  A two-dimensional
cross-section cannot rank these patterns at all, since rings, checkerboards and
spirals differ precisely in the direction a cross-section integrates out.

The metal is taken 1 um thick here (an electroplated electrode) rather than the
200 nm of the reference chip, because the 3-D mesh cannot resolve 200 nm over an
80 um span at an affordable cost.  The value is the same for all four patterns,
so the comparison is unaffected; the absolute sensitivities are not the reference
chip's and are not quoted as such.

The ranking verdict follows the criterion established in study_fabrication.py:
two designs are certified-ranked only when their enclosures are disjoint AND
separated by more than the fabrication envelope, which is +/- 1.38 % at a 100 nm
tolerance.  A certified enclosure narrower than that envelope cannot decide
between two chips coming off the same mask.

Outputs: patterns_data.json and figures/fig_patterns.png
"""

import json
import time

import numpy as np

import device as dv
import figstyle as fs
from certified_core import (capacitance_bracket, capacitance_upper, Bracket,
                            DIELECTRIC, HOT, GND)
from geometry import planar_pattern_3d

PATTERNS = [("comb", "P1 interdigitated comb"),
            ("rings", "P2 concentric rings"),
            ("checker", "P3 patch array"),
            ("spiral", "P4 double spiral")]

SPAN = 80e-6            # footprint of the compared cell
LAM = 20e-6             # pattern pitch
H_FLUID = 20e-6
H_SUB = 4e-6
T_METAL = 1e-6          # see the note in the module docstring
MESHES = (1.0e-6, 0.8e-6, 0.625e-6, 0.5e-6)

FAB_ENVELOPE = 0.0138   # +/- 1.38 %, from study_fabrication.py at 100 nm


def sensitivity(kind, h):
    """Certified enclosure of S = C_bound / C_blank - 1 for one pattern.

    The two enclosures are combined the way interval arithmetic requires,
        S in [L_lo / B_hi - 1,  L_hi / B_lo - 1],
    and not by differencing midpoints, which would throw the guarantee away.
    """
    kw = dict(kind=kind, lam=LAM, span=SPAN, h_fluid=H_FLUID, t_metal=T_METAL,
              h_sub=H_SUB, eps_fluid=dv.EPS_PBS, eps_sub=dv.EPS_GLASS)
    blank = capacitance_bracket(*planar_pattern_3d(h, **kw).as_args())
    bound = capacitance_bracket(*planar_pattern_3d(
        h, bound_layer=dv.CAPTURE_LAYER, **kw).as_args())
    return blank, bound, Bracket(bound.lo / blank.hi - 1.0,
                                 bound.hi / blank.lo - 1.0)


def main():
    fs.use_paper_style()
    import matplotlib.pyplot as plt

    out = {"span_m": SPAN, "lam_m": LAM, "t_metal_m": T_METAL,
           "h_fluid_m": H_FLUID, "fab_envelope": FAB_ENVELOPE, "meshes": MESHES}

    # ---------------------------------------------------------------- solve
    res = {}
    for h in MESHES:
        n = int(round(SPAN / h))
        print(f"mesh h = {h*1e6:.2f} um  ({n} x {n} x "
              f"{int(round((H_SUB+H_FLUID)/h))} cells)")
        for kind, label in PATTERNS:
            t0 = time.time()
            blank, bound, S = sensitivity(kind, h)
            res[(kind, h)] = (blank, bound, S)
            print(f"  {label:26s} blank +/-{blank.relwidth:6.3f} %   "
                  f"S = [{100*S.lo:+7.3f}, {100*S.hi:+7.3f}] %   "
                  f"width {100*(S.hi-S.lo)/2:5.3f} %   {time.time()-t0:5.1f} s")

    # ---------------------------------------------------------------- convergence
    print("\nconvergence on the two FINEST meshes (the ones the ranking uses)")
    conv = {}
    for kind, label in PATTERNS:
        a = res[(kind, MESHES[-2])][2]
        b = res[(kind, MESHES[-1])][2]
        ok = not (a.hi < b.lo or b.hi < a.lo)          # they must overlap
        widths = [100 * (res[(kind, m)][2].hi - res[(kind, m)][2].lo) / 2
                  for m in MESHES]
        conv[kind] = dict(compatible=bool(ok),
                          widths_pc=[float(w) for w in widths],
                          shrinking=bool(all(np.diff(widths) < 1e-9)))
        print(f"  {label:26s} [{100*a.lo:+7.3f},{100*a.hi:+7.3f}] -> "
              f"[{100*b.lo:+7.3f},{100*b.hi:+7.3f}]  "
              f"{'compatible' if ok else 'INCOMPATIBLE'}   "
              f"half-widths {' > '.join(f'{w:.2f}' for w in widths)} %")
    out["convergence"] = conv

    # The absolute values still drift with the mesh, so the honest claim is about
    # the ORDER, not the numbers: check that the ranking is the same on every
    # mesh before asserting it.
    orders = []
    for m in MESHES:
        Sm = {k: res[(k, m)][2] for k, _ in PATTERNS}
        orders.append(tuple(sorted(Sm, key=lambda k: -abs(0.5 * (Sm[k].lo +
                                                                 Sm[k].hi)))))
    stable = len(set(orders)) == 1
    out["order_stable_across_meshes"] = bool(stable)
    out["orders_per_mesh"] = [list(o) for o in orders]
    print(f"  ranking identical on all {len(MESHES)} meshes: "
          f"{'YES' if stable else 'NO'}")
    if not stable:
        for m, o in zip(MESHES, orders):
            print(f"    h = {m*1e6:.3f} um : {' > '.join(o)}")

    # ---------------------------------------------------------------- ranking
    h = MESHES[-1]
    S = {k: res[(k, h)][2] for k, _ in PATTERNS}
    out["sensitivity"] = {k: [S[k].lo, S[k].hi] for k in S}
    out["blank"] = {k: [res[(k, h)][0].lo, res[(k, h)][0].hi] for k in S}

    order = sorted(S, key=lambda k: -abs(0.5 * (S[k].lo + S[k].hi)))
    print("\ncertified ranking, most sensitive first (finest mesh)")
    for k in order:
        lab = dict(PATTERNS)[k]
        print(f"  {lab:26s} S = [{100*S[k].lo:+7.3f}, {100*S[k].hi:+7.3f}] %")

    print("\npairwise verdicts")
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            Sa, Sb = S[a], S[b]
            disjoint = Sa.hi < Sb.lo or Sb.hi < Sa.lo
            # separation between the two enclosures, relative to the mean level
            level = abs(0.5 * (Sa.lo + Sa.hi + Sb.lo + Sb.hi) / 2)
            sep = (max(Sb.lo - Sa.hi, Sa.lo - Sb.hi)) / max(level, 1e-12)
            survives = disjoint and sep > 2 * FAB_ENVELOPE
            pairs.append(dict(a=a, b=b, disjoint=bool(disjoint),
                              separation=float(sep), survives=bool(survives)))
            print(f"  {a:8s} vs {b:8s}  "
                  f"{'disjoint' if disjoint else 'TIE     '}  "
                  f"separation {100*sep:6.3f} %  "
                  f"vs fabrication {100*2*FAB_ENVELOPE:.2f} %  "
                  f"{'SURVIVES' if survives else 'not certified on a real chip'}")
    out["pairs"] = pairs

    n_disjoint = sum(p["disjoint"] for p in pairs)
    n_survive = sum(p["survives"] for p in pairs)
    print(f"\n  {n_disjoint}/{len(pairs)} pairs separated by the numerical "
          f"certificate alone")
    print(f"  {n_survive}/{len(pairs)} pairs still separated once a 100 nm "
          f"fabrication tolerance is admitted")
    out["n_pairs"] = len(pairs)
    out["n_disjoint"] = n_disjoint
    out["n_survive"] = n_survive

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(fs.COL2, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.55,
                          wspace=0.35)

    # (a) the four masks, top view
    from geometry import _pattern_mask
    hm = 0.25e-6
    nm_ = int(round(SPAN / hm))
    for j, (kind, label) in enumerate(PATTERNS):
        ax = fig.add_subplot(gs[0, j])
        hot, gnd = _pattern_mask(kind, nm_, nm_, hm, LAM)
        img = np.zeros((nm_, nm_, 3)) + 0.93
        img[hot.T] = [0.76, 0.16, 0.05]
        img[gnd.T] = [0.20, 0.20, 0.22]
        ax.imshow(img, origin="lower",
                  extent=[0, SPAN * 1e6, 0, SPAN * 1e6], interpolation="nearest")
        ax.set_title(label, fontsize=8, pad=3)
        ax.set_xticks([0, 40, 80])
        ax.set_yticks([0, 40, 80])
        if j == 0:
            ax.set_ylabel("z  ($\\mu$m)")
        ax.set_xlabel("x  ($\\mu$m)")
    fs.panel_label(fig.axes[0], "(a)", dy=1.30)

    # (b) certified sensitivity with the fabrication envelope
    ax = fig.add_subplot(gs[1, :2])
    y = np.arange(len(order))
    mid = np.array([50 * (S[k].lo + S[k].hi) for k in order])       # in %
    err = np.array([50 * (S[k].hi - S[k].lo) for k in order])
    ax.barh(y, mid, xerr=err, color=fs.C_BAND, edgecolor=fs.C_UP,
            error_kw=dict(ecolor=fs.C_LO, lw=1.4, capsize=3), height=0.62,
            label="certified enclosure")
    for i, k in enumerate(order):
        band = abs(mid[i]) * FAB_ENVELOPE
        ax.barh(y[i], 0, left=mid[i], height=0.86)
        ax.add_patch(plt.Rectangle((mid[i] - band, y[i] - 0.43), 2 * band, 0.86,
                                   color="0.55", alpha=0.30, lw=0,
                                   zorder=0))
    ax.set_yticks(y)
    ax.set_yticklabels([dict(PATTERNS)[k].split(" ", 1)[0] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("certified sensitivity  $S$  (%)")
    ax.plot([], [], color="0.55", alpha=0.5, lw=8,
            label="$\\pm$1.38 % fabrication envelope")
    fs.legend(ax, loc="lower right")
    # Show the blank next to each bar.  It rules out the obvious explanation:
    # the patch array does NOT win by having a small baseline, it has the LARGEST
    # blank of the four (300 fF against 112 fF for the comb).  What it has is
    # edge length.  Its patches alternate polarity along x AND z, so it carries
    # far more electrode edge per unit floor area than a comb, whose edges run in
    # one direction only, and the fringing field at those edges is what reaches
    # the bound layer.  This is the first submission's own argument, and solved
    # in genuine 3-D it points at the patch array rather than at the comb.
    for i, k in enumerate(order):
        b = res[(k, h)][0]
        ax.annotate(f"$C_0$ = {0.5*(b.lo+b.hi)*1e15:.1f} fF",
                    xy=(0, y[i]), xytext=(-4, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7.0, color="0.35")

    # (c) potential on a vertical cut, proving the solve is 3-D
    ax = fig.add_subplot(gs[1, 2:])
    cell = planar_pattern_3d(MESHES[0], kind="comb", lam=LAM, span=SPAN,
                            h_fluid=H_FLUID, t_metal=T_METAL, h_sub=H_SUB,
                            eps_fluid=dv.EPS_PBS, eps_sub=dv.EPS_GLASS)
    eps, elec, hh, depth = cell.as_args()
    _, u = capacitance_upper(eps, elec, hh, depth, return_field=True)
    kz = u.shape[2] // 2
    cut = u[:, :, kz]
    ny, nx = cut.shape
    ext = [0, nx * MESHES[0] * 1e6, 0, ny * MESHES[0] * 1e6]
    im = ax.imshow(cut, origin="lower", extent=ext, aspect="auto",
                   cmap="RdBu_r", vmin=0, vmax=1)
    ax.contour(np.linspace(ext[0], ext[1], nx), np.linspace(ext[2], ext[3], ny),
               cut, levels=11, colors="k", linewidths=0.4, alpha=0.55)
    ax.axhline(H_SUB * 1e6, color="k", lw=0.9, ls=":")
    ax.set_xlabel("x  ($\\mu$m)")
    ax.set_ylabel("height  ($\\mu$m)")
    ax.set_title("potential on a vertical cut, comb (3-D solve)",
                 fontsize=8, pad=3)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("$\\phi / V$", fontsize=8)
    fs.panel_label(ax, "(c)", dy=1.16)
    fs.panel_label(fig.axes[4], "(b)", dy=1.16)

    fs.save(fig, "figures/fig_patterns.png")

    with open("patterns_data.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote patterns_data.json")


if __name__ == "__main__":
    main()
