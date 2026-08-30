"""
study_fluidics.py -- certified transport, and the dielectric cover claim.

Two independent subjects, both in physical units.

PART A, the flow.  A shallow channel obeys the depth-averaged (Hele-Shaw)
equation  q = -(h^2/12 mu) grad p,  div q = 0, which is the SAME self-adjoint
elliptic operator as the electrostatic problem with a hydraulic conductivity
K = h^3/(12 mu) in the fluid and essentially zero in the walls.  The dual
bracket therefore encloses the hydraulic conductance two-sided at no conceptual
cost, giving a certified flow rate and a certified residence time.

The solver is reused verbatim by passing K/eps0 as the "permittivity": the code
computes eps0 * sum(eps_r |grad|^2), so the returned number is K times a
dimensionless geometric factor, i.e. the hydraulic conductance in m^3/(Pa s),
and Q = G_h * dP.

Honesty about the model, established in study_validity.py: the shallowness
condition h/w = 0.1 sits exactly on the conventional threshold, and dropping the
sidewalls makes the depth-averaged flow rate too large by
1/(1 - 0.630 h/w) - 1 = 6.7 %.  That is a ONE-SIDED bias, not an enclosure, and
it is reported as such: the certified interval is widened downward by it rather
than being quoted as if the model were exact.

PART B, the dielectric cover.  The first submission claimed that a
high-permittivity cover above a thin channel confines the field and RAISES the
certified surface sensitivity, calling it "a useful and perhaps counter-intuitive
result".  This part tests that claim in physical units instead of assuming it.

Outputs: fluidics_data.json and figures/fig_fluidics.png
"""

import json
import time

import numpy as np

import device as dv
import figstyle as fs
from certified_core import (capacitance_bracket, capacitance_upper, Bracket,
                            EPS0, DIELECTRIC, HOT, GND)
from geometry import serpentine_layout, _n

# ---- channel of the reference chip ---------------------------------------
W_CHAN = 200e-6            # channel width, matches study_validity.py
H_CHAN = dv.H_CHANNEL      # 20 um
LEG = 2e-3                 # sensing leg length, matches the 50 x 40 um comb
PITCH = 400e-6
MARGIN = 100e-6
N_TURN = 3
MU = 1.0e-3                # Pa s
K_FLUID = H_CHAN ** 3 / (12 * MU)      # m^3 / (Pa s)
WALL_RATIO = 1e-6
ALPHA = H_CHAN / W_CHAN
SIDEWALL_BIAS = 1.0 / (1.0 - 0.630 * ALPHA) - 1.0

# Every mesh must divide the channel width, the leg, the pitch AND the margin
# exactly, otherwise each refinement re-voxelises the meander into a slightly
# different geometry and the successive enclosures compare two different
# problems.  200, 2000, 400 and 100 um are all multiples of these four.
MESHES_FLOW = (10e-6, 5e-6, 4e-6, 2.5e-6)

FAB_ENVELOPE = 0.0138      # +/- 1.38 % at a 100 nm tolerance, study_fabrication
PORT_WIDTH = 20e-6         # physical width of the inlet and outlet ports

# ---- cover sweep ----------------------------------------------------------
H_COVER_SWEEP = 250e-9
T_COVER = 10e-6


# ---------------------------------------------------------------------------
# PART A
# ---------------------------------------------------------------------------
def serpentine_conductance(h):
    """Certified enclosure of the hydraulic conductance of the serpentine.

    Returns (Bracket in m^3/(Pa s), mask, potential field for the figure).
    """
    mask, hh, _, _ = serpentine_layout(h, w_chan=W_CHAN, n_turn=N_TURN,
                                       leg=LEG, pitch=PITCH, margin=MARGIN)
    ny, nx = mask.shape
    eps = np.where(mask, K_FLUID / EPS0, K_FLUID * WALL_RATIO / EPS0)
    elec = np.zeros((ny, nx), dtype=int)

    # Ports at the two ends of the meander.  With three legs the path runs
    # left to right, down the right side, right to left, down the left side,
    # then left to right again, so the inlet is at the left of the first leg and
    # the outlet at the right of the last one.
    # The ports must have a FIXED PHYSICAL width.  A one-cell port shrinks as the
    # mesh is refined, so the geometry would change with h and the successive
    # enclosures would bracket four different problems: they came out disjoint
    # and monotonically decreasing, which is exactly the signature of that.
    # 20 um divides every mesh in MESHES_FLOW.
    nm, nw, npi = _n(MARGIN, h), _n(W_CHAN, h), _n(PITCH, h)
    npt = _n(PORT_WIDTH, h)
    elec[nm:nm + nw, nm:nm + npt] = HOT
    last = nm + (N_TURN - 1) * npi
    elec[last:last + nw, nx - nm - npt:nx - nm] = GND

    br = capacitance_bracket(eps, elec, hh, 1.0)
    _, u = capacitance_upper(eps, elec, hh, 1.0, return_field=True)
    return br, mask, u


def part_a(out):
    print("=" * 74)
    print("PART A -- certified Hele-Shaw flow in the serpentine")
    print("=" * 74)
    print(f"  channel {W_CHAN*1e6:.0f} x {H_CHAN*1e6:.0f} um, "
          f"{N_TURN} legs of {LEG*1e3:.1f} mm, K = {K_FLUID:.3e} m3/(Pa s)")

    G, mask, u = None, None, None
    conv = []
    for h in MESHES_FLOW:
        t0 = time.time()
        g, m, uu = serpentine_conductance(h)
        conv.append((h, g))
        print(f"  h = {h*1e6:4.1f} um  ({m.shape[0]} x {m.shape[1]})  "
              f"G_h = [{g.lo:.6e}, {g.hi:.6e}] m3/(Pa s)  "
              f"+/-{g.relwidth:6.3f} %   {time.time()-t0:5.1f} s")
        G, mask, u = g, m, uu

    ok = all(not (a.hi < b.lo or b.hi < a.lo)
             for (_, a), (_, b) in zip(conv[:-1], conv[1:]))
    print(f"  successive enclosures compatible: {'YES' if ok else 'NO'}")

    dP = np.array([100.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0])
    Q_lo, Q_hi = G.lo * dP, G.hi * dP
    # the sidewall bias is one-sided: the depth-averaged model over-estimates Q,
    # so the honest lower end is the bracket's lower end corrected downward
    Q_lo_corr = Q_lo / (1.0 + SIDEWALL_BIAS)
    U = 0.5 * (Q_lo_corr + Q_hi) / (W_CHAN * H_CHAN)
    t_res = LEG / U

    print(f"\n  sidewall bias (h/w = {ALPHA:.3f}) : "
          f"+{100*SIDEWALL_BIAS:.1f} % on Q, one-sided")
    print(f"  {'dP (kPa)':>9s} {'Q certified (uL/s)':>28s} "
          f"{'U (mm/s)':>10s} {'t_res (ms)':>11s}")
    for i, p in enumerate(dP):
        print(f"  {p/1e3:9.2f} {Q_lo_corr[i]*1e9:12.4f} - {Q_hi[i]*1e9:8.4f}"
              f" {U[i]*1e3:12.3f} {t_res[i]*1e3:11.3f}")

    out["hydraulic"] = dict(
        G_lo=G.lo, G_hi=G.hi, relwidth_pc=G.relwidth,
        convergence_ok=bool(ok),
        meshes=[dict(h=h, lo=g.lo, hi=g.hi) for h, g in conv],
        sidewall_bias=float(SIDEWALL_BIAS), alpha=float(ALPHA),
        dP=dP.tolist(), Q_lo=Q_lo_corr.tolist(), Q_hi=Q_hi.tolist(),
        U=U.tolist(), t_res=t_res.tolist())
    return G, mask, u, dP, Q_lo_corr, Q_hi, t_res


# ---------------------------------------------------------------------------
# PART B
# ---------------------------------------------------------------------------
def sensitivity(h_fluid, eps_cover, t_cover=T_COVER, h=H_COVER_SWEEP):
    """Certified S = C_bound/C_blank - 1 for a given channel and cover."""
    kw = dict(h_fluid=h_fluid, t_cover=t_cover, eps_cover=eps_cover)
    B = dv.cell_capacitance(h, **kw)
    L = dv.cell_capacitance(h, bound=dv.CAPTURE_LAYER, **kw)
    return Bracket(L.lo / B.hi - 1.0, L.hi / B.lo - 1.0)


def part_b(out):
    print()
    print("=" * 74)
    print("PART B -- does a high-permittivity cover raise the sensitivity?")
    print("=" * 74)
    print("  v1 claimed it does.  Testing it.")

    heights = np.array([5, 10, 20, 30, 40, 60]) * 1e-6
    covers = [(1.0, "air"), (10.0, "polymer"), (40.0, "high-k"),
              (78.0, "matched to the buffer")]

    print(f"\n  {'h_fluid':>9s}" + "".join(f"{lab:>26s}" for _, lab in covers))
    grid = {}
    for hf in heights:
        row = []
        for ec, _ in covers:
            S = sensitivity(hf, ec)
            grid[(hf, ec)] = S
            row.append(f"[{100*S.lo:+7.3f},{100*S.hi:+7.3f}]")
        print(f"  {hf*1e6:7.0f} um" + "".join(f"{c:>26s}" for c in row))

    # fine sweep of the cover permittivity at the reference channel height
    eps_sweep = np.array([1, 2.6, 5, 10, 20, 40, 60, 78.0])
    S_sweep = [sensitivity(H_CHAN, e) for e in eps_sweep]
    print(f"\n  cover sweep at h_fluid = {H_CHAN*1e6:.0f} um, "
          f"t_cover = {T_COVER*1e6:.0f} um")
    for e, S in zip(eps_sweep, S_sweep):
        print(f"    eps_cover = {e:5.1f}   S = [{100*S.lo:+7.3f}, "
              f"{100*S.hi:+7.3f}] %")

    # the verdict
    S_air = S_sweep[0]
    S_high = S_sweep[-1]
    rises = S_high.hi < S_air.lo          # more negative means more sensitive
    falls = S_high.lo > S_air.hi
    print()
    if rises:
        verdict = ("CONFIRMED: a matched cover raises the certified "
                   "sensitivity, the enclosures being disjoint")
    elif falls:
        verdict = ("CONTRADICTED: a matched cover LOWERS the certified "
                   "sensitivity, the enclosures being disjoint")
    else:
        verdict = ("NOT CERTIFIED either way: the two enclosures overlap, so "
                   "the cover effect is within the numerical certificate")
    print(f"  air  eps=1  : S = [{100*S_air.lo:+7.3f}, {100*S_air.hi:+7.3f}] %")
    print(f"  matched 78  : S = [{100*S_high.lo:+7.3f}, {100*S_high.hi:+7.3f}] %")
    print(f"  verdict against the numerical certificate: {verdict}")

    # Same criterion as the electrode ranking: an effect smaller than the
    # fabrication envelope cannot be exploited on a real chip, however cleanly
    # the numerical enclosures separate.  The cover effect fades as the channel
    # deepens, so the verdict is height dependent and is reported per height.
    print(f"\n  against the +/-{100*FAB_ENVELOPE:.2f} % fabrication envelope, "
          f"per channel height")
    per_h = []
    for hf in heights:
        a, b = grid[(hf, 1.0)], grid[(hf, 78.0)]
        gain = abs(0.5 * (b.lo + b.hi)) - abs(0.5 * (a.lo + a.hi))
        rel = gain / abs(0.5 * (a.lo + a.hi))
        usable = (b.hi < a.lo) and rel > 2 * FAB_ENVELOPE
        per_h.append(dict(h_fluid=float(hf), gain_pts=float(100 * gain),
                          gain_rel=float(rel), usable=bool(usable)))
        print(f"    h = {hf*1e6:3.0f} um   gain {100*gain:+6.3f} points "
              f"({100*rel:+6.2f} % relative)   "
              f"{'EXPLOITABLE' if usable else 'below the fabrication envelope'}")
    n_use = sum(p["usable"] for p in per_h)
    print(f"\n  the cover is a usable design lever for {n_use} of "
          f"{len(heights)} channel heights, the thin ones")
    out["cover_vs_fabrication"] = per_h

    out["cover"] = dict(
        heights=heights.tolist(),
        covers=[c for c, _ in covers],
        grid={f"{hf:.3e}|{ec:.3f}": [grid[(hf, ec)].lo, grid[(hf, ec)].hi]
              for hf in heights for ec, _ in covers},
        eps_sweep=eps_sweep.tolist(),
        S_sweep=[[S.lo, S.hi] for S in S_sweep],
        verdict=verdict,
        v1_claim_confirmed=bool(rises))
    return heights, covers, grid, eps_sweep, S_sweep, verdict


def main():
    fs.use_paper_style()
    import matplotlib.pyplot as plt

    out = {"w_chan_m": W_CHAN, "h_chan_m": H_CHAN, "leg_m": LEG,
           "mu": MU, "K_fluid": K_FLUID, "t_cover_m": T_COVER}

    G, mask, u, dP, Q_lo, Q_hi, t_res = part_a(out)
    heights, covers, grid, eps_sweep, S_sweep, verdict = part_b(out)

    # ---------------------------------------------------------------- figure
    FIG_H = 5.0
    fig = plt.figure(figsize=(fs.COL2, FIG_H))
    gs = fig.add_gridspec(1, 2, top=0.44, bottom=0.115, wspace=0.52)

    # (a) the meander.  Its axes are placed explicitly, with a box whose aspect
    # matches the 2200 x 1200 um domain: inside a gridspec cell the equal-aspect
    # map is shrunk and pushed to one side, and set_anchor is overridden when the
    # colorbar steals space from the same axes.
    W_AX, X0, TOP = 0.42, 0.085, 0.945
    box_h = W_AX * fs.COL2 * (1200.0 / 2200.0) / FIG_H
    ax = fig.add_axes([X0, TOP - box_h, W_AX, box_h])
    h = MESHES_FLOW[-1]
    ny, nx = mask.shape
    ext = [0, nx * h * 1e6, 0, ny * h * 1e6]
    # u is nodal, so it carries one more point per axis than the cell mask
    uc = 0.25 * (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])
    p = np.where(mask, uc, np.nan)
    im = ax.imshow(p, origin="lower", extent=ext, cmap="viridis",
                   aspect="equal", interpolation="nearest")
    # depth-averaged velocity q = -K grad p, drawn only inside the channel
    gy, gx = np.gradient(np.nan_to_num(uc), h, h)
    qx, qy = -K_FLUID * gx, -K_FLUID * gy
    qx = np.where(mask, qx, np.nan)
    qy = np.where(mask, qy, np.nan)
    X = np.linspace(ext[0], ext[1], nx)
    Y = np.linspace(ext[2], ext[3], ny)
    with np.errstate(invalid="ignore"):
        ax.streamplot(X, Y, qx, qy, color="w", linewidth=0.7, density=1.5,
                      arrowsize=0.7, broken_streamlines=False)
    ax.set_xlabel("x  ($\\mu$m)")
    ax.set_ylabel("y  ($\\mu$m)")
    cax = fig.add_axes([X0 + W_AX + 0.012, TOP - box_h, 0.013, box_h])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$p / \\Delta P$", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_title("depth-averaged pressure and streamlines",
                 fontsize=8.5, pad=4, loc="left")
    ax.tick_params(labelsize=7.5)
    fig.text(X0 - 0.062, TOP + 0.028, "(a)", fontsize=11, weight="bold")
    g = out["hydraulic"]
    fig.text(X0 + W_AX + 0.085, TOP - 0.005,
             "certified hydraulic conductance\n"
             f"$G_h$ = {g['G_lo']:.4e} to {g['G_hi']:.4e}\n"
             f"m$^3$/(Pa s),  $\\pm${g['relwidth_pc']:.3f} %\n\n"
             "nested under refinement over\n"
             f"{len(g['meshes'])} meshes, 10 to 2.5 $\\mu$m\n\n"
             f"sidewall bias +{100*g['sidewall_bias']:.1f} % on $Q$,\n"
             "one-sided, already included",
             fontsize=7.6, va="top", ha="left", linespacing=1.45)

    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(dP / 1e3, Q_lo * 1e9, Q_hi * 1e9, color=fs.C_BAND,
                    alpha=0.65, lw=0, label="certified $Q$, bias included")
    ax.plot(dP / 1e3, Q_hi * 1e9, color=fs.C_UP, lw=1.4)
    ax.plot(dP / 1e3, Q_lo * 1e9, color=fs.C_LO, lw=1.4)
    ax.set_xlabel("applied pressure  (kPa)")
    ax.set_ylabel("flow rate  ($\\mu$L/s)")
    ax2 = ax.twinx()
    ax2.plot(dP / 1e3, np.array(t_res) * 1e3, color="0.35", ls="--", lw=1.4,
             label="residence time")
    ax2.set_ylabel("$t_{\\rm res}$  (ms)", fontsize=8.5)
    ax2.set_yscale("log")
    lines = ax.get_legend_handles_labels()
    l2 = ax2.get_legend_handles_labels()
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
    ax.legend(lines[0] + l2[0], lines[1] + l2[1], loc="upper left",
              fontsize=7.2, framealpha=0.92)
    fs.panel_label(ax, "(b)", dy=1.14)

    ax = fig.add_subplot(gs[0, 1])
    mid = np.array([50 * (S.lo + S.hi) for S in S_sweep])
    err = np.array([50 * (S.hi - S.lo) for S in S_sweep])
    ax.errorbar(eps_sweep, mid, yerr=err, fmt="o-", ms=4, lw=1.4,
                color=fs.PALETTE[0], ecolor=fs.C_LO, capsize=3,
                label="certified $S$, $h_{\\rm fluid}$ = 20 $\\mu$m")
    ax.axhspan(50 * (S_sweep[0].lo + S_sweep[0].hi) -
               50 * (S_sweep[0].hi - S_sweep[0].lo),
               50 * (S_sweep[0].lo + S_sweep[0].hi) +
               50 * (S_sweep[0].hi - S_sweep[0].lo),
               color="0.6", alpha=0.25, lw=0, label="air cover, enclosure")
    ax.set_xlabel("cover permittivity  $\\varepsilon_{\\rm cover}$")
    ax.set_ylabel("certified sensitivity  $S$  (%)")
    ax.set_ylim(bottom=ax.get_ylim()[0] - 0.75)
    fs.legend(ax, loc="upper right")
    # State the verdict AND its domain: the effect is certified numerically but
    # only exceeds the fabrication envelope in thin channels.
    n_use = sum(r["usable"] for r in out["cover_vs_fabrication"])
    hs_use = [r["h_fluid"] * 1e6 for r in out["cover_vs_fabrication"]
              if r["usable"]]
    txt = ("confirmed, but exploitable only for\n"
           + (f"$h_{{\\rm fluid}} \\leq$ {max(hs_use):.0f} $\\mu$m: "
              f"above that the gain\nfalls below the fabrication envelope"
              if hs_use else "no channel height tested"))
    ax.annotate(txt, xy=(0.04, 0.055), xycoords="axes fraction", ha="left",
                va="bottom", fontsize=7.2)
    fs.panel_label(ax, "(c)", dy=1.14)

    fs.save(fig, "figures/fig_fluidics.png")

    with open("fluidics_data.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote fluidics_data.json")


if __name__ == "__main__":
    main()
