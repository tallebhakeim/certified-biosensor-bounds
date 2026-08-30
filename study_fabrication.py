"""
study_fabrication.py -- certified envelope of the FABRICATION tolerances.

Answers Reviewer 1 comment 3 and Reviewer 2 comment 4: the certified interval
must contain the geometric uncertainty of a real chip (line-edge roughness,
critical-dimension tolerance, metallisation thickness), not only the
discretisation error.

The argument is a monotonicity, so it needs no sampling and stays guaranteed.

    Conductor monotonicity.  Let K be the driven conductor and Omega the
    dielectric.  The Dirichlet principle reads
        C(K) = min { int eps |grad psi|^2 ,  psi = 1 on K, 0 on the ground }.
    If K grows to K' the admissible set shrinks (psi is now forced to 1 on a
    larger set), and any psi admissible for K' extends by the constant 1 into
    K' \\ K with no extra energy, hence is admissible for K with the same
    energy.  Therefore
        K subset K'   =>   C(K) <= C(K').
    This is the classical monotonicity of the condenser capacity.

    Consequence.  Let the manufactured electrode differ from the mask by at
    most delta anywhere (edge position and metal thickness).  Then it contains
    the eroded electrode K- and is contained in the dilated electrode K+, so
        C_lo(K-)  <=  C_true(manufactured)  <=  C_up(K+)
    for EVERY admissible fabrication defect, deterministic or random,
    correlated or not.  Combining the geometric monotonicity with the numerical
    bracket gives one guaranteed enclosure.

Line-edge roughness varies along the finger (the out-of-plane direction z).
Each z-slice is a 2-D problem whose capacitance lies between C(K-) and C(K+);
the device capacitance is the integral over z of the slice capacitances, so the
same two numbers bound the total.  The 2-D treatment is therefore not an
approximation of the roughness, it is a bound on it.

Panel (c) needs more care.  A ranking compares SENSITIVITIES, and the blank and
the loaded capacitance of one chip share the same manufactured geometry: taking
their worst cases independently is the classical interval-dependency blow-up
and gives a useless interval.  `device.certified_sensitivity` removes it by
subdividing the tolerance box and using the monotone endpoints on each sub-box;
the union over the partition is guaranteed for any partition and converges to
the exact range as it is refined.

Outputs: fabrication_data.json and figures/fig_fabrication.png
"""

import json
import numpy as np

import device as dv
from certified_core import Bracket
import figstyle as fs

# The mesh must RESOLVE the tolerance being certified.  The eroded and dilated
# electrodes are snapped outward (see geometry.ide_quarter_cell, snap="in"/"out"),
# so a tolerance finer than h would be rounded up to a full cell and the
# envelope, while still guaranteed, would report the mesh step instead of the
# tolerance.  Every tolerance below is therefore an exact multiple of the mesh.
H = 50e-9                # panels (a) and (b)
H_RANK = 100e-9          # panel (c): 4 solves per sub-box, so a coarser mesh
D_THICK = 50e-9          # metal thickness tolerance, +/- 25 % of 200 nm
N_SUB = 4                # partition of the tolerance box in panel (c)

TOL = np.array([0.0, 50.0, 100.0, 200.0, 400.0, 800.0]) * 1e-9
TOL_RANK = np.array([0.0, 100.0, 200.0, 400.0, 800.0]) * 1e-9


def certified_envelope(delta_edge, delta_t, h=H, **kw):
    """Guaranteed enclosure over every electrode within +/- delta of the mask.

    Eroded electrode (smallest conductor) -> lower bound.
    Dilated electrode (largest conductor) -> upper bound.

    The snapping is what makes this a bound and not an estimate: "in" rounds the
    eroded conductor DOWN to a grid electrode it contains, "out" rounds the
    dilated one UP to a grid electrode containing it.  At delta = 0 there is no
    tolerance to cover and the nominal (nearest) geometry is used, so panel (a)
    starts from the purely numerical bracket.
    """
    if delta_edge == 0.0 and delta_t == 0.0:
        return dv.cell_capacitance(h, 0.0, dv.T_METAL, **kw)
    lo = dv.cell_capacitance(h, -delta_edge, dv.T_METAL - delta_t,
                             snap="in", **kw).lo
    hi = dv.cell_capacitance(h, +delta_edge, dv.T_METAL + delta_t,
                             snap="out", **kw).hi
    return Bracket(lo, hi)


def main():
    fs.use_paper_style()
    import matplotlib.pyplot as plt

    out = {"device": {"pitch_m": dv.LAM, "eta": dv.ETA,
                      "t_metal_m": dv.T_METAL, "h_channel_m": dv.H_CHANNEL,
                      "mesh_m": H}}

    # ---------------------------------------------------------------- 1
    # how the certified enclosure widens with the fabrication tolerance
    # ----------------------------------------------------------------
    deltas = TOL
    env_lo, env_hi = [], []
    for d in deltas:
        e = certified_envelope(d, D_THICK if d > 0 else 0.0)
        env_lo.append(e.lo)
        env_hi.append(e.hi)
    env_lo = np.array(env_lo)
    env_hi = np.array(env_hi)
    nominal = 0.5 * (env_lo[0] + env_hi[0])
    out["tolerance_m"] = deltas.tolist()
    out["envelope_lo_F_per_m"] = env_lo.tolist()
    out["envelope_hi_F_per_m"] = env_hi.tolist()
    out["nominal_F_per_m"] = float(nominal)
    out["numerical_halfwidth_pc"] = float(
        100 * 0.5 * (env_hi[0] - env_lo[0]) / nominal)

    print("certified envelope of the blank vs fabrication tolerance")
    for d, a, b in zip(deltas, env_lo, env_hi):
        print(f"  delta = {d*1e9:6.0f} nm   [{a*1e12:9.4f}, {b*1e12:9.4f}] pF/m"
              f"   +/-{100*0.5*(b-a)/(0.5*(a+b)):6.3f} %")

    # ---------------------------------------------------------------- 2
    # a Monte-Carlo over rough edges must fall inside the envelope
    # ----------------------------------------------------------------
    shifts = np.linspace(-400e-9, 400e-9, 17)
    Cw = np.array([dv.cell_capacitance(H, s).mid for s in shifts])
    rng = np.random.default_rng(20260728)
    delta_mc = 200e-9
    n_slice, n_real = 256, 400
    corr = 20                                   # correlation length, in slices
    mc = []
    for _ in range(n_real):
        w = rng.normal(size=n_slice + 4 * corr)
        k = np.exp(-np.arange(4 * corr) / corr)
        k /= np.linalg.norm(k)
        prof = np.convolve(w, k, mode="valid")[:n_slice]
        prof = delta_mc * prof / max(np.abs(prof).max(), 1e-12)
        mc.append(np.interp(prof, shifts, Cw).mean())
    mc = np.array(mc)
    e1 = certified_envelope(delta_mc, D_THICK)
    out["mc_delta_m"] = delta_mc
    out["mc_min"], out["mc_max"] = float(mc.min()), float(mc.max())
    out["mc_envelope"] = [e1.lo, e1.hi]
    contained = bool(mc.min() >= e1.lo and mc.max() <= e1.hi)
    out["mc_contained"] = contained
    print(f"\nMonte-Carlo over {n_real} correlated rough profiles "
          f"(+/-{delta_mc*1e9:.0f} nm): [{mc.min()*1e12:.4f}, "
          f"{mc.max()*1e12:.4f}] pF/m")
    print(f"certified envelope: [{e1.lo*1e12:.4f}, {e1.hi*1e12:.4f}] pF/m"
          f"   containment: {'YES' if contained else 'NO'}")

    # ---------------------------------------------------------------- 3
    # does a certified ranking survive the fabrication tolerance?
    # ----------------------------------------------------------------
    print("\ncertified sensitivity of two candidate designs (capture assay)")
    tol = TOL_RANK
    ranks = []
    for d in tol:
        n = N_SUB if d > 0 else 0
        Sa = dv.certified_sensitivity(d, D_THICK if d > 0 else 0.0, n_sub=n,
                                      eta=0.35, h=H_RANK)
        Sb = dv.certified_sensitivity(d, D_THICK if d > 0 else 0.0, n_sub=n,
                                      eta=0.60, h=H_RANK)
        ok = Sa.clears(Sb)
        ranks.append((d, Sa, Sb, ok))
        print(f"  tol={d*1e9:5.0f} nm  S(eta=0.35)=[{100*Sa.lo:+7.3f},"
              f"{100*Sa.hi:+7.3f}] %  S(eta=0.60)=[{100*Sb.lo:+7.3f},"
              f"{100*Sb.hi:+7.3f}] %  {'CERTIFIED' if ok else 'tie'}")
    keep = [d for d, _, _, ok in ranks if ok]
    out["ranking_survives_up_to_m"] = float(max(keep)) if keep else 0.0
    out["ranking"] = [{"tol_m": float(d), "S_eta035": [s.lo, s.hi],
                       "S_eta060": [t.lo, t.hi], "certified": ok}
                      for d, s, t, ok in ranks]

    # convergence of the subdivision: the enclosure must tighten and stabilise
    print("\nsubdivision convergence at delta = 200 nm (eta = 0.35)")
    conv = []
    for n in (1, 2, 4, 8, 16):
        S = dv.certified_sensitivity(200e-9, D_THICK, n_sub=n, eta=0.35,
                                     h=H_RANK)
        conv.append((n, S.lo, S.hi))
        print(f"  n_sub={n:3d}   S = [{100*S.lo:+7.3f}, {100*S.hi:+7.3f}] %")
    out["subdivision_convergence"] = [{"n_sub": n, "lo": a, "hi": b}
                                      for n, a, b in conv]

    # ---------------------------------------------------------------- figure
    fig, axes = fs.row(3, height=3.05)

    ax = axes[0]
    ax.fill_between(deltas * 1e9, env_lo * 1e12, env_hi * 1e12,
                    color=fs.C_BAND, alpha=0.55, lw=0,
                    label="certified envelope")
    ax.plot(deltas * 1e9, env_hi * 1e12, color=fs.C_UP, lw=1.4)
    ax.plot(deltas * 1e9, env_lo * 1e12, color=fs.C_LO, lw=1.4)
    ax.axhline(nominal * 1e12, color="k", ls=":", lw=1.2,
               label="mask geometry")
    ax.annotate(f"numerical only\n$\\pm${out['numerical_halfwidth_pc']:.3f} %",
                xy=(0, nominal * 1e12), xytext=(14, -26),
                textcoords="offset points", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax.set_xlabel("fabrication tolerance $\\delta$  (nm)")
    ax.set_ylabel("$C$ per period  (pF/m)")
    fs.legend(ax, loc="upper left", headroom=0.30)

    ax = axes[1]
    ax.hist(mc * 1e12, bins=28, color=fs.C_MC, alpha=0.75,
            label="Monte-Carlo,\nrough edges")
    ax.axvspan(e1.lo * 1e12, e1.hi * 1e12, color=fs.C_BAND, alpha=0.35,
               label="certified\nenvelope")
    ax.axvline(e1.lo * 1e12, color=fs.C_LO, lw=1.8)
    ax.axvline(e1.hi * 1e12, color=fs.C_UP, lw=1.8)
    pad = 0.12 * (e1.hi - e1.lo) * 1e12
    ax.set_xlim(e1.lo * 1e12 - pad, e1.hi * 1e12 + pad)
    ax.set_xlabel("$C$ per period  (pF/m)")
    ax.set_ylabel("realisations")
    fs.legend(ax, loc="upper left", headroom=0.42)

    ax = axes[2]
    t_nm = np.array([r[0] for r in ranks]) * 1e9
    lo1 = np.array([r[1].lo for r in ranks]) * 100
    hi1 = np.array([r[1].hi for r in ranks]) * 100
    lo2 = np.array([r[2].lo for r in ranks]) * 100
    hi2 = np.array([r[2].hi for r in ranks]) * 100
    ax.fill_between(t_nm, lo1, hi1, color=fs.PALETTE[0], alpha=0.55, lw=0,
                    label="$\\eta = 0.35$")
    ax.fill_between(t_nm, lo2, hi2, color=fs.PALETTE[1], alpha=0.55, lw=0,
                    label="$\\eta = 0.60$")
    ax.set_xlabel("fabrication tolerance $\\delta$  (nm)")
    ax.set_ylabel("certified sensitivity  (%)")
    fs.legend(ax, loc="lower left", headroom=0.34)
    lo, hi = ax.get_ylim()
    ax.annotate("designs separable\nonly at $\\delta = 0$", xy=(0, hi1[0]),
                xytext=(0.30, 0.93), textcoords="axes fraction", fontsize=7.5,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

    fs.save(fig, "figures/fig_fabrication.png")

    with open("fabrication_data.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote fabrication_data.json")


if __name__ == "__main__":
    main()
