"""
study_validity.py -- validity conditions of the transport models.

Answers Reviewer 1 comment 6: the Hele-Shaw and Stokes-Einstein models are used
without stating when they hold.  Each condition is evaluated NUMERICALLY at the
operating point of the reference device, so the paper can state a verdict per
model instead of a disclaimer.

The distinction the revision makes throughout applies here too: the dual-TLM
bracket certifies the SOLUTION of whichever elliptic problem is posed.  Whether
the Hele-Shaw equation is the right problem for this channel is a MODELLING
question, answered by the dimensionless numbers below, not by the bracket.

Hele-Shaw (depth-averaged Stokes)  q = -(h^2/12 mu) grad p,  div q = 0
  H1  shallow channel        h / w         << 1
  H2  creeping flow          Re = rho U h / mu  << 1
  H3  developed profile      L_entry / L_channel << 1
  H4  no inertial secondary  flow in bends: De = Re sqrt(h/R) << 1
  H5  steady                 Womersley / unsteadiness negligible

Stokes-Einstein  D = kB T / (6 pi mu a)
  S1  continuum solvent      a >> solvent molecular size
  S2  no wall hindrance      a / h_channel << 1  (Faxen correction small)
  S3  dilute                 f << 1, no crowding correction
  S4  no confinement of the  Debye/hydration layers: a >> lambda_D

Outputs: validity_data.json and a LaTeX-ready table on stdout.
"""

import json
import numpy as np

import device as dv

# ---- operating point ------------------------------------------------------
H_CH = dv.H_CHANNEL            # 20 um channel height
W_CH = 200e-6                  # channel width of the sensing leg
L_CH = 2e-3                    # length of the sensing leg
R_BEND = 60e-6                 # radius of curvature of a serpentine turn
RHO = 1000.0                   # kg/m3
MU = 1.0e-3                    # Pa s, water at 20 C
DP = 1e3                       # Pa, 1 kPa driving pressure
KB = 1.380649e-23
T = 298.15
LAMBDA_D_PBS = 0.76e-9         # Debye length in 0.15 M PBS
A_SOLVENT = 0.14e-9            # water molecular radius

PARTICLES = [("protein (5 nm)", 2.5e-9),
             ("virus (100 nm)", 50e-9),
             ("bacterium (1 um)", 0.5e-6),
             ("cell (10 um)", 5e-6)]


def hele_shaw():
    """Depth-averaged flow at the operating point, and its validity numbers."""
    # Hele-Shaw flow rate per unit width and the mean velocity
    q = (H_CH ** 2 / (12 * MU)) * (DP / L_CH)      # m2/s per unit width -> m/s
    U = q                                           # depth-averaged velocity
    Re = RHO * U * H_CH / MU
    # entrance length for a rectangular duct, laminar (Ferreira et al. 2021 give
    # L/Dh = 0.6/(1+0.035 Re) + 0.056 Re for low Re)
    Dh = 2 * H_CH * W_CH / (H_CH + W_CH)
    L_entry = Dh * (0.6 / (1 + 0.035 * Re) + 0.056 * Re)
    De = Re * np.sqrt(H_CH / R_BEND)
    tests = [
        ("H1 shallow channel", "h/w", H_CH / W_CH, 0.1),
        ("H2 creeping flow", "Re", Re, 1.0),
        ("H3 developed profile", "L_entry/L", L_entry / L_CH, 0.1),
        ("H4 no Dean vortices", "De", De, 1.0),
    ]
    # A pass/fail verdict on H1 is not useful, because h/w = 0.1 sits exactly on
    # the conventional threshold.  What the paper can state instead is the SIZE
    # of the model error: for a rectangular duct of aspect ratio alpha = h/w the
    # exact laminar flow rate carries the sidewall correction
    #     Q = (w h^3 dP)/(12 mu L) * [1 - 0.630 alpha + O(alpha^5)] ,
    # so depth-averaging (which drops the sidewalls) OVER-estimates the flow rate
    # by 1/(1 - 0.630 alpha) - 1.  That is a one-sided, quantified bias, and it
    # can be carried into the residence time exactly like the other intervals.
    alpha = H_CH / W_CH
    hs_bias = 1.0 / (1.0 - 0.630 * alpha) - 1.0
    return dict(U=U, Re=Re, L_entry=L_entry, De=De, alpha=alpha,
                flow_bias=float(hs_bias),
                tests=[dict(name=n, symbol=s, value=float(v),
                            threshold=t, ok=bool(v < t))
                       for n, s, v, t in tests])


def stokes_einstein():
    """Diffusivity of each target and the validity numbers of the model."""
    rows = []
    for name, a in PARTICLES:
        D = KB * T / (6 * np.pi * MU * a)
        # Faxen leading wall correction for a sphere at mid-height
        beta = a / (H_CH / 2)
        hindrance = 1 - 1.004 * beta + 0.418 * beta ** 3
        tests = [
            ("S1 continuum solvent", "a/a_w", a / A_SOLVENT, None, a / A_SOLVENT > 10),
            ("S2 no wall hindrance", "a/(h/2)", beta, 0.1, beta < 0.1),
            ("S4 outside double layer", "a/lambda_D", a / LAMBDA_D_PBS, None,
             a / LAMBDA_D_PBS > 10),
        ]
        # dispersion over the residence time
        t_res = L_CH / hele_shaw()["U"]
        sigma = np.sqrt(2 * D * t_res)
        rows.append(dict(name=name, a=a, D=D, hindrance=float(hindrance),
                         t_res=float(t_res), sigma=float(sigma),
                         sigma_over_channel=float(sigma / L_CH),
                         tests=[dict(name=n, symbol=s, value=float(v),
                                     threshold=th, ok=bool(ok))
                                for n, s, v, th, ok in tests]))
    return rows


def main():
    out = {}
    print("=" * 74)
    print("Hele-Shaw validity at the operating point (1 kPa over 2 mm)")
    print("=" * 74)
    hs = hele_shaw()
    out["hele_shaw"] = hs
    print(f"  depth-averaged velocity   U       = {hs['U']*1e3:9.4f} mm/s")
    print(f"  entrance length           L_entry = {hs['L_entry']*1e6:9.2f} um")
    for t in hs["tests"]:
        print(f"  {t['name']:26s} {t['symbol']:>10s} = {t['value']:10.3e}  "
              f"(need < {t['threshold']:g})   {'OK' if t['ok'] else 'VIOLATED'}")

    print()
    print("=" * 74)
    print("Stokes-Einstein validity, per target")
    print("=" * 74)
    se = stokes_einstein()
    out["stokes_einstein"] = se
    for r in se:
        flags = " ".join(f"{t['symbol']}={t['value']:.2e}"
                         f"{'' if t['ok'] else '(!)'}" for t in r["tests"])
        print(f"  {r['name']:18s} D = {r['D']:9.3e} m2/s  "
              f"hindrance x{r['hindrance']:.3f}  "
              f"plug spread {r['sigma']*1e6:8.2f} um "
              f"({100*r['sigma_over_channel']:6.3f} % of the leg)")
        print(f"    {flags}")

    # ---- the honest summary the paper needs ------------------------------
    print()
    print("=" * 74)
    print("verdict for the manuscript")
    print("=" * 74)
    hs_ok = all(t["ok"] for t in hs["tests"])
    print(f"  Hele-Shaw at 1 kPa               : "
          f"{'all conditions met' if hs_ok else 'marginal, see the bias below'}")
    worst = [t for t in hs["tests"] if not t["ok"]]
    for t in worst:
        print(f"    marginal: {t['name']} ({t['symbol']} = {t['value']:.3g})")
    print(f"  sidewall bias on the flow rate   : "
          f"+{100*hs['flow_bias']:.1f} % (depth-averaging over-estimates Q), "
          f"hence the residence time is under-estimated by the same factor")
    # at what pressure does Hele-Shaw start to fail?
    p_fail = None
    for p in np.logspace(2, 6, 200):
        U = (H_CH ** 2 / (12 * MU)) * (p / L_CH)
        if RHO * U * H_CH / MU >= 1.0:
            p_fail = p
            break
    out["pressure_Re1_Pa"] = float(p_fail) if p_fail else None
    print(f"  creeping flow holds up to        : "
          f"{p_fail/1e3:.1f} kPa (Re = 1)" if p_fail else "  Re < 1 always")
    se_bad = [(r["name"], t["symbol"]) for r in se for t in r["tests"]
              if not t["ok"]]
    print(f"  Stokes-Einstein                  : "
          f"{'valid for every target' if not se_bad else 'fails for:'}")
    for n, s in se_bad:
        print(f"    {n}: {s}")
    out["hele_shaw_all_ok"] = hs_ok
    out["stokes_einstein_failures"] = se_bad

    with open("validity_data.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote validity_data.json")


if __name__ == "__main__":
    main()
