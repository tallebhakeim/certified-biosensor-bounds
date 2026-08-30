# certified-biosensor-bounds

Solver, verification suite and study scripts for the paper:

H. Talleb, "Certified-Bound Design of Capacitive and Conductimetric
Lab-on-Chip Biosensors: Guaranteed Geometric Factors, Detection Windows and
Electrode Rankings," *IEEE Sensors Journal*, 2026,
doi: [10.1109/JSEN.2026.3726458](https://doi.org/10.1109/JSEN.2026.3726458).

Every figure and every table of the paper is produced by the scripts in this
repository; each study script writes its `*_data.json` (included here) and its
figure in `figures/`.

## Contents

| file | role |
|---|---|
| `certified_core.py` | the dual bracket. Upper bound: conforming Q1 finite elements (Dirichlet principle); lower bound: cell-centred finite-volume flux with the co-energy of the lowest-order Raviart-Thomas reconstruction evaluated exactly (Thomson principle). Algebraic multigrid beyond 60 000 unknowns, with an explicit check of the conservation residual: an inexact solve makes the equilibrated flux inadmissible, so the lower bound is refused rather than reported. Also contains the Wiener, Hashin-Shtrikman and Maxwell-Garnett bounds. |
| `geometry.py` | voxelisation in physical units: interdigitated comb (quarter cell), plane sandwich, 3D planar patterns, sensing zone, meander, 3D suspension cube. The `snap="in"/"out"` parameter is what makes the geometric bounds guaranteed (conservative rounding of the conductor). |
| `device.py` | reference chip (40 um pitch, 20 um channel, 50 periods x 2 mm, PBS) and derived quantities: capacitance, conduction geometric factor, certified sensitivity by interval subdivision. |
| `figstyle.py` | common figure style. |
| `test_core.py` | verification suite, **16 tests**. Analytical 2D/3D slabs, series capacitors, heterogeneous checkerboard inside the Wiener bounds, decrease and nesting of the enclosures under refinement, detection of an isolated particle. |

## Studies (one per section of the paper)

| script | paper section | output |
|---|---|---|
| `study_edl.py` | IV | double layer, frequency window (f_edl, f_diel) per medium, bounded ionic strength |
| `study_homogenisation.py` | V | Wiener / Hashin-Shtrikman / Maxwell-Garnett, MG as an extremal bound, 3D microstructure check |
| `study_fabrication.py` | VI | tolerance envelope by inclusion monotonicity; Monte-Carlo comparison; ranking survival |
| `study_detection.py` | VII | certified detection on both read-outs (C and G), detection threshold under growing hypotheses |
| `study_single_cell.py` | VII | 3D response of one cell (400 aF) and the baseline budget |
| `study_patterns.py` | VIII | 3D ranking of the four single-mask patterns on four meshes |
| `study_fluidics.py` | IX-A | certified Hele-Shaw transport, dielectric cover |
| `study_validity.py` | IX-A/B | Hele-Shaw and Stokes-Einstein validity conditions at the operating point |
| `study_thermal.py` | IX-C | Joule heating, bounded heat-sink depth, voltage ceilings |

## Reproduce

Dependencies: `numpy`, `scipy`, `matplotlib`, `pyamg` (see `requirements.txt`).

```sh
python3 test_core.py            # 16/16 PASS expected
python3 study_edl.py
python3 study_detection.py
python3 study_single_cell.py
python3 study_fabrication.py    # ~2 min (50 nm mesh)
python3 study_homogenisation.py # ~3 min (3D microstructures + bisection)
python3 study_patterns.py
python3 study_fluidics.py
python3 study_validity.py
python3 study_thermal.py
```

## License

MIT, see `LICENSE`.
