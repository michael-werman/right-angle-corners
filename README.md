# Code for "What Right-Angle Corners Reveal About an Uncalibrated Camera"

This folder verifies every algebraic identity in the article and its Online
Resource, and regenerates every reported number and every data figure.

## Requirements

Python 3 with numpy, scipy, sympy and matplotlib. The results quoted in the
paper were produced with Python 3.12, numpy 2.4, scipy 1.17, sympy 1.14 and
matplotlib 3.10. Random seeds are fixed in every script, so the printed numbers
should reproduce to the precision shown, up to differences between library
versions.

## The York Urban Database

The real-image scripts read one small file, `data/yud_packed.npz`, which holds
the line segments, labels and vanishing points of the York Urban Database
(Denis, Elder and Estrada, 2008) and no images. It is not distributed here.
To make it, download the database from its authors and run

    python3 pack_yud.py --root /path/to/YorkUrbanDB --out data/yud_packed.npz

The file may be placed elsewhere and pointed to with the environment variable
`YUD_PACKED`. No script uses ground-truth intrinsics unless its output says so.

## Symbolic verification

    python3 symbolic.py --aux main.aux

Each check is labelled with the theorem or equation it supports. If the
`.aux` files of the compiled article (and of the Online Resource, as
`supplement.aux` beside it) are given, the labels show the current numbers;
without them the script prints the LaTeX labels instead. Expected last line:
`88/88 checks passed`.

## What reproduces what

Run times are from a single core of the machine on which the package was
tested; they are a guide only. Results are named by title rather than number, since numbers change
between versions of the article; sections marked S are in the Online Resource.

| Result | Script | Time |
|---|---|---|
| **Every algebraic identity** (81 checks) | `symbolic.py --aux main.aux` | 15 s |
| Third referee report: the miss probability near raw right angles; one matched axis in the mirror theorem; the tail near an interior vanishing point; Wilson interval; sign test | `referee3_checks.py` | 60 s |
| Online Resource S12, real images: classical calibration from three vanishing points | `vp3_baseline.py` | 10 s |
| Online Resource S9 and S11: replicated widths for the cube-diagonal frame, slope intervals | `transition_replicates.py` | 4 min |
| Online Resource S8: combining bounds over all trials, with the binomial prediction; Online Resource S13: robustness ablations | `referee3_experiments.py` | 4 min |
| Online Resource S8: the log-plausibility with the threshold-based and the form-based noise scale, paired | `loglik_scale_test.py` | 5 min |
| Online Resource S8, Table: the log-plausibility at low junction counts with the form-based scale (run per noise level: `python lowcount_form.py 0.5`, `2.0`) | `lowcount_form.py` | 2 x 4 min |
| Proposition (certified bounds); Online Resource S6: corollary (distortion with fitted segments): the two inclusions, the angle bounds | `certified_test.py` | 2 min |
| Online Resource S12, real images: certified intervals for the 181 York Urban junctions | `yu_certified.py` | 1 min |
| Limitations: the indoor pipeline, ready to run on a dataset with intrinsics (`python indoor_pipeline.py IMAGE_DIR INTRINSICS.csv`); its self-test on rendered scenes | `indoor_pipeline.py`, `indoor_pipeline_selftest.py` | 1 min |
| Propositions (unknown aspect ratio; unknown principal point): the half-plane characterization, the form f^2 A_ij + d_i d_j, scaling about the junction | `camera_models_test.py` | 1 min |
| Figure of Proposition (unknown principal point): the three hyperbolas of one junction | `figpp.py` | 1 min |
| Online Resource S6: propositions (corners that are not right-angled; radial distortion): the condition and the transformation, coverage | `robustness_proofs_test.py` | 2 min |
| Online Resource S8: disconnected consensus sets, the independence of the flips; mirror resolution intervals | `review7_checks.py` | 4 min |
| Online Resource S8: the log-plausibility at low junction counts, bootstrap intervals, empty intersections | `lowcount_ci.py` | 5 min |
| Online Resource S9: proposition (random junctions); the Lipschitz constant and the non-asymptotic bound of Online Resource S7, proposition (first-order spread); Corollary (a wide field of view suffices) | `grounding_checks.py` | 2 min |
| Theorem (where a junction bounds the focal length): the factored form of gamma, the single-junction interval, the bounded-above region on York Urban | `bound_map_test.py` | 30 s |
| Proposition (one edge-on line through the image) against brute force; Corollary (a camera that only pitches) | `global_oneline_test.py` | 4 min |
| Theorem (the admissible set in closed form), Proposition (focal-length bound) part 2, Theorem (mirror resolution): brute force | `theorem_pass_test.py` | 15 s |
| Theorem (simplicial cone): the criterion agrees with the definition, 200,000 trials | `simplex_test.py` | 14 s |
| Theorem (simplicial cone): predicted cell equals the admissible set, 2.25 million cameras | `prove_cell.py` | 108 s |
| Proposition (random junctions), a first check; uniform-sign corners (Online Resource S4) | `random_triples.py` | 20 s |
| Online Resource S7, proposition (first-order spread): threshold-based and form-based predictions of the miss probability | `noise_thm.py` | 21 s |
| Lemma (a junction of the frame) against the definition; one edge-on line through the image | `exact_converse_test.py` | 19 s |
| What one frame forces: the aspect-ratio camera from one frame | `lp_oneframe.py` | 1 s |
| What one frame forces: the general camera from one frame; cosine between vanishing points | `sdp_frame2.py`, `vp_mechanism.py` | 2 s, 1 s |
| Theorem (what one frame forces): a line that only touches the image forces one condition | `touch_test.py` | 2 s |
| Theorem (several frames): the criterion against the rank of the conditions, 4000 axis sets | `frames_general_test.py` | 1 s |
| Corollary (images that show only their horizons) | `horizons_test.py` | 5 s |
| Theorem (several frames): touching lines; reduction to Theorem (identifiability of the focal length) | `touch_frames_test.py` | 60 s |
| Theorem (focal length and principal point), first order and junction level | `fpp_test.py` | 10 s |
| Theorem (several frames): frames with and without a common axis | `two_frames_test.py`, `sdp_frames_n.py` | 5 s, 4 s |
| Proposition (the joint admissible set in closed form) against the intersection of intervals | `gform_test.py` | 14 s |
| Online Resource S9, Proposition (rate of identifiability), part 4: the limit law, the constant chi against simulation | `limit_law_test.py` | 4 min |
| Online Resource S9, Proposition (rate of identifiability): 24 random frames | `rate.py` | 48 s |
| Online Resource S9, Proposition (rate of identifiability): the three borderline behaviours | `border_test.py` | 20 s |
| Online Resource S9, Proposition (rate of identifiability), part 3: a vanishing point inside the image | `vp_inside_test.py` | 18 s |
| Identifiability of the focal length: level and tilted cameras | `level_test.py` | 25 s |
| Online Resource S10: identifiability (the transition), and Online Resource S9: 54 and 60 degree fields of view | `converge.py` | 63 s |
| Online Resource S10: a misplaced principal point, and the article's Sect. 6 | `pp_offset.py` | 40 s |
| The convex program and Online Resource S11: linear program for the aspect ratio; cutting planes for the full camera | `lp_calib.py`, `sdp_calib.py`, `sdp_frame.py` | 1 s, 1 s, 1 s |
| Experiments, the focal-length bound: coverage and the similarity law | `experiments.py e2`, `experiments.py sim` | 1 s, 2 s |
| Online Resource S13: a junction whose vertex is not seen | `occluded.py`, `occl_noise.py` | 1 s, 10 s |
| Identifiability experiments and Online Resource S11 | `ident.py`, `audit.py`, `sharp.py` | 3 s, 10 s, 44 s |
| Online Resource S8: combining bounds, earlier runs | `experiments.py e6`, `lowcount_merged.py` | 2 min |
| Online Resource S2: mirror resolution by ray separation | `experiments.py e3a` | 4 s |
| Online Resource S12, real images: coverage, sign test, closed form, mirror | `yorkurban.py --packed data/yud_packed.npz` | 54 s |
| Online Resource S12, real images: bound statistics and the similarity law | `e5final.py` | 1 s |
| The single focal length and real images: the camera-free test and the test with the true camera, as detectors | `detector.py` | 1 s |
| Online Resource S1: junction sampling against line sampling on York Urban, paired; Online Resource S13: extraction thresholds | `paired.py` | 212 s |
| Real images and Online Resource S8: the 181 intervals; log-plausibility; robust program; bootstrap | `yu_bounds.py`, `yu_lik.py`, `yu_final.py`, `yu_boot.py` | 1 s, 6 s, 3 s, 9 s |
| Online Resource S12, real images: simulations with York Urban's edge lengths; field of view | `lik_sim.py`, `wide_sim.py` | 43 s, 43 s |
| Online Resource S12, real images: York Urban vertical directions and horizons | `yu_verticals.py` | 2 s |
| Online Resource S12, real images: tilt of the York Urban frames; the 181 junctions without noise | `level_theory.py`, `tilt_exact.py`, `tilt_match.py` | 1 s, 1 s, 1 s |
| Data figures of the article (the intervals of ten junctions, lower bound, real images), the transition figure of Online Resource S10, the mirror figure of Online Resource S2, and Fig. S1; `figgeom.py` also draws the earlier two-dimensional picture of the aspect-ratio model, no longer in the article | `figintervals.py`, `figgeom.py`, `figures.py`, `figreal.py` | 3 s, 29 s, 19 s, 2 s |
| Figure of Theorem (where a junction bounds the focal length); shared figure style; the disc picture of Theorem (simplicial cone), no longer a figure of the article | `figdisc.py`, `figboundmap.py`, `figstyle.py` | 20 s |
| Online Resource S13: closed form, corner-plus-line, whole-box fit | `experiments.py e1`, `e3b`, `e4b` | 4 s, 7 s, 46 s |

## Modules

`common.py` (corner geometry, the admissible interval), `experiments.py`
(synthetic experiments), `calib_core.py` (sign patterns from the junction
alone, and the cutting-plane program), `robust_calib.py`, `lik_calib.py`,
`robust_sim.py` (estimators and scene generator), `yorkurban.py`,
`yu_junctions.py`, `yud_path.py` (real data).

The cube-diagonal transition figure (`figgeom.py`, written as `fig9_transition.pdf`) caches its simulation in
`fig7_panel_c.json`; delete that file to recompute it.
