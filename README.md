# What Right-Angle Corners Reveal About an Uncalibrated Camera

Code and LaTeX source accompanying the paper *What Right-Angle Corners Reveal
About an Uncalibrated Camera: Convex Cones, Certified Bounds and
Identifiability* by Michael Werman (Hebrew University of Jerusalem).

> The corner of a box or a room, where three perpendicular edges meet, appears
> in an image as three segments meeting at a point. We ask what such junctions
> reveal about an uncalibrated camera's intrinsic parameters. Each junction is
> consistent with an explicit convex set of cameras; when only the focal length
> is unknown, this set is an interval in closed form. These sets are computed
> from the image alone, so calibrating from many junctions is a convex
> feasibility problem. Almost every junction bounds the focal length from
> below; we characterize when one also bounds it from above, and when many
> junctions determine it exactly. For a known vertex, the bounds can be
> certified against any bounded error in the measured edge directions.

- **`manuscript.pdf`** — the article.
- **`supplement.pdf`** — Online Resource 1.
- **`code/`** — Python scripts that verify every algebraic identity and
  regenerate every reported number and data figure. See
  [`code/README.md`](code/README.md) for the full reproduction table.
- **`paper/`** — LaTeX source (Springer Nature `sn-jnl` template).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
cd code
python3 symbolic.py          # 15 s, expected last line: "88/88 checks passed"
```

To reproduce the real-image results, download the York Urban Database from its
authors and run `python3 pack_yud.py --root /path/to/YorkUrbanDB --out
data/yud_packed.npz`. The database is not redistributed here.

## Citation

See [`CITATION.cff`](CITATION.cff), or use GitHub's "Cite this repository"
button.

## License

Code and LaTeX written by the author are released under the MIT License (see
[`LICENSE`](LICENSE)). The Springer Nature template files in `paper/`
(`sn-jnl.cls`, `sn-mathphys-num.bst`) are redistributed under their own
license; see `paper/README.md`.
