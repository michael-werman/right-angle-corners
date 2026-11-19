"""
pack_yud.py -- extract from the York Urban Database just the arrays the E5
harness needs, into one small file you can upload.

    python3 pack_yud.py --root /path/to/YorkUrbanDB

writes yud_packed.npz, typically a few hundred kilobytes.  The database itself
is around 100 MB because every LinesAndVP.mat embeds a 480x640 image; the
harness never looks at the pixels, so they are dropped.

What is kept, per image: the line segment endpoints, the vanishing-point
label of each segment, and the ground-truth orthogonal vanishing-point frame.
Plus, once, the focal length and principal point.  Nothing else.

Then run the evaluation on the packed file, with or without the database:

    python3 yorkurban.py --packed yud_packed.npz
"""

import argparse
import glob
import os

import numpy as np

from yorkurban import (_find_labels, _find_lines, _find_vp,
                       _squeeze_candidates, load_camera)

try:
    from scipy.io import loadmat
except ImportError:
    raise SystemExit("this needs scipy (scipy.io.loadmat)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="YorkUrbanDB directory")
    ap.add_argument("--out", default="yud_packed.npz")
    ap.add_argument("--focal", type=float, help="override the ground-truth f")
    ap.add_argument("--pp", type=float, nargs=2, help="override the principal point")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    f_true, pp = load_camera(args.root, args.focal, args.pp, args.verbose)
    print(f"camera: f = {f_true:.4f} px, principal point = "
          f"({pp[0]:.4f}, {pp[1]:.4f})")

    out = {"__focal": np.array([f_true]), "__pp": np.asarray(pp, float)}
    kept = skipped = 0
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        cands = []
        for m in sorted(glob.glob(os.path.join(d, "*.mat"))):
            try:
                cands += [(k, v) for k, v in _squeeze_candidates(loadmat(m))]
            except Exception as exc:                    # noqa: BLE001
                if args.verbose:
                    print(f"  {name}: could not read {os.path.basename(m)}: {exc}")
        lk = _find_lines(cands)
        if lk is None:
            skipped += 1
            if args.verbose:
                print(f"  {name}: no endpoint array; keys "
                      f"{sorted({k for k, _ in cands})}")
            continue
        lines = lk[1]
        bk = _find_labels(cands, len(lines) // 2)
        if bk is None:
            skipped += 1
            if args.verbose:
                print(f"  {name}: no label vector")
            continue
        vk = _find_vp(cands)
        out[name + "/lines"] = lines.astype(np.float32)
        out[name + "/labels"] = bk[1].astype(np.int8)
        if vk is not None:
            out[name + "/vp"] = vk[1].astype(np.float32)
        kept += 1
        if args.verbose:
            print(f"  {name}: {len(lines)//2} segments, lines='{lk[0]}', "
                  f"labels='{bk[0]}', vp='{vk[0] if vk else None}'")

    if not kept:
        raise SystemExit("nothing extracted; rerun with --verbose and check "
                         "that --root points at the folder holding "
                         "cameraParameters.mat and the image subfolders")
    np.savez_compressed(args.out, **out)
    mb = os.path.getsize(args.out) / 1e6
    print(f"packed {kept} images ({skipped} skipped) into {args.out}, "
          f"{mb:.2f} MB")
    print("upload that one file.")


if __name__ == "__main__":
    main()
