#!/usr/bin/env python3
"""Convert ImageCAS / CAS-X coronary data into a SeqSeg dataset (native mm).

Output layout under ``--path``:

    images/{case}.nii.gz      CTA, mm, identity direction
    truths/{case}.nii.gz      from source/segmentations/{id}.coronary.nii.gz only
    centerlines/{case}.vtp    merged L/R anatomical, mm, with radius
    surfaces/{case}.vtp       anatomical surfaces, mm
    seeds.json

Images are taken from ``source/images_*/{id}.img.nii.gz``.
Does not use ``{id}.label.nii.gz``.

vascular-segment-sampler maps points as ``(point - origin) / spacing`` and
ignores the direction matrix. ImageCAS volumes are Y-flipped (dirY=-1); this
script flips that axis and moves the origin so meshes and volumes share space.

Examples::

    python -m seqseg.scripts.prepare_casx_dataset --path /path/to/CAS_X_coronary_dataset
    python -m seqseg.scripts.prepare_casx_dataset --path /path/to/CAS_X_coronary_dataset --fix-direction
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

DEFAULT_RADIUS_MM = 2.0
IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def read_poly(path: Path) -> vtk.vtkPolyData:
    path = Path(path)
    if path.suffix.lower() == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(reader.GetOutput())
    return out


def write_vtp(poly: vtk.vtkPolyData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    writer.Write()


def append_poly(*polys: vtk.vtkPolyData) -> vtk.vtkPolyData:
    app = vtk.vtkAppendPolyData()
    for p in polys:
        if p is not None and p.GetNumberOfPoints() > 0:
            app.AddInputData(p)
    app.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(app.GetOutput())
    return out


def add_inscribed_radius(centerline: vtk.vtkPolyData, surface: vtk.vtkPolyData) -> None:
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(surface)
    n = centerline.GetNumberOfPoints()
    radii = np.empty(n, dtype=np.float64)
    for i in range(n):
        radii[i] = abs(implicit.EvaluateFunction(centerline.GetPoint(i)))
    vtk_arr = n2v(radii, deep=True)
    vtk_arr.SetName("MaximumInscribedSphereRadius")
    centerline.GetPointData().AddArray(vtk_arr)


def radius_at(poly: vtk.vtkPolyData, point_id: int, default: float = DEFAULT_RADIUS_MM) -> float:
    rads = poly.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if rads is None:
        return default
    return max(float(rads.GetTuple1(point_id)), 0.5)


def anatomical_ostia(poly: vtk.vtkPolyData) -> list:
    starts = poly.GetPointData().GetArray("start_points")
    if starts is None:
        return []
    sv = v2n(starts)
    idx = np.where(sv > 0)[0]
    seeds = []
    for sid in idx:
        sid = int(sid)
        start = np.array(poly.GetPoint(sid))
        direction = None
        for c in range(poly.GetNumberOfCells()):
            cell = poly.GetCell(c)
            ids = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
            if sid not in ids:
                continue
            pos = ids.index(sid)
            if pos + 8 < len(ids):
                direction = np.array(poly.GetPoint(ids[pos + 8]))
            elif pos - 8 >= 0:
                direction = np.array(poly.GetPoint(ids[pos - 8]))
            elif len(ids) > 1:
                other = ids[-1] if pos == 0 else ids[0]
                direction = np.array(poly.GetPoint(other))
            break
        if direction is None:
            continue
        seeds.append((start, direction, radius_at(poly, sid)))
    return seeds


def unique_ostia(poly: vtk.vtkPolyData, min_points: int = 20):
    ostia = []
    seen = []
    for c in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(c)
        n = cell.GetNumberOfPoints()
        if n < min_points:
            continue
        ids = [cell.GetPointId(k) for k in range(n)]
        rads = poly.GetPointData().GetArray("MaximumInscribedSphereRadius")
        if rads is not None:
            r0 = rads.GetTuple1(ids[0])
            r1 = rads.GetTuple1(ids[-1])
            if r1 > r0:
                ids = list(reversed(ids))
        start = np.array(poly.GetPoint(ids[0]))
        if any(np.linalg.norm(start - s) < 1e-2 for s in seen):
            continue
        seen.append(start)
        step = min(12, max(3, n // 20))
        direction = np.array(poly.GetPoint(ids[step]))
        ostia.append((start, direction, radius_at(poly, ids[0]), n))
    ostia.sort(key=lambda t: t[2] * t[3], reverse=True)
    return ostia[:2]


def canonicalize_identity_direction(img: sitk.Image) -> sitk.Image:
    """Flip negative-direction axes so origin+spacing matches mesh coordinates."""
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    if np.allclose(direction, np.eye(3)):
        return img
    size = np.array(img.GetSize(), dtype=float)
    spacing = np.array(img.GetSpacing(), dtype=float)
    origin = np.array(img.GetOrigin(), dtype=float)
    arr = sitk.GetArrayFromImage(img)
    new_origin = origin.copy()
    # SimpleITK index axes (x, y, z) <-> numpy array axes (z, y, x)
    for ax, np_ax in enumerate((2, 1, 0)):
        if direction[ax, ax] >= 0:
            continue
        arr = np.flip(arr, axis=np_ax)
        new_origin[ax] = origin[ax] + direction[ax, ax] * (size[ax] - 1) * spacing[ax]
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    out.SetOrigin(tuple(float(v) for v in new_origin))
    out.SetDirection(IDENTITY_DIRECTION)
    return out


def write_canonical_volume(src: Path, dst: Path) -> None:
    img = canonicalize_identity_direction(sitk.ReadImage(str(src)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    sitk.WriteImage(img, str(dst), useCompression=True)


def fix_one_volume(path_str: str) -> tuple[str, str]:
    path = Path(path_str)
    img = sitk.ReadImage(str(path))
    direction = img.GetDirection()
    if all(abs(a - b) < 1e-9 for a, b in zip(direction, IDENTITY_DIRECTION)):
        return path.name, "skip"
    out = canonicalize_identity_direction(img)
    if path.exists():
        path.unlink()
    sitk.WriteImage(out, str(path), useCompression=True)
    return path.name, "wrote"


def image_dirs(source: Path) -> list[Path]:
    return sorted(p for p in source.glob("images_*") if p.is_dir())


def find_image(case: str, source: Path) -> Path | None:
    for d in image_dirs(source):
        p = d / f"{case}.img.nii.gz"
        if p.is_file():
            return p
    return None


def move_originals(root: Path) -> dict:
    source = root / "source"
    mapping = {
        root / "images" / "1-200": source / "images_1-200",
        root / "images" / "sv_smooth_surface": source / "sv_smooth_surface",
        root / "centerlines": source / "centerlines_anatomical",
        root / "surfaces": source / "surfaces_anatomical",
        root / "segmentations": source / "segmentations",
    }
    source.mkdir(exist_ok=True)
    for src, dst in mapping.items():
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"moved {src.relative_to(root)} -> {dst.relative_to(root)}")
    (root / "images").mkdir(exist_ok=True)
    dirs = image_dirs(source)
    print("image source folders:", ", ".join(d.name for d in dirs) or "(none)")
    return {
        "root": root,
        "source": source,
        "image_dirs": dirs,
        "cent_anat": source / "centerlines_anatomical",
        "surf_anat": source / "surfaces_anatomical",
        "segs": source / "segmentations",
    }


def case_ids(paths: dict) -> list[str]:
    ids = set()
    for p in paths["segs"].glob("*.coronary.nii.gz"):
        ids.add(p.name.split(".")[0])
    for d in paths["image_dirs"]:
        for p in d.glob("*.img.nii.gz"):
            ids.add(p.name.split(".")[0])
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def refresh_seed_radii(seed_list: list, merged: vtk.vtkPolyData) -> list:
    if not seed_list:
        ostia = unique_ostia(merged)
        return [(s, d, r) for s, d, r, _ in ostia]
    pts = np.array([merged.GetPoint(i) for i in range(merged.GetNumberOfPoints())])
    if pts.size == 0:
        return seed_list
    refreshed = []
    for start, direction, radius in seed_list:
        dist = np.linalg.norm(pts - np.asarray(start), axis=1)
        pid = int(np.argmin(dist))
        refreshed.append((start, direction, radius_at(merged, pid, radius)))
    return refreshed


def process_case(case: str, root_str: str) -> dict:
    root = Path(root_str)
    source = root / "source"
    cent_anat = source / "centerlines_anatomical"
    surf_anat = source / "surfaces_anatomical"
    segs = source / "segmentations"

    img_orig = find_image(case, source)
    seg = segs / f"{case}.coronary.nii.gz"
    surf_src = surf_anat / f"{case}.coronary_surface.vtk"
    left = cent_anat / f"{case}.coronary_left_centerline.vtk"
    right = cent_anat / f"{case}.coronary_right_centerline.vtk"

    out_img = root / "images" / f"{case}.nii.gz"
    out_truth = root / "truths" / f"{case}.nii.gz"
    out_cent = root / "centerlines" / f"{case}.vtp"
    out_surf = root / "surfaces" / f"{case}.vtp"

    for d in ("images", "truths", "centerlines", "surfaces"):
        (root / d).mkdir(exist_ok=True)

    status = {
        "case": case,
        "image": False,
        "truth": False,
        "centerline": False,
        "surface": False,
        "seeds": [],
    }

    if img_orig is not None:
        write_canonical_volume(img_orig, out_img)
        status["image"] = True

    if seg.exists():
        write_canonical_volume(seg, out_truth)
        status["truth"] = True

    parts = []
    seed_list = []
    for branch in (left, right):
        if not branch.exists():
            continue
        p = read_poly(branch)
        parts.append(p)
        seed_list.extend(anatomical_ostia(p))
    if parts:
        if out_cent.exists():
            merged = read_poly(out_cent)
            seed_list = refresh_seed_radii(seed_list, merged)
        else:
            merged = append_poly(*parts)
            if surf_src.exists():
                add_inscribed_radius(merged, read_poly(surf_src))
                seed_list = refresh_seed_radii(seed_list, merged)
            write_vtp(merged, out_cent)
        status["centerline"] = True
        status["seeds"] = [[s.tolist(), d.tolist(), float(r)] for s, d, r in seed_list]

    if surf_src.exists():
        if not out_surf.exists():
            write_vtp(read_poly(surf_src), out_surf)
        status["surface"] = True

    return status


def n_workers(requested: int | None) -> int:
    if requested is not None and requested > 0:
        return requested
    return max(1, min(8, os.cpu_count() or 4))


def fix_existing_volumes(root: Path, workers: int) -> None:
    paths = [str(p) for p in sorted((root / "images").glob("*.nii.gz"))]
    paths += [str(p) for p in sorted((root / "truths").glob("*.nii.gz"))]
    print(f"Fixing direction on {len(paths)} volumes ({workers} workers)")
    n_wrote = n_skip = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fix_one_volume, p): p for p in paths}
        done = 0
        for fut in as_completed(futs):
            _name, status = fut.result()
            n_wrote += int(status == "wrote")
            n_skip += int(status == "skip")
            done += 1
            if done % 50 == 0 or done == len(paths):
                print(f"  {done}/{len(paths)} wrote={n_wrote} skip={n_skip}")
    print(f"direction fix done wrote={n_wrote} already_identity={n_skip}")


def run_prepare(root: Path, workers: int) -> None:
    paths = move_originals(root)
    cases = case_ids(paths)
    print(
        f"{len(cases)} cases, {workers} workers "
        "(mm, truths from segmentations/, images from source/images_*)"
    )

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_case, c, str(root)): c for c in cases}
        done = 0
        for fut in as_completed(futs):
            case = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = {"case": case, "error": str(e)}
            results.append(res)
            done += 1
            if done % 25 == 0 or done == len(cases):
                print(f"  {done}/{len(cases)}")

    seeds = []
    n_img = n_truth = n_cent = n_surf = n_err = 0
    for r in sorted(
        results,
        key=lambda x: int(x.get("case", "0")) if str(x.get("case", "0")).isdigit() else 0,
    ):
        if r.get("error"):
            n_err += 1
            print("ERROR", r["case"], r["error"])
            continue
        n_img += int(bool(r.get("image")))
        n_truth += int(bool(r.get("truth")))
        n_cent += int(bool(r.get("centerline")))
        n_surf += int(bool(r.get("surface")))
        if r.get("image"):
            seeds.append(
                {
                    "name": r["case"],
                    "seeds": r.get("seeds") or [],
                    "cardiac_mesh": False,
                }
            )

    seeds.sort(key=lambda s: int(s["name"]) if str(s["name"]).isdigit() else s["name"])
    seeds_path = root / "seeds.json"
    with open(seeds_path, "w", encoding="utf-8") as f:
        json.dump(seeds, f, indent=2)

    print(
        f"done images={n_img} truths={n_truth} centerlines={n_cent} "
        f"surfaces={n_surf} errors={n_err} seeds={len(seeds)} -> {seeds_path}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert ImageCAS / CAS-X coronary data into a SeqSeg dataset.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Dataset root (contains source/ or native ImageCAS folders)",
    )
    parser.add_argument(
        "--fix-direction",
        action="store_true",
        help="Rewrite existing images/ and truths/ to identity direction (mm)",
    )
    parser.add_argument("--num-cores", type=int, default=None, help="Worker processes")
    args = parser.parse_args(argv)

    root = Path(args.path).expanduser().resolve()
    workers = n_workers(args.num_cores)
    if args.fix_direction:
        fix_existing_volumes(root, workers)
        return
    run_prepare(root, workers)


if __name__ == "__main__":
    main()
