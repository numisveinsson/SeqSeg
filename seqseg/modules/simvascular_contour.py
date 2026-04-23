"""
Build SimVascular ``.ctgr`` contour groups from a SeqSeg probability (or scalar)
volume and path planes matching ``.pth`` output.

**Probability convention:** values are expected in ``[0, 1]``, where **1** is
most vessel-like and **0** is most background-like (typical SeqSeg assembly).
The default isovalue ``0.5`` matches ``BinaryThreshold(..., 0.5, 1)`` used
elsewhere in SeqSeg.

Contours are intersections of an isosurface threshold with oblique planes
defined by each path point's ``pos``, ``tangent``, and ``rotation`` (same
basis SimVascular uses for reslicing along a path). Output uses
``type="Contour"`` with two ``control_points`` and dense ``contour_points`` in
**world** coordinates (SimVascular threshold-style groups).

When a slice intersects **multiple** vessel components, the default is **not**
to take the largest 2D loop (another branch can dominate the cross-section).
Instead, the connected contour **closest to the path point** ``pos`` is kept
(``vtkPolyDataConnectivityFilter`` in ``ClosestPointRegion`` mode), with a
fallback to the largest region if that fails.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import vtk

from vtk.util.numpy_support import vtk_to_numpy as v2n

from .simvascular import (
    _perpendicular_normal_unit,
    compute_tangents,
    indent,
    resample_path_like_simvascular,
)
from .vtk_functions import exportSitk2VTK

# Default lofting block copied from ``seqseg/tutorial/data/Segmentations/aorta.ctgr``
_DEFAULT_LOFTING_ATTRIBS = {
    "method": "nurbs",
    "sampling": "60",
    "sample_per_seg": "12",
    "use_linear_sample": "1",
    "linear_multiplier": "10",
    "use_fft": "0",
    "num_modes": "20",
    "u_degree": "2",
    "v_degree": "2",
    "u_knot_type": "derivative",
    "v_knot_type": "average",
    "u_parametric_type": "centripetal",
    "v_parametric_type": "chord",
}


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def _plane_basis(
    tangent: Sequence[float], rotation: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthonormal frame: ``u``, ``v`` span the slice plane; ``w`` is the plane
    normal (along centerline tangent).
    """
    w = _normalize(tangent)
    u = _normalize(rotation)
    if abs(float(np.dot(u, w))) > 0.02:
        u = _normalize(np.cross(w, np.cross(u, w)))
    v = _normalize(np.cross(w, u))
    return u, v, w


def parse_pth(path: str) -> Dict[str, Any]:
    """
    Read a SimVascular ``.pth`` file produced by ``simvascular.create_pth``.

    Returns
    -------
    dict
        Keys: ``path_id`` (int), ``method``, ``calculation_number``, ``spacing``,
        ``reslice_size``, ``path_points`` (list of dicts with ``id``, ``pos``,
        ``tangent``, ``rotation``), ``control_points`` (optional list of xyz).
    """
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "path":
        raise ValueError(f"Expected root element 'path', got {root.tag!r}")

    timestep = root.find("timestep")
    if timestep is None:
        raise ValueError("Missing <timestep> in .pth")

    pe = timestep.find("path_element")
    if pe is None:
        raise ValueError("Missing <path_element> in .pth")

    ppoints_el = pe.find("path_points")
    if ppoints_el is None:
        raise ValueError("Missing <path_points> in .pth")

    path_points: List[Dict[str, Any]] = []
    for pp in ppoints_el.findall("path_point"):
        pid = int(pp.attrib["id"])
        pos_el = pp.find("pos")
        tan_el = pp.find("tangent")
        rot_el = pp.find("rotation")
        if pos_el is None or tan_el is None or rot_el is None:
            continue
        path_points.append(
            {
                "id": pid,
                "pos": np.array(
                    [float(pos_el.attrib["x"]), float(pos_el.attrib["y"]), float(pos_el.attrib["z"])],
                    dtype=float,
                ),
                "tangent": np.array(
                    [
                        float(tan_el.attrib["x"]),
                        float(tan_el.attrib["y"]),
                        float(tan_el.attrib["z"]),
                    ],
                    dtype=float,
                ),
                "rotation": np.array(
                    [
                        float(rot_el.attrib["x"]),
                        float(rot_el.attrib["y"]),
                        float(rot_el.attrib["z"]),
                    ],
                    dtype=float,
                ),
            }
        )

    ctrl: List[np.ndarray] = []
    c_el = pe.find("control_points")
    if c_el is not None:
        for pt in c_el.findall("point"):
            ctrl.append(
                np.array(
                    [float(pt.attrib["x"]), float(pt.attrib["y"]), float(pt.attrib["z"])],
                    dtype=float,
                )
            )

    return {
        "path_id": int(root.attrib["id"]),
        "method": root.attrib.get("method", "0"),
        "calculation_number": root.attrib.get("calculation_number", "100"),
        "spacing": root.attrib.get("spacing", "0"),
        "reslice_size": root.attrib.get("reslice_size", "5"),
        "path_points": path_points,
        "control_points": ctrl,
    }


def sample_path_points_from_control_polyline(
    control_points: Sequence[Sequence[float]],
    *,
    spline_resample: bool = True,
    calculation_number: int = 100,
    spacing: float = 0.0,
    further_subdivision_number: int = 10,
) -> List[Dict[str, Any]]:
    """
    Build dense path samples like ``create_pth`` (VTK Cardinal spline resampling).

    Returns the same structure as ``parse_pth`` ``path_points`` entries.
    """
    pts = [tuple(float(x) for x in p) for p in control_points]
    use_spacing = float(spacing) > 0.0
    if spline_resample or use_spacing:
        sampled = resample_path_like_simvascular(
            pts,
            calculation_number=int(calculation_number),
            further_subdivision_number=int(further_subdivision_number),
            closed=False,
            spacing=float(spacing) if use_spacing else None,
        )
    else:
        tangents = compute_tangents(pts)
        sampled = []
        for i, ((x, y, z), t) in enumerate(zip(pts, tangents)):
            pos = np.array([x, y, z], dtype=float)
            tan = np.asarray(t, dtype=float)
            tn = float(np.linalg.norm(tan))
            if tn > 1e-12:
                tan = tan / tn
            else:
                tan = np.array([0.0, 0.0, 1.0], dtype=float)
            rot = _perpendicular_normal_unit(tan)
            sampled.append({"id": i, "pos": pos, "tangent": tan, "rotation": rot})

    out: List[Dict[str, Any]] = []
    for row in sampled:
        out.append(
            {
                "id": int(row["id"]),
                "pos": np.asarray(row["pos"], dtype=float).reshape(3),
                "tangent": np.asarray(row["tangent"], dtype=float).reshape(3),
                "rotation": np.asarray(row["rotation"], dtype=float).reshape(3),
            }
        )
    return out


def _reslice_plane_scalar(
    vtk_vol: vtk.vtkImageData,
    pos: np.ndarray,
    tangent: np.ndarray,
    rotation: np.ndarray,
    *,
    half_extent_mm: float,
    plane_spacing_mm: float,
) -> Tuple[Optional[vtk.vtkImageData], vtk.vtkMatrix4x4]:
    """
    Scalar image resliced onto a plane perpendicular to ``tangent``.

    Returns
    -------
    slice_image, local_to_world
        ``slice_image`` is the 2D reslice output. ``local_to_world`` maps
        reslice physical coordinates (u/v plane frame) to world coordinates.
    """
    u, v, w = _plane_basis(tangent, rotation)
    pos = np.asarray(pos, dtype=float).reshape(3)

    sp = max(float(plane_spacing_mm), 1e-6)
    nx = max(8, int(math.ceil(2.0 * float(half_extent_mm) / sp)) + 1)

    cx = float(pos[0] - 0.5 * float(nx - 1) * sp * u[0] - 0.5 * float(nx - 1) * sp * v[0])
    cy = float(pos[1] - 0.5 * float(nx - 1) * sp * u[1] - 0.5 * float(nx - 1) * sp * v[1])
    cz = float(pos[2] - 0.5 * float(nx - 1) * sp * u[2] - 0.5 * float(nx - 1) * sp * v[2])

    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(vtk_vol)
    reslice.SetInterpolationModeToLinear()
    reslice.SetOutputDimensionality(2)
    reslice.SetResliceAxesDirectionCosines(
        float(u[0]),
        float(u[1]),
        float(u[2]),
        float(v[0]),
        float(v[1]),
        float(v[2]),
        float(w[0]),
        float(w[1]),
        float(w[2]),
    )
    reslice.SetResliceAxesOrigin(cx, cy, cz)
    reslice.SetOutputSpacing(sp, sp, sp)
    reslice.SetOutputExtent(0, nx - 1, 0, nx - 1, 0, 0)
    if hasattr(reslice, "SetOutputScalarType"):
        reslice.SetOutputScalarType(vtk.VTK_FLOAT)
    if hasattr(reslice, "SetBackgroundLevel"):
        reslice.SetBackgroundLevel(0.0)
    elif hasattr(reslice, "SetBackgroundValue"):
        reslice.SetBackgroundValue(0.0)
    reslice.Update()
    out = reslice.GetOutput()

    local_to_world = vtk.vtkMatrix4x4()
    local_to_world.Identity()
    # world = c + x*u + y*v + z*w
    local_to_world.SetElement(0, 0, float(u[0]))
    local_to_world.SetElement(1, 0, float(u[1]))
    local_to_world.SetElement(2, 0, float(u[2]))
    local_to_world.SetElement(0, 1, float(v[0]))
    local_to_world.SetElement(1, 1, float(v[1]))
    local_to_world.SetElement(2, 1, float(v[2]))
    local_to_world.SetElement(0, 2, float(w[0]))
    local_to_world.SetElement(1, 2, float(w[1]))
    local_to_world.SetElement(2, 2, float(w[2]))
    local_to_world.SetElement(0, 3, float(cx))
    local_to_world.SetElement(1, 3, float(cy))
    local_to_world.SetElement(2, 3, float(cz))

    if out is None or out.GetNumberOfPoints() == 0:
        return None, local_to_world
    return out, local_to_world


def _polyline_world_from_region(
    poly: vtk.vtkPolyData,
    *,
    seed_xyz: Optional[np.ndarray],
    component_selection: str,
) -> Optional[np.ndarray]:
    """
    Pick one contour component and return an ordered Nx3 polyline in world space.

    ``component_selection``:
        - ``"closest_to_path"`` (default): region whose geometry is closest to
          ``seed_xyz`` (path ``pos``). Avoids picking a larger unrelated branch
          in the same oblique slice.
        - ``"largest"``: legacy behaviour (biggest connected contour only).
    """
    if poly is None or poly.GetNumberOfPoints() == 0:
        return None

    def _connectivity(mode: str) -> vtk.vtkPolyData:
        c = vtk.vtkPolyDataConnectivityFilter()
        c.SetInputData(poly)
        if mode == "largest":
            c.SetExtractionModeToLargestRegion()
        elif mode == "closest_to_path":
            if seed_xyz is None:
                c.SetExtractionModeToLargestRegion()
            else:
                c.SetExtractionModeToClosestPointRegion()
                c.SetClosestPoint(
                    float(seed_xyz[0]),
                    float(seed_xyz[1]),
                    float(seed_xyz[2]),
                )
        else:
            raise ValueError(
                f"Unknown component_selection {mode!r}; use 'closest_to_path' or 'largest'"
            )
        c.Update()
        return c.GetOutput()

    region = _connectivity(component_selection)
    if (
        region is None
        or region.GetNumberOfPoints() == 0
    ) and component_selection == "closest_to_path":
        region = _connectivity("largest")
    if region is None or region.GetNumberOfPoints() == 0:
        return None

    strip = vtk.vtkStripper()
    strip.SetInputData(region)
    strip.Update()
    stripped = strip.GetOutput()
    if stripped is None or stripped.GetNumberOfPoints() == 0:
        pts = v2n(region.GetPoints().GetData())
        return np.asarray(pts, dtype=float)

    pts = v2n(stripped.GetPoints().GetData())
    arr = np.asarray(pts, dtype=float)
    if arr.shape[0] < 3:
        return None

    if stripped.GetNumberOfCells() > 0:
        cell = stripped.GetCell(0)
        n = cell.GetNumberOfPoints()
        if n > 0:
            ids = [cell.GetPointId(i) for i in range(n)]
            arr = arr[ids]

    return arr


def _smooth_closed_polyline(
    xyz: np.ndarray, iterations: int = 12, relaxation: float = 0.2
) -> np.ndarray:
    """
    Cyclic Laplacian smoothing for closed contours.

    Keeps point count unchanged and smooths high-frequency jaggedness.
    """
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 4:
        return arr
    alpha = float(relaxation)
    if alpha <= 0.0 or iterations <= 0:
        return arr
    alpha = min(alpha, 0.5)
    out = arr.copy()
    for _ in range(int(iterations)):
        prev = np.roll(out, 1, axis=0)
        nxt = np.roll(out, -1, axis=0)
        out = (1.0 - alpha) * out + 0.5 * alpha * (prev + nxt)
    return out


def extract_plane_contour_world(
    sitk_volume: sitk.Image,
    pos: Sequence[float],
    tangent: Sequence[float],
    rotation: Sequence[float],
    *,
    iso_value: float = 0.5,
    half_extent_mm: float = 50.0,
    plane_spacing_mm: Optional[float] = None,
    vtk_cache: Optional[Tuple[vtk.vtkImageData, Any]] = None,
    component_selection: str = "closest_to_path",
) -> Optional[np.ndarray]:
    """
    Reslice ``sitk_volume`` on the path plane and extract a closed polyline at
    ``iso_value`` (marching squares / contour filter).

    Parameters
    ----------
    sitk_volume
        Float probability map in ``[0, 1]`` (1 = vessel) or any scalar image;
        values are compared to ``iso_value`` directly (default ``0.5``).
    pos
        Path point in world space; also used to choose among multiple disjoint
        iso-contours when ``component_selection`` is ``"closest_to_path"``.
    vtk_cache
        Optional ``(vtk_image, unused)`` from ``exportSitk2VTK`` to avoid
        reconverting the same volume for many planes.
    component_selection
        ``"closest_to_path"`` (default): keep the connected contour region
        closest to ``pos``. ``"largest"``: keep the largest region only.

    Returns
    -------
    ndarray of shape (N, 3) or None if no contour is found.
    """
    # Use a consistently finer in-plane sampling: half of native minimum spacing.
    sp_arr = np.array(sitk_volume.GetSpacing(), dtype=float)
    plane_spacing_mm = float(max(1e-6, 0.5 * float(np.min(sp_arr))))

    if vtk_cache is not None:
        vtk_vol = vtk_cache[0]
    else:
        vtk_vol, _ = exportSitk2VTK(sitk_volume, interpolation="linear")

    sl, local_to_world = _reslice_plane_scalar(
        vtk_vol,
        np.asarray(pos, dtype=float),
        np.asarray(tangent, dtype=float),
        np.asarray(rotation, dtype=float),
        half_extent_mm=float(half_extent_mm),
        plane_spacing_mm=float(plane_spacing_mm),
    )
    if sl is None:
        return None

    cf = vtk.vtkContourFilter()
    cf.SetInputData(sl)
    cf.SetValue(0, float(iso_value))
    cf.ComputeScalarsOff()
    cf.ComputeNormalsOff()
    cf.Update()

    # vtkContourFilter output is in the 2D reslice frame; map it back to world
    # before component selection and XML serialization.
    tfm = vtk.vtkTransform()
    tfm.SetMatrix(local_to_world)
    tpf = vtk.vtkTransformPolyDataFilter()
    tpf.SetTransform(tfm)
    tpf.SetInputData(cf.GetOutput())
    tpf.Update()

    pos_arr = np.asarray(pos, dtype=float).reshape(3)
    loop = _polyline_world_from_region(
        tpf.GetOutput(),
        seed_xyz=pos_arr,
        component_selection=component_selection,
    )
    if loop is None or loop.shape[0] < 4:
        return loop
    return _smooth_closed_polyline(loop)


def _decimate_indices(n: int, max_control: int) -> np.ndarray:
    if n <= max_control:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, max_control, dtype=int))


def _two_control_points_for_contour(
    xyz: np.ndarray, path_pos: np.ndarray
) -> np.ndarray:
    """
    Two world-space control points for ``type="Contour"`` (threshold line).

    Uses the contour vertex closest to the path ``pos`` and the vertex farthest
    from ``pos``. If those coincide (e.g. symmetric loop), falls back to
    vertices separated along the polyline order.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape[0] < 2:
        raise ValueError("Contour needs at least two points for control_points")
    p = np.asarray(path_pos, dtype=float).reshape(3)
    d = np.linalg.norm(xyz - p, axis=1)
    i0 = int(np.argmin(d))
    i1 = int(np.argmax(d))
    n = xyz.shape[0]
    if i1 == i0:
        i1 = (i0 + max(1, n // 2)) % n
    p0, p1 = xyz[i0].copy(), xyz[i1].copy()
    if float(np.linalg.norm(p1 - p0)) < 1e-9:
        i1 = (i0 + 1) % n
        p1 = xyz[i1]
    return np.stack([p0, p1], axis=0)


def _fmt_coord(x: float) -> str:
    return f"{float(x):.15g}"


def _append_path_point_xml(parent: ET.Element, pp: Dict[str, Any]) -> None:
    ppe = ET.SubElement(parent, "path_point", {"id": str(int(pp["id"]))})
    pos = pp["pos"]
    tan = pp["tangent"]
    rot = pp["rotation"]
    ET.SubElement(
        ppe,
        "pos",
        {"x": _fmt_coord(pos[0]), "y": _fmt_coord(pos[1]), "z": _fmt_coord(pos[2])},
    )
    ET.SubElement(
        ppe,
        "tangent",
        {"x": _fmt_coord(tan[0]), "y": _fmt_coord(tan[1]), "z": _fmt_coord(tan[2])},
    )
    ET.SubElement(
        ppe,
        "rotation",
        {"x": _fmt_coord(rot[0]), "y": _fmt_coord(rot[1]), "z": _fmt_coord(rot[2])},
    )


def _append_points_block(
    parent: ET.Element, tag: str, xyz: np.ndarray, max_points: Optional[int] = None
) -> None:
    block = ET.SubElement(parent, tag)
    n = xyz.shape[0]
    if max_points is not None and n > max_points:
        idx = _decimate_indices(n, max_points)
        pts = xyz[idx]
    else:
        pts = xyz
    for i in range(pts.shape[0]):
        x, y, z = pts[i]
        ET.SubElement(block, "point", {"id": str(i), "x": _fmt_coord(x), "y": _fmt_coord(y), "z": _fmt_coord(z)})


def write_ctgr(
    path_name: str,
    path_id: int,
    path_points: Sequence[Dict[str, Any]],
    contours_xyz: Sequence[Optional[np.ndarray]],
    output_path: str,
    *,
    reslice_size: str = "5",
    lofting_attribs: Optional[Dict[str, str]] = None,
    contour_method: str = "Threshold",
) -> str:
    """
    Write a SimVascular ``.ctgr`` contour group XML.

    ``path_points[i]`` must correspond to ``contours_xyz[i]`` (same length).
    ``contours_xyz[i]`` may be ``None`` to skip that slice.

    Each written slice uses ``type="Contour"`` with two ``control_points`` and
    full-resolution ``contour_points`` in world coordinates (native SimVascular
    threshold contour layout).
    """
    if len(path_points) != len(contours_xyz):
        raise ValueError("path_points and contours_xyz must have the same length")

    loft = dict(_DEFAULT_LOFTING_ATTRIBS)
    if lofting_attribs:
        loft.update(lofting_attribs)

    root = ET.Element(
        "contourgroup",
        {
            "path_name": path_name,
            "path_id": str(int(path_id)),
            "reslice_size": str(reslice_size),
            "point_2D_display_size": "",
            "point_size": "",
            "version": "1.0",
        },
    )
    ts = ET.SubElement(root, "timestep", {"id": "0"})
    ET.SubElement(ts, "lofting_parameters", loft)

    contour_id = 0
    for pp, xyz in zip(path_points, contours_xyz):
        if xyz is None or xyz.shape[0] < 3:
            continue
        co = ET.SubElement(
            ts,
            "contour",
            {
                "id": str(contour_id),
                "type": "Contour",
                "method": str(contour_method),
                "closed": "true",
                "min_control_number": "2",
                "max_control_number": "2",
                "subdivision_type": "0",
                "subdivision_number": "0",
                "subdivision_spacing": "0",
            },
        )
        contour_id += 1
        _append_path_point_xml(co, pp)
        ctrl2 = _two_control_points_for_contour(xyz, pp["pos"])
        _append_points_block(co, "control_points", ctrl2, max_points=None)
        _append_points_block(co, "contour_points", xyz, max_points=None)

    if contour_id == 0:
        raise ValueError(
            "No contours were written; check iso_value, volume overlap with the path, "
            "half_extent_mm, or stride."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="    ", level=0)
    except AttributeError:
        indent(root)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    return output_path


def build_contours_for_path_points(
    sitk_volume: sitk.Image,
    path_points: Sequence[Dict[str, Any]],
    *,
    iso_value: float = 0.5,
    half_extent_mm: float = 50.0,
    plane_spacing_mm: Optional[float] = None,
    stride: int = 1,
    component_selection: str = "closest_to_path",
) -> Tuple[List[Dict[str, Any]], List[Optional[np.ndarray]]]:
    """
    For each path point (optionally ``stride`` subsampled), extract a contour.

    Returns
    -------
    kept_path_points, contours
        Lists of equal length with contours aligned to kept path points.
    """
    if stride < 1:
        stride = 1

    sitk_volume = _as_float_probability_volume(sitk_volume)

    vtk_cache = exportSitk2VTK(sitk_volume, interpolation="linear")
    kept_pp: List[Dict[str, Any]] = []
    contours: List[Optional[np.ndarray]] = []

    for i, pp in enumerate(path_points):
        if i % stride != 0:
            continue
        loop = extract_plane_contour_world(
            sitk_volume,
            pp["pos"],
            pp["tangent"],
            pp["rotation"],
            iso_value=iso_value,
            half_extent_mm=half_extent_mm,
            plane_spacing_mm=plane_spacing_mm,
            vtk_cache=vtk_cache,
            component_selection=component_selection,
        )
        kept_pp.append(pp)
        contours.append(loop)

    return kept_pp, contours


def _as_float_probability_volume(vol: sitk.Image) -> sitk.Image:
    """Integer label maps cannot use a 0.5 isocontour; cast to float32."""
    pid = vol.GetPixelID()
    if pid in (sitk.sitkFloat32, sitk.sitkFloat64):
        return vol
    return sitk.Cast(vol, sitk.sitkFloat32)


def write_ctgr_for_pth(
    sitk_volume: sitk.Image,
    pth_path: str,
    output_ctgr_path: str,
    *,
    iso_value: float = 0.5,
    half_extent_mm: float = 50.0,
    plane_spacing_mm: Optional[float] = None,
    stride: int = 1,
    path_name: Optional[str] = None,
    contour_method: str = "Threshold",
    component_selection: str = "closest_to_path",
) -> str:
    """
    Parse ``pth_path``, extract contours from ``sitk_volume``, write ``.ctgr``.

    ``path_name`` defaults to the basename of the ``.pth`` file without extension.
    ``stride`` keeps every ``stride``-th path point (1 = all points).
    """
    meta = parse_pth(pth_path)
    pts = meta["path_points"]
    if not pts:
        raise ValueError(f"No path points parsed from {pth_path}")

    kept, contours = build_contours_for_path_points(
        sitk_volume,
        pts,
        iso_value=iso_value,
        half_extent_mm=half_extent_mm,
        plane_spacing_mm=plane_spacing_mm,
        stride=stride,
        component_selection=component_selection,
    )

    if path_name is None:
        path_name = os.path.splitext(os.path.basename(pth_path))[0]

    return write_ctgr(
        path_name,
        int(meta["path_id"]),
        kept,
        contours,
        output_ctgr_path,
        reslice_size="5",
        contour_method=contour_method,
    )


if __name__ == "__main__":
    import tempfile

    # Synthetic sphere probability map and short axial path (sanity check).
    dim = 80
    gx = np.arange(dim, dtype=np.float32) - (dim - 1) * 0.5
    gxx, gyy, gzz = np.meshgrid(gx, gx, gx, indexing="ij")
    r = np.sqrt(gxx * gxx + gyy * gyy + gzz * gzz)
    arr = np.clip(1.0 - r / 12.0, 0.0, 1.0).astype(np.float32)
    vol = sitk.GetImageFromArray(arr.transpose(2, 1, 0))
    vol.SetSpacing((1.0, 1.0, 1.0))
    # Align physical space with the grid used to build ``arr`` (sphere at world origin).
    half = (dim - 1) * 0.5
    vol.SetOrigin((-half, -half, -half))

    zvals = np.linspace(-20, 20, 9, dtype=float)
    ctrl = [(0.0, 0.0, float(z)) for z in zvals]
    ppath = sample_path_points_from_control_polyline(
        ctrl, spline_resample=True, calculation_number=80
    )
    kept, contours = build_contours_for_path_points(
        vol, ppath, iso_value=0.5, half_extent_mm=35.0, stride=1
    )
    ok = sum(1 for c in contours if c is not None)
    fd, out = tempfile.mkstemp(suffix=".ctgr")
    os.close(fd)
    try:
        write_ctgr("test_sphere", 1, kept, contours, out)
        print(f"wrote {out}, contours={ok}/{len(contours)}")
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass
