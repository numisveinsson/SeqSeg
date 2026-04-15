import math
import xml.etree.ElementTree as ET
import numpy as np
import os
import vtk


def compute_tangents(points):
    """Compute approximate tangent vectors using central differences."""
    tangents = []
    n = len(points)
    for i in range(n):
        if i == 0:
            t = np.array(points[1]) - np.array(points[0])
        elif i == n - 1:
            t = np.array(points[-1]) - np.array(points[-2])
        else:
            t = np.array(points[i + 1]) - np.array(points[i - 1])
        t_norm = np.linalg.norm(t)
        t = t / t_norm if t_norm > 0 else np.zeros(3)
        tangents.append(t)
    return tangents


def _perpendicular_normal_unit(tangent, eps=1e-12):
    """Unit vector perpendicular to tangent (SimVascular uses cvMath::GetPerpendicularNormalVector)."""
    t = np.asarray(tangent, dtype=float).reshape(3)
    nrm = np.linalg.norm(t)
    if nrm < eps:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    t = t / nrm
    # Pick the cardinal axis least aligned with tangent to avoid bias
    # (e.g. always using x-axis can force one rotation component to 0).
    i = int(np.argmin(np.abs(t)))
    a = np.zeros(3, dtype=float)
    a[i] = 1.0
    r = np.cross(t, a)
    rn = np.linalg.norm(r)
    if rn < eps:
        for j in (0, 1, 2):
            if j == i:
                continue
            a = np.zeros(3, dtype=float)
            a[j] = 1.0
            r = np.cross(t, a)
            rn = np.linalg.norm(r)
            if rn >= eps:
                break
    return r / rn


def _vtk_parametric_spline_eval(spline, t):
    """Evaluate SimVascular-style vtkParametricSpline at parameter t (knot index space)."""
    pts = spline.GetPoints()
    n = pts.GetNumberOfPoints()
    if n == 0:
        return np.zeros(3, dtype=float)
    if spline.GetClosed():
        tmax = float(n)
    else:
        tmax = float(n - 1)
    t = max(0.0, min(float(t), tmax))
    u = [t, 0.0, 0.0]
    p = [0.0, 0.0, 0.0]
    # VTK Python requires a derivative buffer (unused here).
    du = [0.0] * 9
    spline.Evaluate(u, p, du)
    return np.array(p, dtype=float)


def _make_vtk_parametric_spline(control_points, closed=False):
    """vtkParametricSpline with vtkCardinalSpline components; ParameterizeByLengthOff like sv3::Spline."""
    pts = vtk.vtkPoints()
    for p in control_points:
        pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(pts)
    spline.ParameterizeByLengthOff()
    if closed:
        spline.ClosedOn()
    else:
        spline.ClosedOff()
    return spline


def resample_path_like_simvascular(
    control_points,
    calculation_number=100,
    further_subdivision_number=10,
    closed=False,
    spacing=None,
):
    """
    Resample a 3D polyline the same way SimVascular sv3::Spline does for paths.

    Uses VTK vtkParametricSpline (default: three vtkCardinalSpline in x,y,z),
    ParameterizeByLengthOff, and the CONSTANT_TOTAL_NUMBER / CONSTANT_SPACING
    subdivision rules from Code/Source/sv3/Path/sv3_Spline.cxx.

    Returns
    -------
    list of dict
        Each item has keys ``pos``, ``tangent``, ``rotation`` (length-3 float arrays).
    """
    if len(control_points) < 2:
        return []

    ctrl = [np.asarray(p, dtype=float).reshape(3) for p in control_points]
    spline = _make_vtk_parametric_spline(ctrl, closed=closed)
    input_point_number = len(ctrl)
    method_total = spacing is None
    method_spacing = spacing is not None

    inter_number = 1
    if method_spacing:
        m_spacing = float(spacing)
    else:
        if closed:
            inter_number = max(
                1, int(math.ceil((calculation_number * 1.0) / input_point_number))
            )
        else:
            inter_number = max(
                1,
                int(
                    math.ceil(
                        (calculation_number - 1.0) / (input_point_number - 1.0)
                    )
                ),
            )

    spline_points = []
    spline_point_id = 0
    fsd = max(1, int(further_subdivision_number))

    for i in range(input_point_number):
        pt1 = ctrl[i]

        if method_spacing:
            if i < input_point_number - 1 or closed:
                seg_len = 0.0
                subdiv = 10
                interval = 1.0 / subdiv
                for k in range(subdiv):
                    t1 = i + k * interval
                    t2 = i + (k + 1) * interval
                    p1 = _vtk_parametric_spline_eval(spline, t1)
                    p2 = _vtk_parametric_spline_eval(spline, t2)
                    seg_len += float(np.linalg.norm(p2 - p1))
                inter_number = max(5, int(math.ceil(seg_len / m_spacing)))

        spline_point = {"id": spline_point_id}

        if i == input_point_number - 1 and not closed:
            tx = i - 1.0 / inter_number / fsd
            ptx = _vtk_parametric_spline_eval(spline, tx)
            tan = pt1 - ptx
            ln = np.linalg.norm(tan)
            if ln > 1e-12:
                tan = tan / ln
            else:
                tan = np.array([0.0, 0.0, 1.0], dtype=float)
            spline_point["pos"] = pt1.copy()
            spline_point["tangent"] = tan
            spline_point["rotation"] = _perpendicular_normal_unit(tan)
            spline_points.append(spline_point)
            spline_point_id += 1
            break

        txx = i + 1.0 / inter_number / fsd
        ptx = _vtk_parametric_spline_eval(spline, txx)
        tan = ptx - pt1
        ln = np.linalg.norm(tan)
        if ln > 1e-12:
            tan = tan / ln
        else:
            tan = np.array([0.0, 0.0, 1.0], dtype=float)

        spline_point["pos"] = pt1.copy()
        spline_point["tangent"] = tan
        spline_point["rotation"] = _perpendicular_normal_unit(tan)
        spline_points.append(spline_point)
        spline_point_id += 1

        for j in range(1, inter_number):
            tnew = i + j * (1.0 / inter_number)
            tx = tnew + 1.0 / inter_number / fsd
            pt_mid = _vtk_parametric_spline_eval(spline, tnew)
            ptx2 = _vtk_parametric_spline_eval(spline, tx)
            tan2 = ptx2 - pt_mid
            ln2 = np.linalg.norm(tan2)
            if ln2 > 1e-12:
                tan2 = tan2 / ln2
            else:
                tan2 = np.array([0.0, 0.0, 1.0], dtype=float)
            spline_points.append(
                {
                    "id": spline_point_id,
                    "pos": pt_mid.copy(),
                    "tangent": tan2,
                    "rotation": _perpendicular_normal_unit(tan2),
                }
            )
            spline_point_id += 1

    return spline_points


def indent(elem, level=0):
    """Manual pretty-printer for XML (for Python < 3.9)."""
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for child in elem:
            indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def create_pth(
    points,
    output_path,
    path_id=1,
    spline_resample=False,
    calculation_number=100,
    spacing=0.0,
    further_subdivision_number=10,
):
    """Create a SimVascular-style .pth XML file given a list of (x,y,z) points.

    Parameters
    ----------
    points : list of (x, y, z)
        Centerline / path vertices. Written as ``control_points``.
    output_path : str
        Destination ``.pth`` file.
    path_id : int, optional
        XML ``<path id=...>``.
    spline_resample : bool, optional
        If True, ``path_points`` are generated with VTK ``vtkParametricSpline``
        (Cardinal splines per axis) using the same sampling idea as SimVascular
        ``sv3::Spline`` / ``PathElement::CreatePathPoints``. ``control_points``
        stay as ``points``. If False, ``path_points`` match ``points`` with
        polyline tangents (legacy SeqSeg behavior).
    calculation_number : int, optional
        Maps to SimVascular ``calculation_number`` (total path samples for
        CONSTANT_TOTAL_NUMBER when not using spacing).
    spacing : float, optional
        If > 0, uses CONSTANT_SPACING-style subdivision between controls (see
        SimVascular ``sv3_Spline.cxx``). When > 0, ``method`` is set to 2 in XML.
    further_subdivision_number : int, optional
        Matches SimVascular ``m_FurtherSubdivisionNumber`` for tangent fineness.
    """
    use_spacing = float(spacing) > 0.0
    method_enum = "2" if use_spacing else "0"
    calc_str = str(int(calculation_number))
    spacing_str = str(float(spacing))

    if spline_resample or use_spacing:
        sampled = resample_path_like_simvascular(
            points,
            calculation_number=int(calculation_number),
            further_subdivision_number=int(further_subdivision_number),
            closed=False,
            spacing=float(spacing) if use_spacing else None,
        )
    else:
        sampled = None

    path_elem = ET.Element(
        "path",
        attrib={
            "id": str(int(path_id)),
            "method": method_enum,
            "calculation_number": calc_str,
            "spacing": spacing_str,
            "version": "1.0",
            "reslice_size": "5",
            "point_2D_display_size": "",
            "point_size": "",
        },
    )

    timestep_elem = ET.SubElement(path_elem, "timestep", {"id": "0"})
    path_element_elem = ET.SubElement(
        timestep_elem,
        "path_element",
        {
            "method": method_enum,
            "calculation_number": calc_str,
            "spacing": spacing_str,
        },
    )

    # --- Control points ---
    control_points_elem = ET.SubElement(path_element_elem, "control_points")
    for i, (x, y, z) in enumerate(points):
        ET.SubElement(
            control_points_elem,
            "point",
            {"id": str(i), "x": f"{x}", "y": f"{y}", "z": f"{z}"},
        )

    # --- Path points ---
    path_points_elem = ET.SubElement(path_element_elem, "path_points")
    if sampled is not None:
        path_iter = [
            (row["pos"], row["tangent"], row["rotation"]) for row in sampled
        ]
    else:
        tangents = compute_tangents(points)
        path_iter = [
            (
                np.asarray([x, y, z], dtype=float),
                np.asarray(t, dtype=float),
                _perpendicular_normal_unit(t),
            )
            for (x, y, z), t in zip(points, tangents)
        ]

    for i, (pos, tan, rot) in enumerate(path_iter):
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        path_point_elem = ET.SubElement(path_points_elem, "path_point", {"id": str(i)})

        ET.SubElement(path_point_elem, "pos", {"x": f"{x}", "y": f"{y}", "z": f"{z}"})

        ET.SubElement(
            path_point_elem,
            "tangent",
            {"x": f"{tan[0]}", "y": f"{tan[1]}", "z": f"{tan[2]}"},
        )

        ET.SubElement(
            path_point_elem,
            "rotation",
            {"x": f"{rot[0]}", "y": f"{rot[1]}", "z": f"{rot[2]}"},
        )

    # --- Write to file ---
    tree = ET.ElementTree(path_elem)
    try:
        # Python 3.9+ only
        ET.indent(tree, space="    ", level=0)
    except AttributeError:
        indent(path_elem)

    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    print(f"✅ SimVascular .pth file written to {output_path}")


def write_pths_from_vtp(vtp_path, out_dir=None, prefix=None, verbose=True):
    """Read a centerline `.vtp` and write one SimVascular `.pth` per polyline branch.

    Parameters
    ----------
    vtp_path : str
        Path to the input `.vtp` file containing centerline polydata (lines).
    out_dir : str, optional
        Directory to write `.pth` files to. If None, uses the directory of `vtp_path`.
    prefix : str, optional
        Filename prefix for output files. If None, uses the base name of `vtp_path`.
    verbose : bool
        If True, prints progress messages.

    Returns
    -------
    written : list of str
        List of written `.pth` file paths.
    """
    if not os.path.exists(vtp_path):
        raise FileNotFoundError(f"Input VTP not found: {vtp_path}")

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(vtp_path))
    os.makedirs(out_dir, exist_ok=True)

    if prefix is None:
        prefix = os.path.splitext(os.path.basename(vtp_path))[0]

    # Read VTP
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    num_cells = polydata.GetNumberOfCells()
    pts = polydata.GetPoints()
    if verbose:
        print(f"Read VTP: {vtp_path} -> {num_cells} cells, {pts.GetNumberOfPoints()} points")

    written = []

    # Iterate cells; each cell that is a polyline becomes one .pth
    for ci in range(num_cells):
        cell = polydata.GetCell(ci)
        pid_list = cell.GetPointIds()
        n = pid_list.GetNumberOfIds()
        if n < 2:
            # skip degenerate
            if verbose:
                print(f"  Skipping cell {ci} (less than 2 points)")
            continue

        branch_points = []
        for pi in range(n):
            pid = pid_list.GetId(pi)
            xyz = pts.GetPoint(pid)
            branch_points.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))

        out_name = f"{prefix}_branch{ci+1}.pth"
        out_path = os.path.join(out_dir, out_name)
        create_pth(
            branch_points,
            out_path,
            path_id=ci + 1,
            spline_resample=True,
        )
        written.append(out_path)
        if verbose:
            print(f"  Wrote branch {ci+1}/{num_cells} -> {out_path}")

    if verbose:
        print(f"Done: wrote {len(written)} .pth files to {out_dir}")

    return written


def write_pths_from_dir(vtp_dir, out_dir=None, verbose=True):
    """Scan a directory for `.vtp` centerline files and write `.pth` files for each.

    Parameters
    ----------
    vtp_dir : str
        Directory containing `.vtp` files (non-recursive).
    out_dir : str, optional
        Directory to place generated `.pth` files. If None, a subdirectory
        named `<vtp_dir>_pths` will be created next to `vtp_dir`.
    verbose : bool
        If True, prints progress messages.

    Returns
    -------
    results : dict
        Mapping from input `.vtp` path to list of written `.pth` file paths.
    """
    vtp_dir = os.path.abspath(vtp_dir)
    if not os.path.isdir(vtp_dir):
        raise NotADirectoryError(f"Not a directory: {vtp_dir}")

    if out_dir is None:
        out_dir = vtp_dir.rstrip(os.sep) + '_pths'
    os.makedirs(out_dir, exist_ok=True)

    vtp_files = sorted([f for f in os.listdir(vtp_dir) if f.lower().endswith('.vtp')])
    if verbose:
        print(f"Found {len(vtp_files)} .vtp files in {vtp_dir}")

    results = {}
    for vtp_fn in vtp_files:
        vtp_path = os.path.join(vtp_dir, vtp_fn)
        prefix = os.path.splitext(vtp_fn)[0]
        if verbose:
            print(f"Processing: {vtp_path}")
        written = write_pths_from_vtp(vtp_path, out_dir=out_dir, prefix=prefix, verbose=verbose)
        results[vtp_path] = written

    if verbose:
        total = sum(len(v) for v in results.values())
        print(f"Wrote {total} .pth files for {len(results)} VTPs into {out_dir}")

    return results


if __name__ == "__main__":

    dir = '/Users/nsveinsson/Documents/datasets/CAS_coronary_dataset/1-200/centerlines_fmm_only_successful_scaled'
    out_dir = dir + '_pths'

    write_pths_from_dir(dir, out_dir=out_dir, verbose=True)

    # Example usage
    example_points = [
        (-2.6108, 12.7837, 166.0114),
        (-2.5841, 12.7916, 165.9611),
        (-2.5053, 12.8150, 165.8120),
        (-2.3787, 12.8530, 165.5696),
        (-2.2108, 12.9046, 165.2429),
        (-2.0102, 12.9681, 164.8432),
    ]

    create_pth(example_points, "example_path.pth")
