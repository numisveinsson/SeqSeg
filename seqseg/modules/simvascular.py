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


def create_pth(points, output_path):
    """Create a SimVascular-style .pth XML file given a list of (x,y,z) points."""
    path_elem = ET.Element(
        "path",
        attrib={
            "id": "1",
            "method": "0",
            "calculation_number": "100",
            "spacing": "0",
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
        {"method": "0", "calculation_number": "100", "spacing": "0"},
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
    tangents = compute_tangents(points)

    for i, ((x, y, z), t) in enumerate(zip(points, tangents)):
        path_point_elem = ET.SubElement(path_points_elem, "path_point", {"id": str(i)})

        # Position
        ET.SubElement(path_point_elem, "pos", {"x": f"{x}", "y": f"{y}", "z": f"{z}"})

        # Tangent
        ET.SubElement(
            path_point_elem,
            "tangent",
            {"x": f"{t[0]}", "y": f"{t[1]}", "z": f"{t[2]}"},
        )

        # Placeholder rotation (SimVascular will recompute)
        ET.SubElement(
            path_point_elem,
            "rotation",
            {"x": "0", "y": "1.0", "z": "0"},
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
        create_pth(branch_points, out_path)
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

    dir = '/Users/nsveinsson/Documents/datasets/CAS_coronary_dataset/1-200/centerlines_fmm_only_successful/'
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
