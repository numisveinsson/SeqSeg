"""
Standalone CLI for Taubin smoothing of VTK PolyData surfaces.

Example:
    python -m seqseg.scripts.taubin_smooth \
        --in input_surface.vtp \
        --out output_surface_smooth.vtp
"""

import argparse
from pathlib import Path

import vtk
from seqseg.modules import vtk_functions as vf


def force_recompute_point_normals(polydata):
    """Drop existing normals arrays, then recompute from current geometry."""
    cleaned = vtk.vtkPolyData()
    cleaned.DeepCopy(polydata)

    point_data = cleaned.GetPointData()
    if point_data is not None:
        # Remove active normals and any arrays named like normal(s).
        point_data.SetNormals(None)
        remove_names = []
        for idx in range(point_data.GetNumberOfArrays()):
            arr = point_data.GetArray(idx)
            if arr is None:
                continue
            arr_name = arr.GetName() or ""
            if "normal" in arr_name.lower():
                remove_names.append(arr_name)
        for arr_name in remove_names:
            point_data.RemoveArray(arr_name)

    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(cleaned)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOff()
    normals_filter.AutoOrientNormalsOn()
    normals_filter.ConsistencyOn()
    normals_filter.SplittingOff()
    normals_filter.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(normals_filter.GetOutput())
    out.GetPointData().SetActiveNormals("Normals")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply Taubin smoothing to a .vtp/.vtk surface mesh."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        required=True,
        help="Input surface path (.vtp or .vtk).",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        required=True,
        help="Output smoothed surface path (.vtp or .vtk).",
    )
    parser.add_argument(
        "--it",
        type=int,
        default=75,
        help="Number of Taubin iterations (default: 75).",
    )
    parser.add_argument(
        "--mu1",
        type=float,
        default=0.5,
        help="Taubin smoothing coefficient mu1 (default: 0.5).",
    )
    parser.add_argument(
        "--mu2",
        type=float,
        default=0.51,
        help="Taubin inflation coefficient mu2 (default: 0.51).",
    )
    parser.add_argument(
        "--skip-normals",
        action="store_true",
        help="Skip recomputing surface normals after smoothing.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    reader = vf.read_geo(str(input_path))
    poly = reader.GetOutput()

    poly_smooth = vf.taubin_smooth_polydata(
        poly,
        it=args.it,
        mu1=args.mu1,
        mu2=args.mu2,
    )

    if not args.skip_normals:
        # Ensure triangle topology and force a fresh normals pass.
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(poly_smooth)
        tri.Update()
        poly_smooth = force_recompute_point_normals(tri.GetOutput())

        normals = poly_smooth.GetPointData().GetNormals()
        if normals is None:
            raise RuntimeError(
                "Normals recomputation failed: output has no point normals."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf.write_vtk_polydata(poly_smooth, str(output_path))

    normals = poly_smooth.GetPointData().GetNormals()
    normals_status = "present" if normals is not None else "missing"

    print(
        "Saved smoothed surface to "
        f"{output_path} (it={args.it}, mu1={args.mu1}, mu2={args.mu2}, "
        f"point_normals={normals_status})"
    )


if __name__ == "__main__":
    main()
