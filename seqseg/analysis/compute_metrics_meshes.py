"""Compute quality and smoothness metrics for surface meshes (.vtp) without ground truth.

Usage:
	python -m seqseg.analysis.compute_metrics_meshes /path/to/meshes_dir --out-csv results.csv

Example:
	python -m seqseg.analysis.compute_metrics_meshes /data/pred_vtps --out-csv mesh_quality.csv

Outputs metrics on mesh quality and smoothness:
	- Laplacian (smoothness): mean, std, max of |mean curvature| across vertices.
	  Lower values indicate smoother meshes. Uses VTK's vtkCurvatures (Mean curvature).
	- Curvature variation: mean local std of curvature among neighbors. Lower = smoother.
	- Aspect ratio: mean, std, max per-triangle aspect ratio (longest/shortest edge).
	  Closer to 1 = more regular triangles.
	- Manifold check: n_non_manifold_edges, is_manifold (True if no non-manifold edges).
	- Self-intersection: n_self_intersecting_triangles (requires trimesh; NaN if unavailable).
	- Basic mesh stats: n_vertices, n_cells, volume, surface_area

Requires `vtk` Python package. For self-intersection: `pip install trimesh`.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import List, Optional

try:
	import vtk
except Exception as e:
	raise ImportError(
		"The 'vtk' Python package is required to run this script. "
		"Install it with `pip install vtk` or `conda install -c conda-forge vtk`."
	) from e

import numpy as np
from vtk.util import numpy_support

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable, desc=None, disable=False, **kwargs):
		if disable:
			return iterable
		return iterable


def load_vtp(path: str) -> vtk.vtkPolyData:
	reader = vtk.vtkXMLPolyDataReader()
	reader.SetFileName(path)
	reader.Update()
	poly = reader.GetOutput()
	if poly is None:
		raise RuntimeError(f"Failed to read VTP: {path}")
	return poly


def laplacian_smoothness_metrics(poly: vtk.vtkPolyData) -> dict:
	"""Compute Laplacian-based smoothness metrics using mean curvature.

	Uses VTK's vtkCurvatures to compute mean curvature at each vertex. The mean curvature
	H is related to the Laplace-Beltrami operator: Δx = 2Hn. Lower |H| indicates
	smoother surface (flat regions have H=0).

	Returns
	-------
	dict
		Keys: 'laplacian_mean', 'laplacian_std', 'laplacian_max' (mean curvature magnitude).
		Lower values = smoother mesh. Units: 1/length (e.g. 1/mm for mm coordinates).
	"""
	if poly is None or poly.GetNumberOfPoints() == 0 or poly.GetNumberOfCells() == 0:
		return {
			'laplacian_mean': float('nan'),
			'laplacian_std': float('nan'),
			'laplacian_max': float('nan'),
		}
	try:
		curvatures = vtk.vtkCurvatures()
		curvatures.SetInputData(poly)
		curvatures.SetCurvatureTypeToMean()
		curvatures.Update()
		out = curvatures.GetOutput()
		mean_curv = out.GetPointData().GetArray("Mean_Curvature")
		if mean_curv is None:
			return {
				'laplacian_mean': float('nan'),
				'laplacian_std': float('nan'),
				'laplacian_max': float('nan'),
			}
		arr = numpy_support.vtk_to_numpy(mean_curv)
		abs_curv = np.abs(arr)
		valid = np.isfinite(abs_curv)
		if not np.any(valid):
			return {
				'laplacian_mean': float('nan'),
				'laplacian_std': float('nan'),
				'laplacian_max': float('nan'),
			}
		a = abs_curv[valid]
		n_valid = a.size
		std_val = float(np.std(a)) if n_valid >= 2 else float('nan')
		return {
			'laplacian_mean': float(np.mean(a)),
			'laplacian_std': std_val,
			'laplacian_max': float(np.max(a)),
		}
	except Exception:
		return {
			'laplacian_mean': float('nan'),
			'laplacian_std': float('nan'),
			'laplacian_max': float('nan'),
		}


def mesh_volume(poly: vtk.vtkPolyData) -> float:
	"""Volume enclosed by a closed triangulated surface (same units as mesh coordinates, e.g. mm³)."""
	if poly is None or poly.GetNumberOfCells() == 0:
		return float('nan')
	try:
		mass = vtk.vtkMassProperties()
		mass.SetInputData(poly)
		mass.Update()
		return float(mass.GetVolume())
	except Exception:
		return float('nan')


def mesh_surface_area(poly: vtk.vtkPolyData) -> float:
	"""Surface area of a closed triangulated mesh (same units as mesh coordinates squared, e.g. mm²)."""
	if poly is None or poly.GetNumberOfCells() == 0:
		return float('nan')
	try:
		mass = vtk.vtkMassProperties()
		mass.SetInputData(poly)
		mass.Update()
		return float(mass.GetSurfaceArea())
	except Exception:
		return float('nan')


def aspect_ratio_metrics(poly: vtk.vtkPolyData) -> dict:
	"""Compute aspect ratio (longest/shortest edge) per triangle. Closer to 1 = more regular.

	Uses vtkMeshQuality. Returns mean, std, max. Ideal triangles have aspect ratio 1.
	"""
	if poly is None or poly.GetNumberOfCells() == 0:
		return {'aspect_ratio_mean': float('nan'), 'aspect_ratio_std': float('nan'), 'aspect_ratio_max': float('nan')}
	try:
		quality = vtk.vtkMeshQuality()
		quality.SetInputData(poly)
		quality.SetTriangleQualityMeasureToAspectRatio()
		quality.SaveCellQualityOn()
		quality.Update()
		out = quality.GetOutput()
		arr = out.GetCellData().GetArray("Quality")
		if arr is None:
			arr = out.GetCellData().GetArray(0)
		if arr is None:
			return {'aspect_ratio_mean': float('nan'), 'aspect_ratio_std': float('nan'), 'aspect_ratio_max': float('nan')}
		a = numpy_support.vtk_to_numpy(arr).astype(float)
		valid = np.isfinite(a) & (a > 0)
		if not np.any(valid):
			return {'aspect_ratio_mean': float('nan'), 'aspect_ratio_std': float('nan'), 'aspect_ratio_max': float('nan')}
		a = a[valid]
		n_valid = a.size
		std_val = float(np.std(a)) if n_valid >= 2 else float('nan')
		return {
			'aspect_ratio_mean': float(np.mean(a)),
			'aspect_ratio_std': std_val,
			'aspect_ratio_max': float(np.max(a)),
		}
	except Exception:
		return {'aspect_ratio_mean': float('nan'), 'aspect_ratio_std': float('nan'), 'aspect_ratio_max': float('nan')}


def manifold_check(poly: vtk.vtkPolyData) -> dict:
	"""Check for non-manifold edges. Uses vtkFeatureEdges.

	Returns n_non_manifold_edges (count) and is_manifold (True if count == 0).
	"""
	if poly is None:
		return {'n_non_manifold_edges': -1, 'is_manifold': False}
	try:
		edges = vtk.vtkFeatureEdges()
		edges.SetInputData(poly)
		edges.BoundaryEdgesOff()
		edges.FeatureEdgesOff()
		edges.ManifoldEdgesOff()
		edges.NonManifoldEdgesOn()
		edges.Update()
		out = edges.GetOutput()
		n_edges = out.GetNumberOfCells() if out else 0
		return {
			'n_non_manifold_edges': n_edges,
			'is_manifold': n_edges == 0,
		}
	except Exception:
		return {'n_non_manifold_edges': -1, 'is_manifold': False}


def self_intersection_count(poly: vtk.vtkPolyData) -> float:
	"""Count triangles that improperly intersect other parts of the mesh.

	Requires trimesh. Returns nan if trimesh is not available.
	"""
	try:
		import trimesh
	except ImportError:
		return float('nan')
	if poly is None or poly.GetNumberOfCells() == 0:
		return float('nan')
	try:
		pts = poly.GetPoints()
		n_pts = pts.GetNumberOfPoints()
		points = np.array([pts.GetPoint(i) for i in range(n_pts)])
		cells = poly.GetPolys()
		cells.InitTraversal()
		faces = []
		id_list = vtk.vtkIdList()
		while cells.GetNextCell(id_list):
			if id_list.GetNumberOfIds() == 3:
				faces.append([id_list.GetId(j) for j in range(3)])
		faces = np.array(faces, dtype=np.int64) if faces else np.empty((0, 3))
		tm = trimesh.Trimesh(vertices=points, faces=faces)
		inter = trimesh.intersections.mesh_self_intersection(tm)
		return float(len(inter)) if inter is not None else 0.0
	except Exception:
		return float('nan')


def curvature_variation_metrics(poly: vtk.vtkPolyData) -> dict:
	"""Compute curvature variation: mean local std of curvature among vertex neighbors.

	For each vertex, compute std of mean curvature at that vertex and its neighbors.
	Higher values indicate more variation (less smooth). Requires mean curvature from vtkCurvatures.
	"""
	if poly is None or poly.GetNumberOfPoints() == 0 or poly.GetNumberOfCells() == 0:
		return {'curvature_variation_mean': float('nan'), 'curvature_variation_std': float('nan'), 'curvature_variation_max': float('nan')}
	try:
		curvatures = vtk.vtkCurvatures()
		curvatures.SetInputData(poly)
		curvatures.SetCurvatureTypeToMean()
		curvatures.Update()
		out = curvatures.GetOutput()
		mean_curv = out.GetPointData().GetArray("Mean_Curvature")
		if mean_curv is None:
			return {'curvature_variation_mean': float('nan'), 'curvature_variation_std': float('nan'), 'curvature_variation_max': float('nan')}
		curv_arr = numpy_support.vtk_to_numpy(mean_curv).astype(float)
		n_pts = curv_arr.size

		# Build point-to-neighbors via cells
		neighbors = [[] for _ in range(n_pts)]
		cells = poly.GetPolys()
		cells.InitTraversal()
		id_list = vtk.vtkIdList()
		while cells.GetNextCell(id_list):
			if id_list.GetNumberOfIds() == 3:
				a, b, c = id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)
				for p, q in [(a, b), (a, c), (b, a), (b, c), (c, a), (c, b)]:
					if q not in neighbors[p]:
						neighbors[p].append(q)

		local_variations = []
		for i in range(n_pts):
			neigh = [i] + neighbors[i]
			vals = curv_arr[neigh]
			vals = vals[np.isfinite(vals)]
			if vals.size >= 2:
				local_variations.append(float(np.std(vals)))
		if not local_variations:
			return {'curvature_variation_mean': float('nan'), 'curvature_variation_std': float('nan'), 'curvature_variation_max': float('nan')}
		arr = np.array(local_variations)
		n_valid = arr.size
		std_val = float(np.std(arr)) if n_valid >= 2 else float('nan')
		return {
			'curvature_variation_mean': float(np.mean(arr)),
			'curvature_variation_std': std_val,
			'curvature_variation_max': float(np.max(arr)),
		}
	except Exception:
		return {'curvature_variation_mean': float('nan'), 'curvature_variation_std': float('nan'), 'curvature_variation_max': float('nan')}


def compute_metrics(poly: vtk.vtkPolyData) -> dict:
	"""Compute quality and smoothness metrics for a single mesh (no ground truth)."""
	n_pts = poly.GetNumberOfPoints() if poly else 0
	n_cells = poly.GetNumberOfCells() if poly else 0

	result = {
		'n_vertices': n_pts,
		'n_cells': n_cells,
		'volume': mesh_volume(poly),
		'surface_area': mesh_surface_area(poly),
		**laplacian_smoothness_metrics(poly),
	}
	return result


RESULTS_FIELDNAMES = [
	'case',
	'n_vertices',
	'n_cells',
	'volume',
	'surface_area',
	'laplacian_mean',
	'laplacian_std',
	'laplacian_max',
]


def find_vtps_in_dir(mesh_dir: str, ext: str = '.vtp') -> List[tuple]:
	"""Return list of (case_name, path) for .vtp files in directory."""
	files = sorted(f for f in os.listdir(mesh_dir) if f.lower().endswith(ext))
	return [(os.path.splitext(f)[0], os.path.join(mesh_dir, f)) for f in files]


def write_results_csv(out_csv: str, rows: List[dict]) -> None:
	fieldnames = RESULTS_FIELDNAMES
	with open(out_csv, 'w', newline='') as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in rows:
			w.writerow({k: r.get(k, '') for k in fieldnames})

		if rows:
			mean_row = {'case': 'MEAN'}
			std_row = {'case': 'STD'}
			for k in fieldnames[1:]:
				values = []
				for r in rows:
					v = r.get(k, None)
					if isinstance(v, (int, float)) and not math.isnan(v):
						values.append(float(v))
				if values:
					arr = np.asarray(values, dtype=float)
					mean_row[k] = float(np.mean(arr))
					std_row[k] = float(np.std(arr)) if arr.size >= 2 else float('nan')
				else:
					mean_row[k] = ''
					std_row[k] = ''

			w.writerow(mean_row)
			w.writerow(std_row)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description='Compute quality and smoothness metrics for .vtp meshes (no ground truth)'
	)
	p.add_argument('mesh_dir', help='Directory containing .vtp mesh files')
	p.add_argument('--out-csv', default='mesh_quality.csv', help='Output CSV path')
	p.add_argument('--ext', default='.vtp', help='File extension to look for (default: .vtp)')
	p.add_argument('--quiet', action='store_true', help='Reduce logging')
	return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	mesh_dir = args.mesh_dir
	out_csv = args.out_csv
	ext = args.ext if args.ext.startswith('.') else '.' + args.ext

	if not os.path.isdir(mesh_dir):
		print(f'Error: Mesh directory does not exist: {mesh_dir}', file=sys.stderr)
		return 1

	cases = find_vtps_in_dir(mesh_dir, ext=ext)
	if not cases:
		print(f'No {ext} files found in {mesh_dir}', file=sys.stderr)
		return 2

	rows = []
	for case, path in tqdm(cases, desc='Meshes', disable=args.quiet):
		try:
			poly = load_vtp(path)
			metrics = compute_metrics(poly)
			row = {'case': case, **metrics}
			rows.append(row)
			if not args.quiet:
				lap = metrics.get('laplacian_mean', float('nan'))
				vol = metrics.get('volume', float('nan'))
				msg = f"  {case}: laplacian_mean={lap:.6g}"
				if np.isfinite(vol):
					msg += f", volume={vol:.6g}"
				print(msg)
		except Exception as e:
			print(f'Error processing {case}: {e}', file=sys.stderr)

	write_results_csv(out_csv, rows)

	lap_list = [r['laplacian_mean'] for r in rows if not math.isnan(r.get('laplacian_mean', math.nan))]
	if lap_list:
		print(f'Average laplacian_mean over {len(lap_list)} meshes: {float(np.mean(lap_list)):.6g}')
	print(f'Wrote results to {out_csv}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
