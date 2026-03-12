"""Compute quality and smoothness metrics for surface meshes (.vtp) without ground truth.

Usage:
	python -m seqseg.analysis.compute_metrics_meshes /path/to/meshes_dir --out-csv results.csv

Example:
	python -m seqseg.analysis.compute_metrics_meshes /data/pred_vtps --out-csv mesh_quality.csv

Outputs metrics on mesh quality and smoothness:
	- Laplacian (smoothness): mean, std, max of |mean curvature| across vertices.
	  Lower values indicate smoother meshes. Uses VTK's vtkCurvatures (Mean curvature).
	- Basic mesh stats: n_vertices, n_cells, volume, surface_area

Requires `vtk` Python package: `pip install vtk` (or `conda install -c conda-forge vtk`).
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
