"""Compute Hausdorff distance and ASSD between matching surface meshes (.vtp).

Usage:
	python -m seqseg.analysis.compute_metrics_meshes /path/to/gt_dir /path/to/pred_dir \
			--out-csv results.csv

Example:
	# basic run, write CSV
	python -m seqseg.analysis.compute_metrics_meshes /data/gt_vtps /data/pred_vtps --out-csv metrics.csv

	# compute Dice using full occupancy maps (1.0mm isotropic voxelization, no sampling);
	# --max-points limits sampling for Hausdorff/ASSD only
	python -m seqseg.analysis.compute_metrics_meshes /data/gt_vtps /data/pred_vtps \
			--out-csv metrics.csv --spacing 1.0 --max-points 5000

	# compute centerline overlap metric
	python -m seqseg.analysis.compute_metrics_meshes /data/gt_vtps /data/pred_vtps \
			--out-csv metrics.csv --centerline-dir /data/centerlines

	# clip meshes using centerlines before computing metrics
	python -m seqseg.analysis.compute_metrics_meshes /data/gt_vtps /data/pred_vtps \
			--out-csv metrics.csv --clip --centerline-dir /data/centerlines

	# clip meshes and save clipped meshes for debugging
	python -m seqseg.analysis.compute_metrics_meshes /data/gt_vtps /data/pred_vtps \
			--out-csv metrics.csv --clip --centerline-dir /data/centerlines \
			--clip-output-dir /data/clipped_meshes

	# multiple prediction folders: compare gt_dir to each subfolder of predictions_root, write summary.csv
	python -m seqseg.analysis.compute_metrics_meshes /data/gt_vtps /data/pred_dummy \
			--predictions-root /data/prediction_runs --summary-csv summary.csv

	# compute only specific metrics (faster when only some are needed)
	python -m seqseg.analysis.compute_metrics_meshes /data/gt_vtps /data/pred_vtps \
			--out-csv metrics.csv --metrics distance,dice
	python -m seqseg.analysis.compute_metrics_meshes /data/gt_vtps /data/pred_vtps \
			--out-csv metrics.csv --metrics volume

Assumes meshes have the same filenames in both directories (e.g. `case001.vtp`).
This script computes one-way point-to-surface distances by evaluating
the implicit distance from points in the GT mesh to the prediction mesh surface using
`vtkImplicitPolyDataDistance`. From these distances it computes:
	- Hausdorff distance (max distance from GT points to prediction surface)
	- HD95 (95th percentile distance from GT points to prediction surface)
	- ASSD (average surface distance from GT points to prediction surface)
	- Centerline overlap (ratio of centerline length within prediction surface, if --centerline-dir provided)
	- Volume error per radius bucket (volume error along centerline by MaximumInscribedSphereRadius, if --centerline-dir provided)
	- Dice per radius bucket (Dice along centerline by MaximumInscribedSphereRadius, if --centerline-dir provided)
	- Normal angular error (degrees): mean, std and max between GT and prediction normals at corresponding surface locations (one-way)
	- Volume and volume error: enclosed volume (e.g. mm³) per mesh, signed and relative error (requires closed triangulated meshes)
	
Note: This is a one-way metric (GT -> prediction only), not symmetric.

Units: Distance metrics (Hausdorff, HD95, ASSD) are computed in the same units as the
input mesh coordinates. For medical imaging data, meshes are typically in physical space
coordinates (millimeters), so distances are reported in millimeters. The units depend on
the coordinate system used when the meshes were created.

Notes:
- This operates on vertex points of the input `.vtp` meshes; for very dense
	meshes you may want to subsample using `--max-points`.
- Requires `vtk` Python package: `pip install vtk` (or `conda install -c conda-forge vtk`).
- Clipping (--clip) uses centerline-based clipping boxes to remove boundary regions
	before computing metrics, similar to the clipping functionality in compute_metrics.py.
	This requires centerline files with 'CenterlineId' and 'MaximumInscribedSphereRadius'
	point data arrays.
- Centerline overlap metric calculates the ratio of centerline length that is found within
	the prediction surface mesh, similar to the percent_centerline_length function in compute_metrics.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import math
from typing import List, Tuple, Optional

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

# Import clipping functions and volume-per-radii
try:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
	from modules import vtk_functions as vf
	from modules.capping import (
		bryan_get_clipping_parameters,
		bryan_generate_oriented_boxes,
		bryan_clip_surface
	)
	from analysis.volume_per_radii import (
		volume_error_per_radii,
		get_volume_per_radii_fieldnames,
		DEFAULT_RADIUS_BUCKETS,
	)
	from analysis.dice_per_radii import (
		dice_per_radii,
		get_dice_per_radii_fieldnames,
	)
except ImportError:
	# Clipping functionality will not be available if imports fail
	vf = None
	bryan_get_clipping_parameters = None
	bryan_generate_oriented_boxes = None
	bryan_clip_surface = None
	volume_error_per_radii = None
	get_volume_per_radii_fieldnames = None
	dice_per_radii = None
	get_dice_per_radii_fieldnames = None
	DEFAULT_RADIUS_BUCKETS = None


def load_vtp(path: str) -> vtk.vtkPolyData:
	reader = vtk.vtkXMLPolyDataReader()
	reader.SetFileName(path)
	reader.Update()
	poly = reader.GetOutput()
	if poly is None:
		raise RuntimeError(f"Failed to read VTP: {path}")
	return poly


def write_vtp(path: str, poly: vtk.vtkPolyData) -> None:
	"""Write a vtkPolyData to a .vtp file."""
	writer = vtk.vtkXMLPolyDataWriter()
	writer.SetFileName(path)
	writer.SetInputData(poly)
	writer.Write()


def clip_surface_mesh(surface: vtk.vtkPolyData, centerline: vtk.vtkPolyData, case_name: str = None, temp_dir: Optional[str] = None) -> vtk.vtkPolyData:
	"""Clip a surface mesh using centerline-based clipping boxes.
	
	This function applies the same clipping logic as in compute_metrics.py:
	1. Gets clipping parameters from centerline (endpoints, radii, unit vectors)
	2. Generates oriented boxes for clipping
	3. Clips the surface using those boxes
	4. Keeps only the largest connected component
	
	Parameters
	----------
	surface : vtk.vtkPolyData
		Surface mesh to clip
	centerline : vtk.vtkPolyData
		Centerline with clipping parameters
	case_name : str, optional
		Case name for temporary file naming (if temp_dir provided)
	temp_dir : str, optional
		Directory for temporary clipping box files (optional)
	
	Returns
	-------
	vtk.vtkPolyData
		Clipped surface mesh (largest connected component only)
	"""
	if vf is None or bryan_get_clipping_parameters is None:
		raise ImportError(
			"Clipping functionality requires modules.vtk_functions and modules.capping. "
			"Make sure these modules are available."
		)
	
	# Get clipping parameters from centerline
	endpts, radii, unit_vecs = bryan_get_clipping_parameters(centerline)
	
	# Generate oriented boxes for clipping
	# Use a temporary directory or current directory if not provided
	if temp_dir is None:
		temp_dir = os.path.dirname(os.path.abspath(__file__))
	os.makedirs(temp_dir, exist_ok=True)
	
	box_name = f"{case_name}_boxclips" if case_name else "boxclips"
	boxpd, _ = bryan_generate_oriented_boxes(endpts, unit_vecs, radii, box_name, temp_dir, 4)
	
	# Clip the surface
	clippedpd = bryan_clip_surface(surface, boxpd)
	
	# Keep only the largest connected component
	largest = vf.get_largest_connected_polydata(clippedpd)
	
	return largest


def sample_indices(num: int, max_points: Optional[int]) -> np.ndarray:
	if max_points is None or num <= max_points:
		return np.arange(num, dtype=np.int64)
	# random sample without replacement
	return np.random.choice(num, size=max_points, replace=False)


def point_coords_as_numpy(poly: vtk.vtkPolyData, indices: Optional[np.ndarray] = None) -> np.ndarray:
	pts = poly.GetPoints()
	n = pts.GetNumberOfPoints()
	if indices is None:
		indices = np.arange(n, dtype=np.int64)
	coords = np.empty((indices.shape[0], 3), dtype=float)
	for i, idx in enumerate(indices):
		coords[i, :] = pts.GetPoint(int(idx))
	return coords


def is_point_in_surface(surface: vtk.vtkPolyData, point: Tuple[float, float, float]) -> bool:
	"""Check if a point is inside a closed surface mesh.
	
	Uses signed distance from implicit function, which is more robust than
	vtkSelectEnclosedPoints for surfaces that may not be perfectly closed.
	
	Parameters
	----------
	surface : vtk.vtkPolyData
		Closed surface mesh
	point : tuple of 3 floats
		Point coordinates (x, y, z)
	
	Returns
	-------
	bool
		True if point is inside the surface, False otherwise
	"""
	# Use signed distance from implicit function (more robust than vtkSelectEnclosedPoints)
	# Negative distance means inside, positive means outside
	# This method works even if the surface is not perfectly closed
	try:
		implicit = vtk.vtkImplicitPolyDataDistance()
		implicit.SetInput(surface)
		signed_dist = implicit.EvaluateFunction(point)
		# Negative signed distance indicates the point is inside
		# Use a small tolerance to handle numerical precision issues
		return signed_dist < 1e-6
	except Exception:
		# If evaluation fails, return False (conservative approach)
		# This means we assume the point is outside if we can't determine
		return False


def percent_centerline_length_mesh(surface: vtk.vtkPolyData, centerline: vtk.vtkPolyData) -> float:
	"""Calculate the ratio of centerline length that is found within the surface mesh.
	
	This function iterates through all cells in the centerline, calculates the length
	of segments between consecutive points, and checks if both endpoints of each segment
	are within the surface mesh. Returns the ratio of centerline length inside the surface
	to total centerline length.
	
	Parameters
	----------
	surface : vtk.vtkPolyData
		Surface mesh (segmentation)
	centerline : vtk.vtkPolyData
		Centerline polydata
	
	Returns
	-------
	float
		Ratio of centerline length within surface (0-1), or NaN if centerline is empty
	"""
	num_cells = centerline.GetNumberOfCells()
	if num_cells == 0:
		return float('nan')
	
	# Create implicit distance function once for the entire surface (much faster)
	# This is the key optimization - we create it once and reuse it for all points
	try:
		implicit = vtk.vtkImplicitPolyDataDistance()
		implicit.SetInput(surface)
	except Exception:
		return float('nan')
	
	centerline_length = 0.0
	cent_length_in = 0.0
	points_checked = set()  # Use set for O(1) lookup instead of list
	
	# Iterate through all cells in centerline
	for i in range(num_cells):
		cell = centerline.GetCell(i)
		if cell is None:
			continue
		num_points = cell.GetNumberOfPoints()
		if num_points < 2:
			continue
		
		points = cell.GetPoints()
		length = 0.0
		length_in = 0.0
		
		# Calculate length of segments between consecutive points
		for j in range(num_points - 1):
			p1 = points.GetPoint(j)
			p2 = points.GetPoint(j + 1)
			
			# Convert to tuple for set membership check
			p2_tuple = tuple(p2)
			
			# Skip if we've already checked this point (p2 was already processed as p1 in previous iteration)
			if p2_tuple in points_checked:
				continue
			
			# Calculate segment length
			segment_length = np.linalg.norm(np.array(p1) - np.array(p2))
			length += segment_length
			
			# Check if both endpoints are within the surface using the pre-created implicit function
			try:
				# Evaluate signed distance: negative = inside, positive = outside
				signed_dist_p1 = implicit.EvaluateFunction(p1)
				signed_dist_p2 = implicit.EvaluateFunction(p2)
				
				# Use small tolerance for numerical precision
				p1_inside = signed_dist_p1 < 1e-6
				p2_inside = signed_dist_p2 < 1e-6
				
				if p1_inside and p2_inside:
					length_in += segment_length
			except Exception:
				# If point check fails, exclude this segment from total length
				length -= segment_length
			
			points_checked.add(tuple(p1))
		
		centerline_length += length
		cent_length_in += length_in
	
	if centerline_length == 0:
		return float('nan')
	
	return cent_length_in / centerline_length


def distances_pointset_to_surface(source: vtk.vtkPolyData, target: vtk.vtkPolyData, max_points: Optional[int] = None) -> np.ndarray:
	"""Compute unsigned distances from each (sampled) point of `source` to surface `target`.

	Returns a numpy array of distances (float) for the sampled source points.
	"""
	pts = source.GetPoints()
	if pts is None:
		return np.array([], dtype=float)
	n = pts.GetNumberOfPoints()
	if n == 0:
		return np.array([], dtype=float)

	indices = sample_indices(n, max_points)
	# implicit distance evaluator
	implicit = vtk.vtkImplicitPolyDataDistance()
	implicit.SetInput(target)

	dists = np.empty(indices.shape[0], dtype=float)
	for i, idx in enumerate(indices):
		x, y, z = pts.GetPoint(int(idx))
		val = implicit.EvaluateFunction((x, y, z))
		# EvaluateFunction may return signed distance; take absolute
		dists[i] = abs(float(val))
	return dists


def _polydata_with_normals(poly: vtk.vtkPolyData, point_normals: bool = True, cell_normals: bool = True) -> vtk.vtkPolyData:
	"""Return a copy of polydata with point and/or cell normals computed if missing."""
	out = vtk.vtkPolyData()
	out.DeepCopy(poly)
	need_point = point_normals and (out.GetPointData().GetNormals() is None)
	need_cell = cell_normals and (out.GetCellData().GetNormals() is None)
	if not need_point and not need_cell:
		return out
	normals_filter = vtk.vtkPolyDataNormals()
	normals_filter.SetInputData(out)
	normals_filter.ComputePointNormalsOn() if point_normals else normals_filter.ComputePointNormalsOff()
	normals_filter.ComputeCellNormalsOn() if cell_normals else normals_filter.ComputeCellNormalsOff()
	normals_filter.Update()
	return normals_filter.GetOutput()


def normal_angular_error_gt_to_pred(
	gt_poly: vtk.vtkPolyData,
	pred_poly: vtk.vtkPolyData,
	max_points: Optional[int] = None,
) -> dict:
	"""Mean, std and max angular error (degrees) between GT normals and prediction normals at corresponding locations.

	For each sampled GT point, finds the closest point on the prediction surface and compares
	the GT point normal to the prediction face normal at that location. Uses the same point
	sampling as distance metrics (sample_indices). One-way: GT -> prediction only.

	Returns
	-------
	dict
		Keys: 'mean', 'std', 'max' (degrees). std is nan if fewer than 2 valid points.
	"""
	gt_with_normals = _polydata_with_normals(gt_poly, point_normals=True, cell_normals=False)
	pred_with_normals = _polydata_with_normals(pred_poly, point_normals=False, cell_normals=True)

	pts = gt_with_normals.GetPoints()
	if pts is None:
		return {'mean': float('nan'), 'std': float('nan'), 'max': float('nan')}
	n = pts.GetNumberOfPoints()
	if n == 0:
		return {'mean': float('nan'), 'std': float('nan'), 'max': float('nan')}

	gt_normals_arr = gt_with_normals.GetPointData().GetNormals()
	if gt_normals_arr is None:
		return {'mean': float('nan'), 'std': float('nan'), 'max': float('nan')}
	gt_normals = numpy_support.vtk_to_numpy(gt_normals_arr)

	pred_cell_normals_arr = pred_with_normals.GetCellData().GetNormals()
	if pred_cell_normals_arr is None:
		return {'mean': float('nan'), 'std': float('nan'), 'max': float('nan')}
	pred_cell_normals = numpy_support.vtk_to_numpy(pred_cell_normals_arr)

	locator = vtk.vtkCellLocator()
	locator.SetDataSet(pred_with_normals)
	locator.BuildLocator()

	# vtk.reference (VTK 7) vs vtk.mutable (VTK 8+) for FindClosestPoint output args
	vtk_ref = getattr(vtk, 'mutable', None) or getattr(vtk, 'reference', None)
	if vtk_ref is None:
		raise RuntimeError('VTK provides neither vtk.mutable nor vtk.reference; cannot use vtkCellLocator.FindClosestPoint')
	indices = sample_indices(n, max_points)
	angles_deg = np.empty(indices.shape[0], dtype=float)
	closest_point = [0.0, 0.0, 0.0]
	cell_id_ref = vtk_ref(0)
	sub_id_ref = vtk_ref(0)
	dist2_ref = vtk_ref(0.0)
	for i, idx in enumerate(indices):
		idx = int(idx)
		x, y, z = pts.GetPoint(idx)
		locator.FindClosestPoint([x, y, z], closest_point, cell_id_ref, sub_id_ref, dist2_ref)
		cell_id = cell_id_ref.get()
		if cell_id < 0:
			angles_deg[i] = float('nan')
			continue
		n_gt = gt_normals[idx]
		n_pred = pred_cell_normals[cell_id]
		dot = np.clip(float(np.dot(n_gt, n_pred)), -1.0, 1.0)
		angles_deg[i] = math.degrees(math.acos(dot))
	valid = np.isfinite(angles_deg)
	if not np.any(valid):
		return {'mean': float('nan'), 'std': float('nan'), 'max': float('nan')}
	a = angles_deg[valid]
	n_valid = a.size
	std_val = float(np.std(a)) if n_valid >= 2 else float('nan')
	return {'mean': float(np.mean(a)), 'std': std_val, 'max': float(np.max(a))}


def mesh_volume(poly: vtk.vtkPolyData) -> float:
	"""Volume enclosed by a closed triangulated surface (same units as mesh coordinates, e.g. mm³).

	Uses vtkMassProperties. Returns nan if the mesh is not closed, has no cells, or computation fails.
	"""
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
	"""Surface area of a closed triangulated mesh (same units as mesh coordinates squared, e.g. mm²).

	Uses vtkMassProperties. Returns nan if the mesh has no cells or computation fails.
	"""
	if poly is None or poly.GetNumberOfCells() == 0:
		return float('nan')
	try:
		mass = vtk.vtkMassProperties()
		mass.SetInputData(poly)
		mass.Update()
		return float(mass.GetSurfaceArea())
	except Exception:
		return float('nan')


def volume_error_metrics(gt_poly: vtk.vtkPolyData, pred_poly: vtk.vtkPolyData) -> dict:
	"""Volume of each mesh and signed/relative error between them.

	Returns
	-------
	dict
		Keys: 'volume_gt', 'volume_pred', 'volume_error_abs' (pred - gt), 'volume_error_rel' ((pred - gt) / gt, nan if gt == 0).
	"""
	v_gt = mesh_volume(gt_poly)
	v_pred = mesh_volume(pred_poly)
	err_abs = float('nan')
	err_rel = float('nan')
	if np.isfinite(v_gt) and np.isfinite(v_pred):
		err_abs = v_pred - v_gt
		if v_gt != 0:
			err_rel = err_abs / v_gt
	return {
		'volume_gt': v_gt,
		'volume_pred': v_pred,
		'volume_error_abs': err_abs,
		'volume_error_rel': err_rel,
	}


def surface_area_error_metrics(gt_poly: vtk.vtkPolyData, pred_poly: vtk.vtkPolyData) -> dict:
	"""Surface area of each mesh and signed/relative error between them."""
	a_gt = mesh_surface_area(gt_poly)
	a_pred = mesh_surface_area(pred_poly)
	err_abs = float('nan')
	err_rel = float('nan')
	if np.isfinite(a_gt) and np.isfinite(a_pred):
		err_abs = a_pred - a_gt
		if a_gt != 0:
			err_rel = err_abs / a_gt
	return {
		'surface_area_gt': a_gt,
		'surface_area_pred': a_pred,
		'surface_area_error_abs': err_abs,
		'surface_area_error_rel': err_rel,
	}


def surface_dice(
	gt_poly: vtk.vtkPolyData,
	pred_poly: vtk.vtkPolyData,
	tolerance: float,
	max_points: Optional[int] = None,
) -> float:
	"""Surface Dice between GT and prediction at a given tolerance.

	Uses the standard definition:
		(|{p in S_gt: d(p,S_pred)<=τ}| + |{q in S_pred: d(q,S_gt)<=τ}|) / (|S_gt| + |S_pred|)

	Here |·| is approximated by uniform point sampling on each surface, using the same
	sampling strategy as other distance metrics (sample_indices).
	"""
	if tolerance < 0:
		raise ValueError('tolerance must be non-negative')

	d_gt_to_pred = distances_pointset_to_surface(gt_poly, pred_poly, max_points=max_points)
	d_pred_to_gt = distances_pointset_to_surface(pred_poly, gt_poly, max_points=max_points)

	n_gt = d_gt_to_pred.size
	n_pred = d_pred_to_gt.size
	if n_gt == 0 and n_pred == 0:
		return float('nan')

	c_gt = int(np.count_nonzero(d_gt_to_pred <= tolerance)) if n_gt else 0
	c_pred = int(np.count_nonzero(d_pred_to_gt <= tolerance)) if n_pred else 0

	den = n_gt + n_pred
	if den == 0:
		return float('nan')
	return float((c_gt + c_pred) / den)


def mesh_to_binary_numpy(poly: vtk.vtkPolyData, spacing: Tuple[float, float, float]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
	"""Rasterize `poly` into a binary numpy array using given `spacing`.

	Returns (mask_flat, dims) where mask_flat is a 1D boolean numpy array of length nx*ny*nz
	and dims is (nx, ny, nz).
	"""
	bounds = poly.GetBounds()
	if bounds is None:
		return np.array([], dtype=bool), (0, 0, 0)
	xmin, xmax, ymin, ymax, zmin, zmax = bounds
	sx, sy, sz = spacing
	if sx <= 0 or sy <= 0 or sz <= 0:
		raise ValueError('spacing values must be positive')

	nx = max(1, int(math.ceil((xmax - xmin) / sx)) + 1)
	ny = max(1, int(math.ceil((ymax - ymin) / sy)) + 1)
	nz = max(1, int(math.ceil((zmax - zmin) / sz)) + 1)

	image = vtk.vtkImageData()
	image.SetSpacing((sx, sy, sz))
	image.SetOrigin((xmin, ymin, zmin))
	image.SetDimensions(nx, ny, nz)
	image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

	scalars = image.GetPointData().GetScalars()
	arr = numpy_support.vtk_to_numpy(scalars)
	arr[:] = 1  # fill with foreground

	pol2st = vtk.vtkPolyDataToImageStencil()
	pol2st.SetInputData(poly)
	pol2st.SetOutputOrigin(image.GetOrigin())
	pol2st.SetOutputSpacing(image.GetSpacing())
	pol2st.SetOutputWholeExtent(image.GetExtent())
	pol2st.Update()

	imgstenc = vtk.vtkImageStencil()
	imgstenc.SetInputData(image)
	imgstenc.SetStencilConnection(pol2st.GetOutputPort())
	imgstenc.ReverseStencilOff()
	imgstenc.SetBackgroundValue(0)
	imgstenc.Update()

	out = imgstenc.GetOutput()
	out_scalars = out.GetPointData().GetScalars()
	out_arr = numpy_support.vtk_to_numpy(out_scalars).astype(bool)
	return out_arr, (nx, ny, nz)


def voxelize_pair(gt_poly: vtk.vtkPolyData, pred_poly: vtk.vtkPolyData, spacing: Tuple[float, float, float]):
	"""Rasterize both meshes into a common grid defined by `spacing`.

	Returns (mask_gt, mask_pred, dims) where masks are 1D boolean arrays of length nx*ny*nz.
	"""
	# compute combined bounds
	b1 = gt_poly.GetBounds()
	b2 = pred_poly.GetBounds()
	if b1 is None or b2 is None:
		return np.array([], dtype=bool), np.array([], dtype=bool), (0, 0, 0)
	xmin = min(b1[0], b2[0])
	xmax = max(b1[1], b2[1])
	ymin = min(b1[2], b2[2])
	ymax = max(b1[3], b2[3])
	zmin = min(b1[4], b2[4])
	zmax = max(b1[5], b2[5])

	sx, sy, sz = spacing
	nx = max(1, int(math.ceil((xmax - xmin) / sx)) + 1)
	ny = max(1, int(math.ceil((ymax - ymin) / sy)) + 1)
	nz = max(1, int(math.ceil((zmax - zmin) / sz)) + 1)

	# base image template
	base = vtk.vtkImageData()
	base.SetSpacing((sx, sy, sz))
	base.SetOrigin((xmin, ymin, zmin))
	base.SetDimensions(nx, ny, nz)
	base.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

	def _rasterize(poly: vtk.vtkPolyData):
		img = vtk.vtkImageData()
		img.DeepCopy(base)
		scalars = img.GetPointData().GetScalars()
		arr = numpy_support.vtk_to_numpy(scalars)
		arr[:] = 1

		pol2st = vtk.vtkPolyDataToImageStencil()
		pol2st.SetInputData(poly)
		pol2st.SetOutputOrigin(img.GetOrigin())
		pol2st.SetOutputSpacing(img.GetSpacing())
		pol2st.SetOutputWholeExtent(img.GetExtent())
		pol2st.Update()

		imgstenc = vtk.vtkImageStencil()
		imgstenc.SetInputData(img)
		imgstenc.SetStencilConnection(pol2st.GetOutputPort())
		imgstenc.ReverseStencilOff()
		imgstenc.SetBackgroundValue(0)
		imgstenc.Update()

		out = imgstenc.GetOutput()
		out_scalars = out.GetPointData().GetScalars()
		out_arr = numpy_support.vtk_to_numpy(out_scalars).astype(bool)
		return out_arr

	mask_gt = _rasterize(gt_poly)
	mask_pred = _rasterize(pred_poly)
	return mask_gt, mask_pred, (nx, ny, nz)


def compute_metrics(gt_poly: vtk.vtkPolyData, pred_poly: vtk.vtkPolyData, max_points: Optional[int] = None, voxel_spacing: Optional[Tuple[float, float, float]] = None, centerline: Optional[vtk.vtkPolyData] = None, metrics_to_compute: Optional[frozenset] = None) -> dict:
	"""Compute Hausdorff and ASSD between gt and pred polydata (one-way: GT to prediction).

	Computes distances from each point in GT to the nearest point on the prediction surface.
	This is a one-way metric: only GT -> prediction, not prediction -> GT.

	Parameters
	----------
	gt_poly : vtk.vtkPolyData
		Ground truth surface mesh
	pred_poly : vtk.vtkPolyData
		Prediction surface mesh
	max_points : int, optional
		Maximum number of points to sample for distance computation
	voxel_spacing : tuple of 3 floats, optional
		Voxel spacing for Dice computation. Dice uses full occupancy maps (numpy arrays)
		from voxelization—no point sampling. In same units as mesh coordinates.
	centerline : vtk.vtkPolyData, optional
		Centerline for computing centerline overlap metric
	metrics_to_compute : frozenset, optional
		If set, only compute these metrics (names from METRIC_NAMES). If None, compute all.
	
	Returns
	-------
	dict
		Dictionary with keys:
		  - 'hausdorff_sym' : Hausdorff distance (same as gt_to_pred since one-way)
		  - 'hausdorff_gt_to_pred' : Hausdorff distance (gt -> pred)
		  - 'hausdorff_pred_to_gt' : NaN (not computed)
		  - 'mean_gt_to_pred' : mean distance (gt points to pred surface)
		  - 'mean_pred_to_gt' : NaN (not computed)
		  - 'assd' : average surface distance (same as mean_gt_to_pred since one-way)
		  - 'centerline_overlap' : ratio of centerline length within prediction surface (0-1)
		  - 'mean_normal_angular_error_gt_to_pred' : mean angular error (degrees) between GT and pred normals (one-way)
		  - 'std_normal_angular_error_gt_to_pred' : std of angular errors (degrees)
		  - 'max_normal_angular_error_gt_to_pred' : max angular error (degrees)
		  - 		'volume_gt', 'volume_pred' : enclosed volume (e.g. mm³)
		  - 'volume_error_abs' : signed volume error (pred - gt)
		  - 'volume_error_rel' : relative volume error (pred - gt) / gt
		  - 'volume_error_radii_*' : volume error per radius bucket (when centerline provided)
		  - 'dice_radii_*' : Dice per radius bucket (when centerline provided)
		  - 'surface_area_gt', 'surface_area_pred' : surface area (e.g. mm²)
		  - 'surface_area_error_abs' : signed surface area error (pred - gt)
		  - 'surface_area_error_rel' : relative surface area error (pred - gt) / gt
		  - 'surface_dice_t1', 'surface_dice_t2' : surface Dice at two fixed tolerances (same units as mesh coordinates)
	
	Note
	----
	Distance metrics are in the same units as the input mesh coordinates. For medical
	imaging data, meshes are typically in physical space (millimeters), so distances
	are reported in millimeters.
	"""
	compute_all = metrics_to_compute is None
	do_distance = compute_all or ('distance' in metrics_to_compute)
	do_dice = compute_all or ('dice' in metrics_to_compute)
	do_centerline_overlap = compute_all or ('centerline_overlap' in metrics_to_compute)
	do_volume_radii = compute_all or ('volume_radii' in metrics_to_compute)
	do_dice_radii = compute_all or ('dice_radii' in metrics_to_compute)
	do_normal = compute_all or ('normal' in metrics_to_compute)
	do_volume = compute_all or ('volume' in metrics_to_compute)
	do_surface_area = compute_all or ('surface_area' in metrics_to_compute)
	do_surface_dice = compute_all or ('surface_dice' in metrics_to_compute)

	# Only compute distances from GT points to prediction surface (for distance metrics)
	if do_distance:
		d_gt_to_pred = distances_pointset_to_surface(gt_poly, pred_poly, max_points=max_points)
	else:
		d_gt_to_pred = np.array([], dtype=float)

	# protect against empty arrays
	def _safe_mean(a: np.ndarray) -> float:
		return float(np.mean(a)) if a.size else float('nan')

	mean_gt_to_pred = _safe_mean(d_gt_to_pred)

	# classic (max) Hausdorff distance (one-way: GT -> pred)
	haus_gt_to_pred = float(np.max(d_gt_to_pred)) if d_gt_to_pred.size else float('nan')
	# Since we only compute one-way, symmetric is the same as directed
	haus_sym = haus_gt_to_pred

	# 95th percentile (HD95) (one-way: GT -> pred)
	hd95_gt_to_pred = float(np.percentile(d_gt_to_pred, 95)) if d_gt_to_pred.size else float('nan')
	# Since we only compute one-way, symmetric is the same as directed
	hd95_sym = hd95_gt_to_pred

	# ASSD is just the mean distance since we only compute one-way
	assd = mean_gt_to_pred

	result = {
		'hausdorff_sym': haus_sym,
		'hausdorff_gt_to_pred': haus_gt_to_pred,
		'hausdorff_pred_to_gt': float('nan'),  # Not computed (one-way only)
		'hd95_sym': hd95_sym,
		'hd95_gt_to_pred': hd95_gt_to_pred,
		'hd95_pred_to_gt': float('nan'),  # Not computed (one-way only)
		'mean_gt_to_pred': mean_gt_to_pred,
		'mean_pred_to_gt': float('nan'),  # Not computed (one-way only)
		'assd': assd,
		'n_sampled_gt': int(d_gt_to_pred.size),
		'n_sampled_pred': 0,  # Not computed (one-way only)
		'dice': float('nan'),
		'n_voxels_gt': 0,
		'n_voxels_pred': 0,
		'centerline_overlap': float('nan'),
		'mean_normal_angular_error_gt_to_pred': float('nan'),
		'std_normal_angular_error_gt_to_pred': float('nan'),
		'max_normal_angular_error_gt_to_pred': float('nan'),
		'volume_gt': float('nan'),
		'volume_pred': float('nan'),
		'volume_error_abs': float('nan'),
		'volume_error_rel': float('nan'),
		**{fn: float('nan') for fn in _VOLUME_RADII_FIELDNAMES},
		**{fn: float('nan') for fn in _DICE_RADII_FIELDNAMES},
		'surface_area_gt': float('nan'),
		'surface_area_pred': float('nan'),
		'surface_area_error_abs': float('nan'),
		'surface_area_error_rel': float('nan'),
		'surface_dice_t1': float('nan'),
		'surface_dice_t2': float('nan'),
	}

	# Compute Dice coefficient using full occupancy maps (no sampling): voxelize both meshes
	# onto a common grid and compute 2*|A∩B|/(|A|+|B|) from numpy arrays
	if do_dice and voxel_spacing is not None:
		try:
			mask_gt, mask_pred, dims = voxelize_pair(gt_poly, pred_poly, voxel_spacing)
			if mask_gt.size and mask_pred.size and mask_gt.size == mask_pred.size:
				n_gt = int(np.count_nonzero(mask_gt))
				n_pred = int(np.count_nonzero(mask_pred))
				inter = int(np.count_nonzero(mask_gt & mask_pred))
				dice = float((2.0 * inter) / (n_gt + n_pred)) if (n_gt + n_pred) > 0 else float('nan')
				result['dice'] = dice
				result['n_voxels_gt'] = n_gt
				result['n_voxels_pred'] = n_pred
		except Exception:
			# if voxelization fails, leave dice as nan
			pass

	# Compute centerline overlap and volume-per-radii if centerline provided
	if centerline is not None and (do_centerline_overlap or do_volume_radii or do_dice_radii):
		if do_centerline_overlap:
			try:
				centerline_overlap = percent_centerline_length_mesh(pred_poly, centerline)
				result['centerline_overlap'] = centerline_overlap
			except Exception as e:
				# if centerline overlap computation fails, leave as nan
				pass
		# Volume error per radius bucket (requires MaximumInscribedSphereRadius)
		if do_volume_radii and volume_error_per_radii is not None:
			try:
				vpr = volume_error_per_radii(gt_poly, pred_poly, centerline)
				for k, v in vpr.items():
					if k != "volume_error_radii_buckets":
						result[k] = v
			except Exception as e:
				import warnings
				warnings.warn(
					f"volume_error_per_radii failed: {e}; metrics will be NaN.",
					UserWarning,
					stacklevel=2,
				)
		# Dice per radius bucket (requires MaximumInscribedSphereRadius)
		if do_dice_radii and dice_per_radii is not None:
			try:
				dpr = dice_per_radii(gt_poly, pred_poly, centerline, spacing=voxel_spacing)
				for k, v in dpr.items():
					if k != "dice_radii_buckets":
						result[k] = v
			except Exception as e:
				import warnings
				warnings.warn(
					f"dice_per_radii failed: {e}; metrics will be NaN.",
					UserWarning,
					stacklevel=2,
				)

	# Normal angular error (GT -> pred): mean, std, max (degrees); same point sampling as distance metrics
	if do_normal:
		try:
			nae = normal_angular_error_gt_to_pred(gt_poly, pred_poly, max_points=max_points)
			result['mean_normal_angular_error_gt_to_pred'] = nae['mean']
			result['std_normal_angular_error_gt_to_pred'] = nae['std']
			result['max_normal_angular_error_gt_to_pred'] = nae['max']
		except Exception:
			pass  # leave as nan if computation fails

	# Volume and volume error (requires closed triangulated meshes)
	if do_volume:
		try:
			ve = volume_error_metrics(gt_poly, pred_poly)
			result['volume_gt'] = ve['volume_gt']
			result['volume_pred'] = ve['volume_pred']
			result['volume_error_abs'] = ve['volume_error_abs']
			result['volume_error_rel'] = ve['volume_error_rel']
		except Exception:
			pass

	# Surface area and surface area error (requires closed triangulated meshes)
	if do_surface_area:
		try:
			sa = surface_area_error_metrics(gt_poly, pred_poly)
			result['surface_area_gt'] = sa['surface_area_gt']
			result['surface_area_pred'] = sa['surface_area_pred']
			result['surface_area_error_abs'] = sa['surface_area_error_abs']
			result['surface_area_error_rel'] = sa['surface_area_error_rel']
		except Exception:
			pass

	# Surface Dice at two fixed tolerances (e.g. 1.0 and 2.0 in mesh units)
	if do_surface_dice:
		try:
			t1 = 0.02
			t2 = 0.05
			result['surface_dice_t1'] = surface_dice(gt_poly, pred_poly, tolerance=t1, max_points=max_points)
			result['surface_dice_t2'] = surface_dice(gt_poly, pred_poly, tolerance=t2, max_points=max_points)
		except Exception:
			pass

	return result


def find_matching_vtps(dir1: str, dir2: str, ext: str = '.vtp') -> List[Tuple[str, str, str]]:
	"""Return list of tuples (case_name, path_gt, path_pred) for files present in both dirs."""
	files1 = {f for f in os.listdir(dir1) if f.lower().endswith(ext)}
	files2 = {f for f in os.listdir(dir2) if f.lower().endswith(ext)}
	common = sorted(files1.intersection(files2))
	pairs = []
	for f in common:
		case = os.path.splitext(f)[0]
		pairs.append((case, os.path.join(dir1, f), os.path.join(dir2, f)))
	return pairs


def find_cases_in_all_dirs(gt_dir: str, pred_dirs: List[str], ext: str = '.vtp') -> set:
	"""Return set of case names present in gt_dir and every pred_dir."""
	gt_files = {os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.lower().endswith(ext)}
	common = gt_files
	for pred_dir in pred_dirs:
		pred_files = {os.path.splitext(f)[0] for f in os.listdir(pred_dir) if f.lower().endswith(ext)}
		common = common.intersection(pred_files)
	return common


def _print_missing_cases(
	gt_dir: str,
	pred_dirs: List[str],
	pred_folder_names: List[str],
	common_cases: set,
	ext: str,
) -> None:
	"""Print which cases are missing from each folder (gt and each pred)."""
	gt_cases = {os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.lower().endswith(ext)}
	excluded = gt_cases - common_cases
	if excluded:
		excl_sorted = sorted(excluded)
		print(f'  gt_dir: {len(excl_sorted)} cases excluded (missing from some pred): {excl_sorted}', file=sys.stderr)
	for pred_dir, name in zip(pred_dirs, pred_folder_names):
		pred_cases = {os.path.splitext(f)[0] for f in os.listdir(pred_dir) if f.lower().endswith(ext)}
		missing_from_pred = gt_cases - pred_cases
		extra_in_pred = pred_cases - gt_cases
		if missing_from_pred:
			ms = sorted(missing_from_pred)
			print(f'  {name}: missing {len(ms)} cases (in gt, not here): {ms}', file=sys.stderr)
		if extra_in_pred:
			ex = sorted(extra_in_pred)
			print(f'  {name}: extra {len(ex)} cases (here, not in gt): {ex}', file=sys.stderr)


# Volume-per-radii bucket field names (radius ranges in mm)
# Derive from volume_per_radii so bucket changes propagate (avoid hardcoding)
_VOLUME_RADII_FIELDNAMES = (
	get_volume_per_radii_fieldnames(DEFAULT_RADIUS_BUCKETS)
	if (get_volume_per_radii_fieldnames is not None and DEFAULT_RADIUS_BUCKETS is not None)
	else []
)

# Dice-per-radii bucket field names (dice_radii_* only; n_segments shared with volume)
_DICE_RADII_FIELDNAMES = (
	[f for f in get_dice_per_radii_fieldnames(DEFAULT_RADIUS_BUCKETS) if f.startswith('dice_radii_')]
	if (get_dice_per_radii_fieldnames is not None and DEFAULT_RADIUS_BUCKETS is not None)
	else []
)

# Valid metric names for --metrics argument (comma-separated)
# 'all' = compute everything (default)
METRIC_NAMES = frozenset({
	'distance',       # Hausdorff, HD95, ASSD (point-to-surface)
	'dice',           # Dice coefficient (requires --spacing)
	'centerline_overlap',  # Centerline overlap ratio (requires --centerline-dir)
	'volume_radii',   # Volume error per radius bucket (requires --centerline-dir)
	'dice_radii',     # Dice per radius bucket (requires --centerline-dir)
	'normal',         # Normal angular error (mean, std, max)
	'volume',         # Volume and volume error
	'surface_area',   # Surface area and surface area error
	'surface_dice',   # Surface Dice at two tolerances
})


# Metric column names (excluding 'case') for CSV and summary
RESULTS_FIELDNAMES = [
	'case',
	'hausdorff_sym',
	'hd95_sym',
	'hausdorff_gt_to_pred',
	'hd95_gt_to_pred',
	'hausdorff_pred_to_gt',
	'hd95_pred_to_gt',
	'mean_gt_to_pred',
	'mean_pred_to_gt',
	'assd',
	'n_sampled_gt',
	'n_sampled_pred',
	'dice',
	'n_voxels_gt',
	'n_voxels_pred',
	'centerline_overlap',
	'mean_normal_angular_error_gt_to_pred',
	'std_normal_angular_error_gt_to_pred',
	'max_normal_angular_error_gt_to_pred',
	'volume_gt',
	'volume_pred',
	'volume_error_abs',
	'volume_error_rel',
] + _VOLUME_RADII_FIELDNAMES + _DICE_RADII_FIELDNAMES + [
	'surface_area_gt',
	'surface_area_pred',
	'surface_area_error_abs',
	'surface_area_error_rel',
	'surface_dice_t1',
	'surface_dice_t2',
]


def write_results_csv(out_csv: str, rows: List[dict]) -> None:
	fieldnames = RESULTS_FIELDNAMES
	with open(out_csv, 'w', newline='') as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in rows:
			w.writerow({k: r.get(k, '') for k in fieldnames})

		# Summary rows: mean and std over cases for numeric fields
		if rows:
			mean_row = {'case': 'MEAN'}
			std_row = {'case': 'STD'}
			for k in fieldnames[1:]:  # skip 'case'
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


def _rows_mean_std(rows: List[dict], metric_keys: List[str]) -> Tuple[dict, dict]:
	"""Compute mean and std for each metric over rows. Returns (mean_row, std_row) as dicts."""
	mean_row = {}
	std_row = {}
	for k in metric_keys:
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
			mean_row[k] = float('nan')
			std_row[k] = float('nan')
	return mean_row, std_row


def write_volume_radii_raw_csv(out_path: str, rows: List[dict]) -> None:
	"""Write raw (case, error_rel, radius) per segment to CSV for volume-per-radii analysis."""
	fieldnames = ['case', 'error_rel', 'radius']
	with open(out_path, 'w', newline='') as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in rows:
			raw = r.get('volume_error_radii_raw') or []
			case = r.get('case', '')
			for error_rel, radius in raw:
				w.writerow({'case': case, 'error_rel': error_rel, 'radius': radius})


def write_dice_radii_raw_csv(out_path: str, rows: List[dict]) -> None:
	"""Write raw (case, dice, radius) per segment to CSV for dice-per-radii analysis."""
	fieldnames = ['case', 'dice', 'radius']
	with open(out_path, 'w', newline='') as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in rows:
			raw = r.get('dice_radii_raw') or []
			case = r.get('case', '')
			for dice_val, radius in raw:
				w.writerow({'case': case, 'dice': dice_val, 'radius': radius})


def write_summary_csv(summary_csv: str, summary_rows: List[dict]) -> None:
	"""Write summary CSV: one row per prediction folder with mean and std for each metric."""
	metric_keys = [k for k in RESULTS_FIELDNAMES if k != 'case']
	fieldnames = ['pred_folder'] + [f'{k}_mean' for k in metric_keys] + [f'{k}_std' for k in metric_keys]
	with open(summary_csv, 'w', newline='') as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in summary_rows:
			w.writerow({k: r.get(k, '') for k in fieldnames})


def parse_metrics_arg(s: str) -> Optional[frozenset]:
	"""Parse --metrics argument. Returns None for 'all', else frozenset of metric names."""
	if not s or s.strip().lower() == 'all':
		return None
	names = [m.strip().lower() for m in s.split(',') if m.strip()]
	invalid = [n for n in names if n not in METRIC_NAMES]
	if invalid:
		raise argparse.ArgumentTypeError(
			f"Invalid metric(s): {invalid}. Valid: {', '.join(sorted(METRIC_NAMES))}, all"
		)
	return frozenset(names)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description='Compute Hausdorff and ASSD between matching .vtp meshes in two dirs')
	p.add_argument('gt_dir', help='Ground-truth directory containing .vtp files')
	p.add_argument('pred_dir', help='Prediction directory containing .vtp files (ignored if --predictions-root is set)')
	p.add_argument('--predictions-root', default=None, help='Folder containing multiple prediction subfolders. If set, metrics are computed for each subfolder and summary.csv is written with mean and std per prediction folder.')
	p.add_argument('--summary-csv', default=None, help='Output path for summary CSV when using --predictions-root (default: <predictions-root>/summary.csv)')
	p.add_argument('--out-csv', default=None, help='Output CSV path (single pred dir). With --predictions-root, per-folder CSVs are written as <stem>_<pred_folder>.csv. Default: <pred-dir>/mesh_metrics.csv or <predictions-root>/mesh_metrics.csv')
	p.add_argument('--metrics', type=parse_metrics_arg, default=None, metavar='M1,M2,...', help=f'Comma-separated metrics to compute. Default: all. Valid: {", ".join(sorted(METRIC_NAMES))}')
	p.add_argument('--max-points', type=int, default=None, help='If set, randomly sample at most this many points from each mesh')
	p.add_argument('--ext', default='.vtp', help='File extension to look for (default: .vtp)')
	p.add_argument('--spacing', default='1.0', help='Voxel spacing for Dice voxelization (full occupancy, no sampling). Single float or comma-separated three floats (e.g. "1.0" or "0.5,0.5,1.0"). Default: 1.0 (isotropic).')
	p.add_argument('--centerline-dir', default=None, help='Directory containing centerline .vtp files. Required if --clip is used. If provided, centerline overlap metric will be computed.')
	p.add_argument('--clip', action='store_true', help='Clip meshes using centerlines before computing metrics. Requires --centerline-dir.')
	p.add_argument('--clip-temp-dir', default=None, help='Temporary directory for clipping box files (default: same as script directory)')
	p.add_argument('--clip-output-dir', default=None, help='Directory to write clipped meshes for debugging. Default: <pred-dir> when --clip is used.')
	p.add_argument('--quiet', action='store_true', help='Reduce logging')
	return p.parse_args(argv)


def run_metrics_for_pred_dir(
	gt_dir: str,
	pred_dir: str,
	ext: str,
	max_points: Optional[int],
	voxel_spacing: Optional[Tuple[float, float, float]],
	clip: bool,
	centerline_dir: Optional[str],
	clip_temp_dir: Optional[str],
	clip_output_dir: Optional[str],
	quiet: bool,
	cases_filter: Optional[set] = None,
	metrics_to_compute: Optional[frozenset] = None,
) -> List[dict]:
	"""Run metrics for all matching cases between gt_dir and pred_dir. Returns list of per-case rows.

	If cases_filter is provided, only process cases in that set (must exist in both gt_dir and pred_dir).
	"""
	pairs = find_matching_vtps(gt_dir, pred_dir, ext=ext)
	if cases_filter is not None:
		pairs = [(c, gt, pred) for c, gt, pred in pairs if c in cases_filter]
	if not pairs:
		return []
	rows = []
	for case, gt_path, pred_path in tqdm(pairs, desc='Cases (Dice etc.)', disable=quiet):
		try:
			gt_poly = load_vtp(gt_path)
			pred_poly = load_vtp(pred_path)
			if clip and centerline_dir is not None:
				centerline_path = os.path.join(centerline_dir, case + ext)
				if os.path.exists(centerline_path):
					try:
						centerline = vf.read_geo(centerline_path).GetOutput() if vf is not None else load_vtp(centerline_path)
						gt_poly = clip_surface_mesh(gt_poly, centerline, case_name=case, temp_dir=clip_temp_dir)
						pred_poly = clip_surface_mesh(pred_poly, centerline, case_name=case, temp_dir=clip_temp_dir)
					except Exception:
						pass
			centerline = None
			if centerline_dir and os.path.exists(os.path.join(centerline_dir, case + ext)):
				try:
					centerline = (vf.read_geo(os.path.join(centerline_dir, case + ext)).GetOutput() if vf is not None else load_vtp(os.path.join(centerline_dir, case + ext)))
				except Exception:
					pass
			metrics = compute_metrics(gt_poly, pred_poly, max_points=max_points, voxel_spacing=voxel_spacing, centerline=centerline, metrics_to_compute=metrics_to_compute)
			row = {'case': case}
			row.update(metrics)
			rows.append(row)
		except Exception as e:
			if not quiet:
				print(f'  Error processing {case}: {e}', file=sys.stderr)
	return rows


def parse_spacing_arg(s: Optional[str]) -> Optional[Tuple[float, float, float]]:
	if s is None:
		return None
	parts = [p.strip() for p in str(s).split(',') if p.strip()]
	try:
		if len(parts) == 1:
			v = float(parts[0])
			return (v, v, v)
		elif len(parts) == 3:
			return (float(parts[0]), float(parts[1]), float(parts[2]))
	except ValueError:
		pass
	raise argparse.ArgumentTypeError('Invalid spacing format. Use single float or three comma-separated floats.')


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	gt_dir = args.gt_dir
	pred_dir = args.pred_dir
	predictions_root = getattr(args, 'predictions_root', None)
	summary_csv = getattr(args, 'summary_csv', None)
	out_csv = args.out_csv
	max_points = args.max_points
	ext = args.ext if args.ext.startswith('.') else '.' + args.ext
	clip = args.clip
	centerline_dir = args.centerline_dir
	clip_temp_dir = args.clip_temp_dir
	clip_output_dir = args.clip_output_dir
	metrics_to_compute = getattr(args, 'metrics', None)

	# Default output paths to prediction root or pred dir
	output_base = predictions_root if predictions_root is not None else pred_dir
	if out_csv is None:
		out_csv = os.path.join(output_base, 'mesh_metrics.csv')
	if summary_csv is None and predictions_root is not None:
		summary_csv = os.path.join(predictions_root, 'summary.csv')
	if clip_output_dir is None and clip:
		clip_output_dir = output_base

	# Validate clipping arguments
	if clip:
		if centerline_dir is None:
			print('Error: --centerline-dir is required when --clip is used', file=sys.stderr)
			return 1
		if not os.path.isdir(centerline_dir):
			print(f'Error: Centerline directory does not exist: {centerline_dir}', file=sys.stderr)
			return 1
		if vf is None or bryan_get_clipping_parameters is None:
			print('Error: Clipping functionality is not available. Required modules could not be imported.', file=sys.stderr)
			return 1

	spacing = parse_spacing_arg(args.spacing) if getattr(args, 'spacing', None) is not None else None

	# Multi-pred mode: folder containing multiple prediction subfolders -> summary.csv
	if predictions_root is not None:
		if not os.path.isdir(predictions_root):
			print(f'Error: Predictions root is not a directory: {predictions_root}', file=sys.stderr)
			return 1
		subdirs = sorted([d for d in os.listdir(predictions_root) if os.path.isdir(os.path.join(predictions_root, d))])
		if not subdirs:
			print(f'Error: No subdirectories found in {predictions_root}', file=sys.stderr)
			return 1
		pred_dirs = [os.path.join(predictions_root, d) for d in subdirs]
		common_cases = find_cases_in_all_dirs(gt_dir, pred_dirs, ext=ext)
		if not common_cases:
			print(f'Error: No cases found in both gt_dir and all {len(subdirs)} prediction folders', file=sys.stderr)
			_print_missing_cases(gt_dir, pred_dirs, subdirs, set(), ext)
			return 2
		if not args.quiet:
			print(f'Processing {len(common_cases)} cases present in gt_dir and all {len(subdirs)} prediction folders')
			_print_missing_cases(gt_dir, pred_dirs, subdirs, common_cases, ext)
		metric_keys = [k for k in RESULTS_FIELDNAMES if k != 'case']
		summary_rows = []
		for pred_folder in subdirs:
			pred_path = os.path.join(predictions_root, pred_folder)
			if not args.quiet:
				print(f'Prediction folder: {pred_folder}')
			rows = run_metrics_for_pred_dir(
				gt_dir, pred_path, ext, max_points, spacing,
				clip, centerline_dir, clip_temp_dir, clip_output_dir, args.quiet,
				cases_filter=common_cases,
				metrics_to_compute=metrics_to_compute,
			)
			if not rows:
				if not args.quiet:
					print(f'  No matching cases for {pred_folder}', file=sys.stderr)
				continue
			# Per-prediction-folder CSV: scores per case (same format as single-pred --out-csv)
			out_dir = os.path.dirname(out_csv)
			base = os.path.basename(out_csv)
			stem, csv_ext = os.path.splitext(base)
			if not csv_ext:
				csv_ext = '.csv'
			per_folder_csv = os.path.join(out_dir, f'{stem}_{pred_folder}{csv_ext}') if out_dir else f'{stem}_{pred_folder}{csv_ext}'
			write_results_csv(per_folder_csv, rows)
			if not args.quiet:
				print(f'  Wrote {per_folder_csv}')
			# Raw volume-per-radii: (case, error_rel, radius) per segment for this prediction folder
			if metrics_to_compute is None or 'volume_radii' in metrics_to_compute:
				raw_path = os.path.join(out_dir, f'volume_radii_raw_{pred_folder}.csv') if out_dir else f'volume_radii_raw_{pred_folder}.csv'
				write_volume_radii_raw_csv(raw_path, rows)
				if not args.quiet:
					n_raw = sum(len(r.get('volume_error_radii_raw') or []) for r in rows)
					if n_raw > 0:
						print(f'  Wrote {raw_path} ({n_raw} segments)')
			# Raw dice-per-radii: (case, dice, radius) per segment for this prediction folder
			if metrics_to_compute is None or 'dice_radii' in metrics_to_compute:
				raw_path = os.path.join(out_dir, f'dice_radii_raw_{pred_folder}.csv') if out_dir else f'dice_radii_raw_{pred_folder}.csv'
				write_dice_radii_raw_csv(raw_path, rows)
				if not args.quiet:
					n_raw = sum(len(r.get('dice_radii_raw') or []) for r in rows)
					if n_raw > 0:
						print(f'  Wrote {raw_path} ({n_raw} segments)')
			mean_row, std_row = _rows_mean_std(rows, metric_keys)
			summary_rows.append({
				'pred_folder': pred_folder,
				**{f'{k}_mean': mean_row[k] for k in metric_keys},
				**{f'{k}_std': std_row[k] for k in metric_keys},
			})
		if summary_rows:
			write_summary_csv(summary_csv, summary_rows)
			if not args.quiet:
				print(f'Wrote summary to {summary_csv} ({len(summary_rows)} prediction folders)')
		else:
			print('No metrics computed for any prediction folder.', file=sys.stderr)
			return 2
		return 0

	# Single-pred mode: one gt_dir and one pred_dir
	if clip_output_dir is not None:
		os.makedirs(clip_output_dir, exist_ok=True)
		if not args.quiet:
			print(f'Clipped meshes will be saved to: {clip_output_dir}')

	pairs = find_matching_vtps(gt_dir, pred_dir, ext=ext)
	if not pairs:
		print(f'No matching {ext} files found in {gt_dir} and {pred_dir}', file=sys.stderr)
		return 2

	rows = []
	for case, gt_path, pred_path in tqdm(pairs, desc='Cases (Dice etc.)', disable=args.quiet):
		try:
			gt_poly = load_vtp(gt_path)
			pred_poly = load_vtp(pred_path)
			
			# Apply clipping if requested
			if clip:
				centerline_path = os.path.join(centerline_dir, case + '.vtp')
				if not os.path.exists(centerline_path):
					print(f'Warning: Centerline not found for {case}: {centerline_path}. Skipping clipping.', file=sys.stderr)
				else:
					if not args.quiet:
						print(f'  Clipping meshes using centerline: {centerline_path}')
					try:
						centerline = vf.read_geo(centerline_path).GetOutput()
						gt_poly_clipped = clip_surface_mesh(gt_poly, centerline, case_name=case, temp_dir=clip_temp_dir)
						pred_poly_clipped = clip_surface_mesh(pred_poly, centerline, case_name=case, temp_dir=clip_temp_dir)
						
						# Write clipped meshes for debugging if output directory specified
						if clip_output_dir is not None:
							gt_clipped_path = os.path.join(clip_output_dir, f'{case}_gt_clipped.vtp')
							pred_clipped_path = os.path.join(clip_output_dir, f'{case}_pred_clipped.vtp')
							write_vtp(gt_clipped_path, gt_poly_clipped)
							write_vtp(pred_clipped_path, pred_poly_clipped)
							if not args.quiet:
								print(f'  Saved clipped meshes: {gt_clipped_path}, {pred_clipped_path}')
						
						# Use clipped meshes for metrics computation
						gt_poly = gt_poly_clipped
						pred_poly = pred_poly_clipped
					except Exception as e:
						print(f'Warning: Clipping failed for {case}: {e}. Using original meshes.', file=sys.stderr)
			
			# Load centerline if centerline directory provided
			centerline = None
			if centerline_dir is not None:
				centerline_path = os.path.join(centerline_dir, case + '.vtp')
				if os.path.exists(centerline_path):
					try:
						if vf is not None:
							centerline = vf.read_geo(centerline_path).GetOutput()
						else:
							centerline = load_vtp(centerline_path)
						if not args.quiet:
							print(f'  Loaded centerline: {centerline_path}')
					except Exception as e:
						if not args.quiet:
							print(f'  Warning: Failed to load centerline {centerline_path}: {e}', file=sys.stderr)
				else:
					if not args.quiet:
						print(f'  Warning: Centerline not found: {centerline_path}', file=sys.stderr)
			
			spacing = parse_spacing_arg(args.spacing) if getattr(args, 'spacing', None) is not None else None
			metrics = compute_metrics(gt_poly, pred_poly, max_points=max_points, voxel_spacing=spacing, centerline=centerline, metrics_to_compute=metrics_to_compute)
			row = {'case': case}
			row.update(metrics)
			rows.append(row)
			if not args.quiet:
				msg = f"  Hausdorff (GT->pred): {metrics['hausdorff_gt_to_pred']:.6g}, HD95 (GT->pred): {metrics['hd95_gt_to_pred']:.6g}, ASSD: {metrics['assd']:.6g}"
				if not math.isnan(metrics.get('dice', math.nan)):
					msg += f", Dice: {metrics['dice']:.6g} (voxels gt/pred: {metrics.get('n_voxels_gt',0)}/{metrics.get('n_voxels_pred',0)})"
				if not math.isnan(metrics.get('centerline_overlap', math.nan)):
					msg += f", Centerline Overlap: {metrics['centerline_overlap']:.6g}"
				if not math.isnan(metrics.get('mean_normal_angular_error_gt_to_pred', math.nan)):
					msg += f", Normal Angular Error (deg) mean: {metrics['mean_normal_angular_error_gt_to_pred']:.6g}, std: {metrics['std_normal_angular_error_gt_to_pred']:.6g}, max: {metrics['max_normal_angular_error_gt_to_pred']:.6g}"
				if np.isfinite(metrics.get('volume_gt', float('nan'))):
					msg += f", Volume error abs: {metrics['volume_error_abs']:.6g}, rel: {metrics['volume_error_rel']:.6g}"
				print(msg)
		except Exception as e:
			print(f'Error processing {case}: {e}', file=sys.stderr)

	# write CSV
	write_results_csv(out_csv, rows)

	# Raw volume-per-radii: (case, error_rel, radius) per segment
	if metrics_to_compute is None or 'volume_radii' in metrics_to_compute:
		out_dir = os.path.dirname(out_csv)
		raw_path = os.path.join(out_dir, 'volume_radii_raw.csv') if out_dir else 'volume_radii_raw.csv'
		write_volume_radii_raw_csv(raw_path, rows)
		n_raw = sum(len(r.get('volume_error_radii_raw') or []) for r in rows)
		if n_raw > 0 and not args.quiet:
			print(f'Wrote raw volume-per-radii to {raw_path} ({n_raw} segments)')

	# Raw dice-per-radii: (case, dice, radius) per segment
	if metrics_to_compute is None or 'dice_radii' in metrics_to_compute:
		out_dir = os.path.dirname(out_csv)
		raw_path = os.path.join(out_dir, 'dice_radii_raw.csv') if out_dir else 'dice_radii_raw.csv'
		write_dice_radii_raw_csv(raw_path, rows)
		n_raw = sum(len(r.get('dice_radii_raw') or []) for r in rows)
		if n_raw > 0 and not args.quiet:
			print(f'Wrote raw dice-per-radii to {raw_path} ({n_raw} segments)')

	# print summary
	haus_list = [r['hausdorff_sym'] for r in rows if not math.isnan(r.get('hausdorff_sym', math.nan))]
	assd_list = [r['assd'] for r in rows if not math.isnan(r.get('assd', math.nan))]
	hd95_list = [r['hd95_sym'] for r in rows if not math.isnan(r.get('hd95_sym', math.nan))]
	dice_list = [r['dice'] for r in rows if not math.isnan(r.get('dice', math.nan))]
	centerline_overlap_list = [r['centerline_overlap'] for r in rows if not math.isnan(r.get('centerline_overlap', math.nan))]
	if haus_list:
		print(f'Average Hausdorff (GT->pred) over {len(haus_list)} cases: {float(np.mean(haus_list)):.6g}')
	if assd_list:
		print(f'Average ASSD (GT->pred) over {len(assd_list)} cases: {float(np.mean(assd_list)):.6g}')
	if hd95_list:
		print(f'Average HD95 (GT->pred) over {len(hd95_list)} cases: {float(np.mean(hd95_list)):.6g}')
	if dice_list:
		print(f'Average Dice over {len(dice_list)} cases: {float(np.mean(dice_list)):.6g}')
	if centerline_overlap_list:
		print(f'Average Centerline Overlap over {len(centerline_overlap_list)} cases: {float(np.mean(centerline_overlap_list)):.6g}')
	if clip_output_dir is not None:
		print(f'Clipped meshes saved to: {clip_output_dir}')

	print(f'Wrote per-case results to: {out_csv}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())

