"""Dice coefficient along centerlines, bucketed by MaximumInscribedSphereRadius.

Computes Dice between prediction and ground truth meshes within local regions
defined by bounding boxes around each centerline segment. Uses bound_polydata_by_image
to clip meshes to the ROI, then voxelizes and computes Dice per segment.
Segments are grouped by radius ranges for analysis of Dice by branch size.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Tuple, Optional

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable, desc=None, **kwargs):
		return iterable

try:
	import vtk
except Exception as e:
	raise ImportError(
		"The 'vtk' Python package is required. "
		"Install it with `pip install vtk` or `conda install -c conda-forge vtk`."
	) from e

import numpy as np
from vtk.util import numpy_support

try:
	from ..modules.vtk_functions import bound_polydata_by_image
except ImportError:
	try:
		from modules.vtk_functions import bound_polydata_by_image
	except ImportError:
		bound_polydata_by_image = None

try:
	from .volume_per_radii import DEFAULT_RADIUS_BUCKETS
except ImportError:
	from volume_per_radii import DEFAULT_RADIUS_BUCKETS


def _create_roi_image(
	center: Tuple[float, float, float],
	radius: float,
	spacing: Tuple[float, float, float],
) -> vtk.vtkImageData:
	"""Create a vtkImageData whose bounds cover a box around the sphere.

	The image defines a region from (center - radius) to (center + radius)
	in each dimension. Dimensions are computed from extent and spacing.

	Parameters
	----------
	center : (x, y, z)
		Center of the ROI
	radius : float
		Half-extent of the ROI in each dimension
	spacing : (sx, sy, sz)
		Voxel spacing in mesh units (e.g. mm)

	Returns
	-------
	vtk.vtkImageData
		Image with bounds covering the box
	"""
	roi = vtk.vtkImageData()
	origin = (
		center[0] - radius,
		center[1] - radius,
		center[2] - radius,
	)
	extent = 2.0 * radius
	sx, sy, sz = spacing
	nx = max(2, int(math.ceil(extent / sx)) + 1)
	ny = max(2, int(math.ceil(extent / sy)) + 1)
	nz = max(2, int(math.ceil(extent / sz)) + 1)
	dims = (nx, ny, nz)

	roi.SetOrigin(origin)
	roi.SetSpacing(spacing)
	roi.SetDimensions(dims)
	roi.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
	return roi


def _voxelize_to_image(poly: vtk.vtkPolyData, ref_im: vtk.vtkImageData) -> np.ndarray:
	"""Rasterize polydata to a binary mask on the reference image grid.

	Returns a 1D boolean array (Fortran order, matching VTK).
	"""
	if poly is None or poly.GetNumberOfCells() == 0:
		# Empty mesh -> all zeros
		n = ref_im.GetNumberOfPoints()
		return np.zeros(n, dtype=bool)

	pol2st = vtk.vtkPolyDataToImageStencil()
	pol2st.SetInputData(poly)
	pol2st.SetOutputOrigin(ref_im.GetOrigin())
	pol2st.SetOutputSpacing(ref_im.GetSpacing())
	pol2st.SetOutputWholeExtent(ref_im.GetExtent())
	pol2st.SetTolerance(0.05)
	pol2st.Update()

	# Create image filled with 1, then stencil to get interior
	img = vtk.vtkImageData()
	img.DeepCopy(ref_im)
	scalars = img.GetPointData().GetScalars()
	arr = numpy_support.vtk_to_numpy(scalars)
	arr[:] = 1

	imgstenc = vtk.vtkImageStencil()
	imgstenc.SetInputData(img)
	imgstenc.SetStencilConnection(pol2st.GetOutputPort())
	imgstenc.ReverseStencilOff()
	imgstenc.SetBackgroundValue(0)
	imgstenc.Update()

	out = imgstenc.GetOutput()
	out_scalars = out.GetPointData().GetScalars()
	return numpy_support.vtk_to_numpy(out_scalars).astype(bool)


def _dice_from_masks(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
	"""Compute Dice = 2*|A∩B| / (|A| + |B|) from boolean masks."""
	n_gt = int(np.count_nonzero(mask_gt))
	n_pred = int(np.count_nonzero(mask_pred))

	if n_gt == 0 or n_pred == 0:
		return float('nan')
	if n_gt + n_pred == 0:
		return float('nan')
	inter = int(np.count_nonzero(mask_gt & mask_pred))
	return float((2.0 * inter) / (n_gt + n_pred))


def _dice_in_roi(
	gt_poly: vtk.vtkPolyData,
	pred_poly: vtk.vtkPolyData,
	roi_image: vtk.vtkImageData,
	bound_threshold: float = 0.0,
) -> float:
	"""Clip both meshes to ROI and compute Dice within it."""
	if bound_polydata_by_image is None:
		return float('nan')

	clipped_gt = bound_polydata_by_image(roi_image, gt_poly, bound_threshold)
	clipped_pred = bound_polydata_by_image(roi_image, pred_poly, bound_threshold)

	mask_gt = _voxelize_to_image(clipped_gt, roi_image)
	mask_pred = _voxelize_to_image(clipped_pred, roi_image)

	return _dice_from_masks(mask_gt, mask_pred)


def dice_per_radii(
	gt_poly: vtk.vtkPolyData,
	pred_poly: vtk.vtkPolyData,
	centerline: vtk.vtkPolyData,
	radius_buckets: Optional[List[Tuple[float, float]]] = None,
	segment_stride: int = 20,
	radius_scale: float = 1.1,
	radius_offset: float = 0.1,
	spacing: Optional[Tuple[float, float, float]] = None,
	bound_threshold: float = 0.0,
) -> dict:
	"""Compute Dice along centerline segments, bucketed by MaximumInscribedSphereRadius.

	For each centerline segment midpoint, creates an ROI image (box) around the point,
	clips GT and Pred meshes using bound_polydata_by_image, voxelizes both to the
	same grid, and computes Dice. Segments are grouped by mean radius into buckets.

	Parameters
	----------
	gt_poly : vtk.vtkPolyData
		Ground truth surface mesh
	pred_poly : vtk.vtkPolyData
		Prediction surface mesh
	centerline : vtk.vtkPolyData
		Centerline with 'MaximumInscribedSphereRadius' point data
	radius_buckets : list of (min_r, max_r), optional
		Radius ranges in mesh units (e.g. mm)
	segment_stride : int
		Process every Nth segment when there are 300+ segments. If fewer than 300 segments
		exist, all are used. Stride is capped so at least 300 segments are computed.
	radius_scale : float
		Scale factor for ROI size. Formula: roi_radius = radius_scale * r + radius_offset
	radius_offset : float
		Additive offset (in mesh units)
	spacing : tuple of 3 floats, optional
		Voxel spacing for Dice voxelization within each ROI (e.g. mm). Default: (1.0, 1.0, 1.0).
	bound_threshold : float
		Threshold for bound_polydata_by_image (expand box by this amount)

	Returns
	-------
	dict
		Keys for each bucket:
		  - 'dice_radii_<min>_<max>' : mean Dice in bucket
		  - 'n_segments_radii_<min>_<max>' : number of segments in bucket
		Plus 'dice_radii_buckets' : list of bucket labels
	"""
	if radius_buckets is None:
		radius_buckets = DEFAULT_RADIUS_BUCKETS
	if spacing is None:
		spacing = (1.0, 1.0, 1.0)
	elif isinstance(spacing, (int, float)):
		s = float(spacing)
		spacing = (s, s, s)
	else:
		spacing = tuple(float(x) for x in spacing)

	radii_arr = (
		centerline.GetPointData().GetArray('MaximumInscribedSphereRadius')
		or centerline.GetPointData().GetArray('Radius')
	)
	if radii_arr is None:
		warnings.warn(
			"Centerline missing 'MaximumInscribedSphereRadius' and 'Radius' point data; "
			"dice-per-radii metrics will be NaN.",
			UserWarning,
			stacklevel=2,
		)
		return _empty_dice_per_radii_result(radius_buckets)

	radii = np.asarray(numpy_support.vtk_to_numpy(radii_arr)).flatten()

	bucket_data = {i: [] for i in range(len(radius_buckets))}  # dice values per bucket
	raw_data: List[Tuple[float, float]] = []  # (dice, radius) for each segment

	num_cells = centerline.GetNumberOfCells()
	if num_cells == 0:
		return _empty_dice_per_radii_result(radius_buckets)

	# Build flat list of all segments (cell_idx, segment_idx)
	all_segments = []
	for cell_idx in range(num_cells):
		cell = centerline.GetCell(cell_idx)
		if cell is None:
			continue
		num_pts = cell.GetNumberOfPoints()
		if num_pts < 2:
			continue
		for j in range(num_pts - 1):
			all_segments.append((cell_idx, j))

	total_segments = len(all_segments)
	min_seg = 300
	# Use stride 1 (all segments) if fewer than min_seg; otherwise stride to get ~min_seg segments
	min_stride_for_100 = max(1, total_segments // min_seg)
	effective_stride = 1 if total_segments < min_seg else min(segment_stride, min_stride_for_100)
	segments = all_segments[::effective_stride]

	if bound_polydata_by_image is None:
		warnings.warn(
			"Cannot import bound_polydata_by_image; dice-per-radii metrics will be NaN.",
			UserWarning,
			stacklevel=2,
		)
		return _empty_dice_per_radii_result(radius_buckets)

	for cell_idx, j in tqdm(segments, desc="Dice per radii"):
		cell = centerline.GetCell(cell_idx)
		pts = cell.GetPoints()
		point_ids = cell.GetPointIds()
		pid1 = point_ids.GetId(j)
		pid2 = point_ids.GetId(j + 1)
		p1 = np.array(pts.GetPoint(j))
		p2 = np.array(pts.GetPoint(j + 1))

		r1 = float(radii[pid1]) if pid1 < len(radii) else 0.0
		r2 = float(radii[pid2]) if pid2 < len(radii) else 0.0
		r_avg = (r1 + r2) / 2.0
		segment_length = float(np.linalg.norm(p2 - p1))

		if segment_length < 1e-10:
			continue

		bucket_idx = None
		for i, (r_min, r_max) in enumerate(radius_buckets):
			if r_min <= r_avg < r_max:
				bucket_idx = i
				break
		if bucket_idx is None:
			continue

		center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)
		base_r = max(r1, r2, r_avg * 0.5)
		sphere_radius = radius_scale * base_r + radius_offset
		if sphere_radius < 1e-6:
			sphere_radius = 0.5

		roi_image = _create_roi_image(center, sphere_radius, spacing)
		dice_val = _dice_in_roi(gt_poly, pred_poly, roi_image, bound_threshold)

		raw_data.append((float(dice_val), float(r_avg)))
		bucket_data[bucket_idx].append(dice_val)

	# Build result dict
	MIN_SEGMENTS_PER_BUCKET = 5
	result = {}
	bucket_labels = []
	for i, (r_min, r_max) in enumerate(radius_buckets):
		label = f"{r_min:.1f}_{r_max:.1f}" if r_max != float('inf') else f"{r_min:.1f}_inf"
		bucket_labels.append(label)
		dice_list = bucket_data[i]
		n_seg = len(dice_list)

		if n_seg < MIN_SEGMENTS_PER_BUCKET:
			dice_mean = float('nan')
		else:
			valid = [d for d in dice_list if not math.isnan(d)]
			dice_mean = float(np.mean(valid)) if valid else float('nan')

		result[f"dice_radii_{label}"] = dice_mean
		result[f"n_segments_radii_{label}"] = n_seg

	result["dice_radii_buckets"] = bucket_labels
	result["dice_radii_raw"] = raw_data  # list of (dice, radius) per segment
	return result


def _empty_dice_per_radii_result(radius_buckets: List[Tuple[float, float]]) -> dict:
	"""Return result dict with NaN for all bucket metrics."""
	result = {"dice_radii_buckets": [], "dice_radii_raw": []}
	for r_min, r_max in radius_buckets:
		label = f"{r_min:.1f}_{r_max:.1f}" if r_max != float('inf') else f"{r_min:.1f}_inf"
		result["dice_radii_buckets"].append(label)
		result[f"dice_radii_{label}"] = float('nan')
		result[f"n_segments_radii_{label}"] = 0
	return result


def format_bucket_label_for_display(label: str, latex: bool = False) -> str:
	"""Format bucket label (e.g. '0.0_0.2', '1.6_inf') for display.

	Display values are scaled by 10 (e.g. 0.1 -> 1.0).
	"""
	parts = label.split('_')
	if len(parts) >= 2:
		if parts[1] == 'inf':
			lo = float(parts[0]) * 10
			hi = '$\\infty$' if latex else '∞'
			return f'[{lo:.1f}, {hi})'
		lo, hi = float(parts[0]) * 10, float(parts[1]) * 10
		return f'[{lo:.1f}, {hi:.1f})'
	return label


def get_dice_per_radii_fieldnames(radius_buckets: Optional[List[Tuple[float, float]]] = None) -> List[str]:
	"""Return CSV field names for dice-per-radii metrics."""
	if radius_buckets is None:
		radius_buckets = DEFAULT_RADIUS_BUCKETS
	names = []
	for r_min, r_max in radius_buckets:
		label = f"{r_min:.1f}_{r_max:.1f}" if r_max != float('inf') else f"{r_min:.1f}_inf"
		names.extend([
			f"dice_radii_{label}",
			f"n_segments_radii_{label}",
		])
	return names


def log_dice_per_radii(result: dict, logger=None) -> None:
	"""Log dice-per-radii results with corresponding radius ranges. If logger is None, uses print."""
	log = logger.info if logger is not None else print
	buckets = result.get("dice_radii_buckets", [])
	for label in buckets:
		dice_key = f"dice_radii_{label}"
		n_key = f"n_segments_radii_{label}"
		dice_val = result.get(dice_key, float('nan'))
		n_seg = result.get(n_key, 0)
		display_label = format_bucket_label_for_display(label)
		log(f"  Radii {display_label}: Dice={dice_val:.6g}, n_segments={n_seg}")


if __name__ == "__main__":
	import argparse
	import os
	import sys

	# Allow running as script
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

	parser = argparse.ArgumentParser(description="Compute Dice per radius bucket along centerline")
	parser.add_argument("gt_vtp", help="Ground truth mesh (.vtp)")
	parser.add_argument("pred_vtp", help="Prediction mesh (.vtp)")
	parser.add_argument("centerline_vtp", help="Centerline (.vtp) with MaximumInscribedSphereRadius")
	parser.add_argument("--spacing", default="1.0", help="Voxel spacing for Dice (mm). Single float or x,y,z (default: 1.0)")
	parser.add_argument("--radius-scale", type=float, default=1.2, help="Scale for ROI radius (default: 1.8)")
	parser.add_argument("--radius-offset", type=float, default=0.05, help="Offset for ROI radius (default: 0.1)")
	args = parser.parse_args()

	# Parse spacing (e.g. "1.0" or "0.5,0.5,1.0")
	parts = [p.strip() for p in args.spacing.split(',') if p.strip()]
	if len(parts) == 1:
		s = float(parts[0])
		spacing = (s, s, s)
	elif len(parts) == 3:
		spacing = (float(parts[0]), float(parts[1]), float(parts[2]))
	else:
		raise ValueError("--spacing must be single float or three comma-separated floats (e.g. 1.0 or 0.5,0.5,1.0)")

	from modules.vtk_functions import read_geo

	gt = read_geo(args.gt_vtp).GetOutput()
	pred = read_geo(args.pred_vtp).GetOutput()
	cl = read_geo(args.centerline_vtp).GetOutput()

	result = dice_per_radii(
		gt, pred, cl,
		spacing=spacing,
		radius_scale=args.radius_scale,
		radius_offset=args.radius_offset,
	)
	log_dice_per_radii(result)
