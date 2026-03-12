"""Volume error along centerlines, bucketed by MaximumInscribedSphereRadius.

Computes volume error (pred - gt) for centerline segments grouped by radius ranges,
to analyze volume loss for different branch sizes (e.g. small vs large vessels).
Negative values indicate volume loss in the prediction.

Extracts surface within a sphere around each centerline segment midpoint, closes
the clipped surface with a custom spherical cap (vertices on sphere, triangulated),
and computes volume via vtkMassProperties.
"""

from __future__ import annotations

import os
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
from scipy.spatial import Delaunay
from vtk.util import numpy_support

try:
	from ..modules.vtk_functions import bound_polydata_by_sphere, clean, appendPolyData, get_largest_connected_polydata
except ImportError:
	try:
		from modules.vtk_functions import bound_polydata_by_sphere, clean, appendPolyData, get_largest_connected_polydata
	except ImportError:
		bound_polydata_by_sphere = None
		clean = None
		appendPolyData = None
		get_largest_connected_polydata = None


# Debug counter for volume_debug_*.vtk outputs
_volume_debug_counter = 0

# When |error_rel| > 0.2, skip segment if cap_area/sphere_area exceeds this (abnormal cut)
CAP_AREA_RATIO_THRESHOLD: float = 0.1

# Default radius buckets (mm): (0, 2), (2, 4), (4, 8), (8, inf)
DEFAULT_RADIUS_BUCKETS: List[Tuple[float, float]] = [
	(0.0, 0.2),
	(0.2, 0.4),
	(0.4, 0.6),
	(0.6, 0.8),
	(1.0, 1.2),
	(1.2, 1.6),
	(1.6, float('inf')),
]


def _polydata_surface_area(poly: vtk.vtkPolyData) -> float:
	"""Sum of triangle areas in polydata (surface area)."""
	area = 0.0
	pts = poly.GetPoints()
	for i in range(poly.GetNumberOfCells()):
		cell = poly.GetCell(i)
		if cell.GetNumberOfPoints() != 3:
			continue
		p0 = np.array(pts.GetPoint(cell.GetPointId(0)))
		p1 = np.array(pts.GetPoint(cell.GetPointId(1)))
		p2 = np.array(pts.GetPoint(cell.GetPointId(2)))
		area += 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
	return area


def _boundary_loops_from_polydata(poly: vtk.vtkPolyData) -> List[np.ndarray]:
	"""Extract ordered boundary loops from polydata with open boundary.

	Returns list of (N, 3) arrays, each an ordered closed loop of 3D points.
	"""
	edges = vtk.vtkFeatureEdges()
	edges.SetInputData(poly)
	edges.BoundaryEdgesOn()
	edges.FeatureEdgesOff()
	edges.NonManifoldEdgesOff()
	edges.ManifoldEdgesOff()
	edges.Update()
	boundary = edges.GetOutput()
	if boundary is None or boundary.GetNumberOfCells() == 0:
		return []

	points = boundary.GetPoints()
	n_pts = boundary.GetNumberOfPoints()
	if n_pts == 0:
		return []

	# Build edge adjacency: for each point, list of connected point ids
	adj: dict[int, List[int]] = {i: [] for i in range(n_pts)}
	for i in range(boundary.GetNumberOfCells()):
		cell = boundary.GetCell(i)
		if cell.GetNumberOfPoints() != 2:
			continue
		a, b = cell.GetPointId(0), cell.GetPointId(1)
		adj[a].append(b)
		adj[b].append(a)

	# Trace loops
	loops = []
	used = set()

	for start in range(n_pts):
		if start in used or len(adj[start]) == 0:
			continue
		path = [start]
		used.add(start)
		curr = start
		prev = -1
		while True:
			neighbors = [n for n in adj[curr] if n != prev]
			if not neighbors:
				break
			next_pt = neighbors[0]
			if next_pt == start:
				break
			path.append(next_pt)
			used.add(next_pt)
			prev, curr = curr, next_pt
		if len(path) >= 3:
			pts_3d = np.array([points.GetPoint(i) for i in path])
			loops.append(pts_3d)
	return loops


def _point_in_polygon_2d(pts: np.ndarray, poly: np.ndarray) -> bool:
	"""Ray casting: point inside polygon (closed, first point = last)."""
	x, y = pts[0], pts[1]
	n = len(poly) - 1
	inside = False
	j = n - 1
	for i in range(n):
		yi, yj = poly[i, 1], poly[j, 1]
		if (yi > y) != (yj > y):
			dy = yj - yi
			if abs(dy) > 1e-12:
				x_cross = (poly[j, 0] - poly[i, 0]) * (y - yi) / dy + poly[i, 0]
				if x < x_cross:
					inside = not inside
		j = i
	return inside


def _project_boundary_to_2d(
	boundary_3d: np.ndarray,
	center: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Project boundary (on sphere) to 2D plane. Returns (boundary_2d, origin, basis u, v)."""
	centroid = np.mean(boundary_3d, axis=0)
	normal = centroid - np.array(center)
	norm = np.linalg.norm(normal)
	if norm < 1e-12:
		normal = np.array([1.0, 0.0, 0.0])
	else:
		normal = normal / norm

	# Orthonormal basis in plane
	ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
	u = np.cross(normal, ref)
	u_norm = np.linalg.norm(u)
	if u_norm < 1e-12:
		u = np.cross(normal, np.array([0.0, 1.0, 0.0]))
		u_norm = np.linalg.norm(u)
	u = u / u_norm
	v = np.cross(normal, u)
	v = v / np.linalg.norm(v)

	boundary_2d = np.column_stack([
		np.dot(boundary_3d - centroid, u),
		np.dot(boundary_3d - centroid, v),
	])
	return boundary_2d, centroid, u, v


def _sample_interior_points_2d(
	boundary_2d: np.ndarray,
	min_points: int = 15,
	grid_res: int = 20,
) -> np.ndarray:
	"""Sample points inside polygon using uniform grid + point-in-polygon.

	Uses Poisson-like spacing by taking a grid and keeping points inside.
	"""
	poly_closed = np.vstack([boundary_2d, boundary_2d[0:1]])
	xmin, ymin = boundary_2d.min(axis=0)
	xmax, ymax = boundary_2d.max(axis=0)
	dx = max((xmax - xmin) / max(grid_res, 1), 1e-10)
	dy = max((ymax - ymin) / max(grid_res, 1), 1e-10)

	interior = []
	for i in range(grid_res + 1):
		for j in range(grid_res + 1):
			x = xmin + i * dx
			y = ymin + j * dy
			if _point_in_polygon_2d(np.array([x, y]), poly_closed):
				interior.append([x, y])

	interior = np.array(interior) if interior else np.empty((0, 2))
	while len(interior) < min_points and grid_res < 80:
		grid_res += 5
		dx = max((xmax - xmin) / grid_res, 1e-10)
		dy = max((ymax - ymin) / grid_res, 1e-10)
		interior = []
		for i in range(grid_res + 1):
			for j in range(grid_res + 1):
				x = xmin + i * dx
				y = ymin + j * dy
				if _point_in_polygon_2d(np.array([x, y]), poly_closed):
					interior.append([x, y])
		interior = np.array(interior) if interior else np.empty((0, 2))
	return interior


def _project_2d_to_sphere(
	pts_2d: np.ndarray,
	centroid: np.ndarray,
	u: np.ndarray,
	v: np.ndarray,
	center: Tuple[float, float, float],
	radius: float,
) -> np.ndarray:
	"""Project 2D points (in plane) back to sphere surface."""
	if len(pts_2d) == 0:
		return np.empty((0, 3))
	pts_3d_plane = centroid + pts_2d[:, 0:1] * u + pts_2d[:, 1:2] * v
	vec = pts_3d_plane - np.array(center)
	dist = np.linalg.norm(vec, axis=1, keepdims=True)
	dist = np.maximum(dist, 1e-12)
	pts_sphere = np.array(center) + radius * (vec / dist)
	return pts_sphere


def _triangulate_cap(
	boundary_3d: np.ndarray,
	interior_3d: np.ndarray,
	center: Tuple[float, float, float],
	radius: float,
) -> np.ndarray:
	"""Triangulate boundary + interior points. Returns (N, 3) triangle indices."""
	n_bound = len(boundary_3d)

	# Project to 2D for Delaunay
	boundary_2d, centroid, u, v = _project_boundary_to_2d(boundary_3d, center)
	if len(interior_3d) > 0:
		interior_2d = np.column_stack([
			np.dot(interior_3d - centroid, u),
			np.dot(interior_3d - centroid, v),
		])
		all_2d = np.vstack([boundary_2d, interior_2d])
	else:
		all_2d = boundary_2d

	poly_closed = np.vstack([boundary_2d, boundary_2d[0:1]])
	tri = Delaunay(all_2d)
	triangles = []
	for simplex in tri.simplices:
		c = all_2d[simplex].mean(axis=0)
		if _point_in_polygon_2d(c, poly_closed):
			triangles.append(simplex)
	if not triangles:
		# Fallback: fan from centroid on sphere
		centroid_3d = boundary_3d.mean(axis=0)
		vec = centroid_3d - np.array(center)
		vec_norm = np.linalg.norm(vec)
		if vec_norm < 1e-12:
			vec = boundary_3d[0] - np.array(center)
			vec_norm = np.linalg.norm(vec)
		for i in range(len(boundary_3d)):
			j = (i + 1) % len(boundary_3d)
			triangles.append([i, j, n_bound])
		return np.array(triangles)

	return np.array(triangles)


def _close_mesh_with_sphere_cap(
	clipped: vtk.vtkPolyData,
	center: Tuple[float, float, float],
	radius: float,
) -> Tuple[Optional[vtk.vtkPolyData], float]:
	"""Close clipped mesh by adding triangulated caps on the sphere surface for all holes.

	Returns
	-------
	tuple
		(closed_polydata, total_cap_area) or (None, 0.0) if no boundary loops.
	"""
	loops = _boundary_loops_from_polydata(clipped)
	if not loops:
		return None, 0.0

	cap_polys = []
	for loop in loops:
		boundary_3d = np.asarray(loop, dtype=np.float64)

		boundary_2d, centroid, u, v = _project_boundary_to_2d(boundary_3d, center)
		interior_2d = _sample_interior_points_2d(boundary_2d)
		interior_3d = _project_2d_to_sphere(interior_2d, centroid, u, v, center, radius)

		tri_indices = _triangulate_cap(boundary_3d, interior_3d, center, radius)

		all_cap_pts = np.vstack([boundary_3d, interior_3d]) if len(interior_3d) > 0 else boundary_3d
		n_bound = len(boundary_3d)
		if len(interior_3d) == 0:
			centroid_3d = boundary_3d.mean(axis=0)
			vec = centroid_3d - np.array(center)
			vec_norm = np.linalg.norm(vec)
			if vec_norm < 1e-12:
				vec = boundary_3d[0] - np.array(center)
				vec_norm = np.linalg.norm(vec)
			centroid_on_sphere = np.array(center) + radius * (vec / vec_norm)
			all_cap_pts = np.vstack([boundary_3d, centroid_on_sphere.reshape(1, 3)])

		cap_points = vtk.vtkPoints()
		for i in range(len(all_cap_pts)):
			cap_points.InsertNextPoint(all_cap_pts[i])

		cells = vtk.vtkCellArray()
		for tri in tri_indices:
			cells.InsertNextCell(3)
			for idx in tri:
				cells.InsertCellPoint(idx)

		cap_poly = vtk.vtkPolyData()
		cap_poly.SetPoints(cap_points)
		cap_poly.SetPolys(cells)
		cap_polys.append(cap_poly)

	total_cap_area = sum(_polydata_surface_area(cp) for cp in cap_polys)

	# Append clipped mesh + all caps
	to_append = [clipped] + cap_polys
	if appendPolyData is not None:
		closed = appendPolyData(to_append)
	else:
		append_filter = vtk.vtkAppendPolyData()
		for poly in to_append:
			append_filter.AddInputData(poly)
		append_filter.Update()
		closed = append_filter.GetOutput()

	# Fill any remaining holes between caps and mesh (gaps at boundaries)
	try:
		fill_holes = vtk.vtkFillHolesFilter()
		fill_holes.SetInputData(closed)
		fill_holes.SetHoleSize(radius * 2.0)  # max hole size to fill (circumsphere radius)
		fill_holes.Update()
		closed = fill_holes.GetOutput()
	except Exception:
		pass  # fallback: use mesh without hole filling if filter unavailable

	if clean is not None:
		closed = clean(closed)
	return closed, total_cap_area


def _volume_of_sphere_clipped_mesh(
	poly: vtk.vtkPolyData,
	center: Tuple[float, float, float],
	radius: float,
) -> Tuple[float, float]:
	"""Volume of mesh region inside a sphere.

	Clips the mesh to the sphere interior, closes the boundary with a custom spherical
	cap (vertices on sphere, triangulated with interior points), and computes
	enclosed volume via vtkMassProperties.

	Parameters
	----------
	poly : vtk.vtkPolyData
		Input closed surface mesh
	center : (x, y, z)
		Sphere center
	radius : float
		Sphere radius

	Returns
	-------
	tuple of (float, float)
		(volume, cap_area). Volume is 0.0 if clipped mesh is empty, not fully closed,
		or volume computation fails. cap_area is the total area of spherical caps used
		to close the clipped surface.
	"""
	if bound_polydata_by_sphere is None:
		return 0.0, 0.0
	clipped = bound_polydata_by_sphere(poly, center, radius)
	if clipped is None or clipped.GetNumberOfCells() == 0:
		return 0.0, 0.0
	if get_largest_connected_polydata is not None:
		clipped = get_largest_connected_polydata(clipped)
		if clipped is None or clipped.GetNumberOfCells() == 0:
			return 0.0, 0.0
	# Remesh: clean duplicate points, then ensure triangle mesh
	if clean is not None:
		clipped = clean(clipped)
		if clipped is None or clipped.GetNumberOfCells() == 0:
			return 0.0, 0.0
	tri_filter = vtk.vtkTriangleFilter()
	tri_filter.SetInputData(clipped)
	tri_filter.Update()
	clipped = tri_filter.GetOutput()
	if clipped is None or clipped.GetNumberOfCells() == 0:
		return 0.0, 0.0
	try:
		closed, cap_area = _close_mesh_with_sphere_cap(clipped, center, radius)
		if closed is None or closed.GetNumberOfCells() == 0:
			return 0.0, 0.0

		# Debug: write filled polydata for inspection
		# try:
		# 	global _volume_debug_counter
		# 	_volume_debug_counter += 1
		# 	outpath = os.path.join("/Users/nsveinsson/Downloads", f"volume_debug_{_volume_debug_counter:04d}.vtk")
		# 	writer = vtk.vtkPolyDataWriter()
		# 	writer.SetFileName(outpath)
		# 	writer.SetInputData(closed)
		# 	writer.Write()
		# except Exception:
		# 	pass

		mass = vtk.vtkMassProperties()
		mass.SetInputData(closed)
		mass.Update()
		return float(mass.GetVolume()), float(cap_area)
	except Exception:
		return 0.0, 0.0


def volume_error_per_radii(
	gt_poly: vtk.vtkPolyData,
	pred_poly: vtk.vtkPolyData,
	centerline: vtk.vtkPolyData,
	radius_buckets: Optional[List[Tuple[float, float]]] = None,
	segment_stride: int = 20,
	radius_scale: float = 3.0,
	radius_offset: float = 0.1,
	cap_area_ratio_threshold: Optional[float] = None,
) -> dict:
	"""Compute volume error along centerline segments, bucketed by MaximumInscribedSphereRadius.

	For each centerline segment midpoint, clips GT and Pred meshes to a sphere around
	the point, closes the clipped surface, and computes volume via vtkMassProperties.

	Segments are grouped by mean radius into buckets. For each segment, relative error
	(pred - gt) / gt is computed; the bucket metric is the mean of these per-segment
	relative errors (negative = volume loss in prediction).

	Parameters
	----------
	gt_poly : vtk.vtkPolyData
		Ground truth surface mesh
	pred_poly : vtk.vtkPolyData
		Prediction surface mesh
	centerline : vtk.vtkPolyData
		Centerline with 'MaximumInscribedSphereRadius' point data
	radius_buckets : list of (min_r, max_r), optional
		Radius ranges in mesh units (e.g. mm). Default: [(0,2), (2,4), (4,8), (8,inf)]
	segment_stride : int
		Process every Nth segment when there are 100+ segments. If fewer than 100 segments
		exist, all are used. Stride is capped so at least 100 segments are computed.
	radius_scale : float
		Scale factor for sphere radius. MaximumInscribedSphereRadius often underestimates;
		use > 1 to capture the full vessel lumen.
	radius_offset : float
		Additive offset (in mesh units). Formula: radius = radius_scale * r + radius_offset.
	cap_area_ratio_threshold : float, optional
		When |error_rel| > 0.2, skip segment if cap_area/sphere_area exceeds this ratio
		(abnormal cut perpendicular to vessel). Default: CAP_AREA_RATIO_THRESHOLD (0.5).

	Returns
	-------
	dict
		Keys for each bucket:
		  - 'volume_error_radii_<min>_<max>_rel' : mean of per-segment relative errors (pred - gt) / gt
		  - 'volume_gt_radii_<min>_<max>' : estimated GT volume in bucket
		  - 'volume_pred_radii_<min>_<max>' : estimated Pred volume in bucket
		Plus 'volume_error_radii_buckets' : list of bucket labels for reference
	"""
	if radius_buckets is None:
		radius_buckets = DEFAULT_RADIUS_BUCKETS

	radii_arr = (
		centerline.GetPointData().GetArray('MaximumInscribedSphereRadius')
		or centerline.GetPointData().GetArray('Radius')
	)
	if radii_arr is None:
		warnings.warn(
			"Centerline missing 'MaximumInscribedSphereRadius' and 'Radius' point data; "
			"volume-per-radii metrics will be NaN.",
			UserWarning,
			stacklevel=2,
		)
		return _empty_volume_per_radii_result(radius_buckets)

	radii = np.asarray(numpy_support.vtk_to_numpy(radii_arr)).flatten()

	# Accumulate per bucket: (sum_vol_gt, sum_vol_pred, n_segments, list of per-segment err_rel)
	bucket_data = {i: [0.0, 0.0, 0, []] for i in range(len(radius_buckets))}
	raw_data: List[Tuple[float, float]] = []  # (error_rel, radius) for each segment

	num_cells = centerline.GetNumberOfCells()
	if num_cells == 0:
		return _empty_volume_per_radii_result(radius_buckets)

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
	# Use stride 1 (all segments) if fewer than 100; otherwise stride to get ~100 segments
	min_stride_for_100 = max(1, total_segments // min_seg)
	effective_stride = 1 if total_segments < min_seg else min(segment_stride, min_stride_for_100)
	segments = all_segments[::effective_stride]

	if bound_polydata_by_sphere is None:
		warnings.warn(
			"Cannot import bound_polydata_by_sphere; volume-per-radii metrics will be NaN.",
			UserWarning,
			stacklevel=2,
		)
		return _empty_volume_per_radii_result(radius_buckets)

	for cell_idx, j in tqdm(segments, desc="Volume per radii"):
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

		vol_gt, cap_area_gt = _volume_of_sphere_clipped_mesh(gt_poly, center, sphere_radius)
		vol_pred, cap_area_pred = _volume_of_sphere_clipped_mesh(pred_poly, center, sphere_radius)
		# print(f"Cap area ratio: {cap_area_gt / cap_area_pred}")
		# if abs(1-(cap_area_gt / cap_area_pred)) > 0.1:
			# import pdb; pdb.set_trace()
		if vol_gt > 1e-12 and vol_pred > 1e-12:
			error_rel = (vol_pred - vol_gt) / vol_gt
			# print(f"Error rel: {error_rel}")
			# Skip if error is high and caps are abnormally large (cut perpendicular to vessel)
			if abs(error_rel) > 0.2:
				threshold = cap_area_ratio_threshold if cap_area_ratio_threshold is not None else CAP_AREA_RATIO_THRESHOLD
				sphere_area = 4.0 * np.pi * sphere_radius ** 2
				max_cap_ratio = max(cap_area_gt, cap_area_pred) / sphere_area if sphere_area > 1e-12 else 0.0
				print(f"Max cap ratio: {max_cap_ratio}")
				if max_cap_ratio > threshold:
					continue  # abnormal cut, skip segment 

			raw_data.append((float(error_rel), float(r_avg)))
			bucket_data[bucket_idx][0] += vol_gt
			bucket_data[bucket_idx][1] += vol_pred
			bucket_data[bucket_idx][2] += 1
			if abs(error_rel) < 0.35:
				bucket_data[bucket_idx][3].append(error_rel)

	# Build result dict
	MIN_SEGMENTS_PER_BUCKET = 5
	result = {}
	bucket_labels = []
	for i, (r_min, r_max) in enumerate(radius_buckets):
		label = f"{r_min:.1f}_{r_max:.1f}" if r_max != float('inf') else f"{r_min:.1f}_inf"
		bucket_labels.append(label)
		vol_gt = bucket_data[i][0]
		vol_pred = bucket_data[i][1]
		n_seg = bucket_data[i][2]
		err_rel_list = bucket_data[i][3]

		if n_seg < MIN_SEGMENTS_PER_BUCKET:
			err_rel = float('nan')
			vol_gt = float('nan')
			vol_pred = float('nan')
		else:
			err_rel = float(np.mean(err_rel_list)) if err_rel_list else float('nan')

		key_prefix = f"volume_error_radii_{label}"
		result[f"{key_prefix}_rel"] = err_rel
		result[f"volume_gt_radii_{label}"] = vol_gt
		result[f"volume_pred_radii_{label}"] = vol_pred
		result[f"n_segments_radii_{label}"] = n_seg

	result["volume_error_radii_buckets"] = bucket_labels
	result["volume_error_radii_raw"] = raw_data  # list of (error_rel, radius) per segment
	return result


def _empty_volume_per_radii_result(radius_buckets: List[Tuple[float, float]]) -> dict:
	"""Return result dict with NaN for all bucket metrics."""
	result = {"volume_error_radii_buckets": [], "volume_error_radii_raw": []}
	for r_min, r_max in radius_buckets:
		label = f"{r_min:.1f}_{r_max:.1f}" if r_max != float('inf') else f"{r_min:.1f}_inf"
		result["volume_error_radii_buckets"].append(label)
		result[f"volume_error_radii_{label}_rel"] = float('nan')
		result[f"volume_gt_radii_{label}"] = float('nan')
		result[f"volume_pred_radii_{label}"] = float('nan')
		result[f"n_segments_radii_{label}"] = 0
	return result


def format_bucket_label_for_display(label: str, latex: bool = False) -> str:
	"""Format bucket label (e.g. '0.0_0.2', '1.6_inf') for display.

	Labels must match the format from result keys: f"{r_min:.1f}_{r_max:.1f}" or f"{r_min:.1f}_inf".
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


def get_volume_per_radii_fieldnames(radius_buckets: Optional[List[Tuple[float, float]]] = None) -> List[str]:
	"""Return CSV field names for volume-per-radii metrics."""
	if radius_buckets is None:
		radius_buckets = DEFAULT_RADIUS_BUCKETS
	names = []
	for r_min, r_max in radius_buckets:
		label = f"{r_min:.1f}_{r_max:.1f}" if r_max != float('inf') else f"{r_min:.1f}_inf"
		names.extend([
			f"volume_error_radii_{label}_rel",
			f"volume_gt_radii_{label}",
			f"volume_pred_radii_{label}",
			f"n_segments_radii_{label}",
		])
	return names
