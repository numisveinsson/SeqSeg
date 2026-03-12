"""Compute metrics between prediction folders (pred vs pred, no ground truth).

Compares each pair of prediction subfolders under a root directory. For each pair
(e.g. obs1 vs obs2, obs1 vs obs3, obs2 vs obs3), computes the same metrics as
compute_metrics_meshes_comparison, treating the first folder as reference and the
second as prediction.

Usage:
	python -m seqseg.analysis.compute_metrics_meshes_pred_vs_pred /path/to/predictions_root

Example:
	# basic run: compare all pairs of subfolders under predictions_root
	python -m seqseg.analysis.compute_metrics_meshes_pred_vs_pred /data/predictions_root

	# specify output directory and metrics
	python -m seqseg.analysis.compute_metrics_meshes_pred_vs_pred /data/predictions_root \
			--out-dir /data/output --metrics distance,dice,volume

	# with centerline and clipping (same as compute_metrics_meshes_comparison)
	python -m seqseg.analysis.compute_metrics_meshes_pred_vs_pred /data/predictions_root \
			--centerline-dir /data/centerlines --clip

Output:
	- One .csv per comparison pair: <ref>_vs_<pred>.csv (e.g. obs1_vs_obs2.csv)
	- summary.csv: one row per comparison with mean and std for each metric
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from typing import List, Optional, Tuple

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable, desc=None, disable=False, **kwargs):
		if disable:
			return iterable
		return iterable

# Support both module run (python -m seqseg.analysis...) and direct run (python path/to/script.py)
if __package__:
	from .compute_metrics_meshes_comparison import (
		RESULTS_FIELDNAMES,
		_rows_mean_std,
		clip_surface_mesh,
		compute_metrics,
		find_matching_vtps,
		load_vtp,
		parse_metrics_arg,
		parse_spacing_arg,
		write_dice_radii_raw_csv,
		write_results_csv,
		write_volume_radii_raw_csv,
	)
	try:
		from ..modules import vtk_functions as vf
	except ImportError:
		vf = None
else:
	_script_dir = os.path.dirname(os.path.abspath(__file__))
	_repo_root = os.path.dirname(os.path.dirname(_script_dir))
	if _repo_root not in sys.path:
		sys.path.insert(0, _repo_root)
	from seqseg.analysis.compute_metrics_meshes_comparison import (
		RESULTS_FIELDNAMES,
		_rows_mean_std,
		clip_surface_mesh,
		compute_metrics,
		find_matching_vtps,
		load_vtp,
		parse_metrics_arg,
		parse_spacing_arg,
		write_dice_radii_raw_csv,
		write_results_csv,
		write_volume_radii_raw_csv,
	)
	try:
		from seqseg.modules import vtk_functions as vf
	except ImportError:
		vf = None


def write_summary_csv(summary_csv: str, summary_rows: List[dict]) -> None:
	"""Write summary CSV: one row per comparison pair with mean and std for each metric."""
	metric_keys = [k for k in RESULTS_FIELDNAMES if k != 'case']
	fieldnames = ['comparison'] + [f'{k}_mean' for k in metric_keys] + [f'{k}_std' for k in metric_keys]
	with open(summary_csv, 'w', newline='') as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in summary_rows:
			w.writerow({k: r.get(k, '') for k in fieldnames})


def run_metrics_for_pair(
	ref_dir: str,
	pred_dir: str,
	ref_name: str,
	pred_name: str,
	ext: str,
	max_points: Optional[int],
	voxel_spacing: Optional[Tuple[float, float, float]],
	clip: bool,
	centerline_dir: Optional[str],
	clip_temp_dir: Optional[str],
	quiet: bool,
	metrics_to_compute: Optional[frozenset] = None,
) -> List[dict]:
	"""Run metrics for all matching cases between ref_dir and pred_dir.

	ref_dir is treated as reference (like GT), pred_dir as prediction.
	Returns list of per-case rows.
	"""
	pairs = find_matching_vtps(ref_dir, pred_dir, ext=ext)
	if not pairs:
		return []

	rows = []
	for case, ref_path, pred_path in tqdm(pairs, desc=f'{ref_name} vs {pred_name}', disable=quiet):
		try:
			ref_poly = load_vtp(ref_path)
			pred_poly = load_vtp(pred_path)

			if clip and centerline_dir is not None:
				centerline_path = os.path.join(centerline_dir, case + ext)
				if os.path.exists(centerline_path):
					try:
						centerline = vf.read_geo(centerline_path).GetOutput() if vf is not None else load_vtp(centerline_path)
						ref_poly = clip_surface_mesh(ref_poly, centerline, case_name=case, temp_dir=clip_temp_dir)
						pred_poly = clip_surface_mesh(pred_poly, centerline, case_name=case, temp_dir=clip_temp_dir)
					except Exception:
						pass

			centerline = None
			if centerline_dir and os.path.exists(os.path.join(centerline_dir, case + ext)):
				try:
					centerline = (
						vf.read_geo(os.path.join(centerline_dir, case + ext)).GetOutput()
						if vf is not None
						else load_vtp(os.path.join(centerline_dir, case + ext))
					)
				except Exception:
					pass

			metrics = compute_metrics(
				ref_poly,
				pred_poly,
				max_points=max_points,
				voxel_spacing=voxel_spacing,
				centerline=centerline,
				metrics_to_compute=metrics_to_compute,
			)
			row = {'case': case}
			row.update(metrics)
			rows.append(row)
		except Exception as e:
			if not quiet:
				print(f'  Error processing {case}: {e}', file=sys.stderr)
	return rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description='Compute metrics between prediction folders (pairwise comparison, no ground truth)'
	)
	p.add_argument(
		'predictions_root',
		help='Root directory containing prediction subfolders (e.g. obs1, obs2, obs3)',
	)
	p.add_argument(
		'--out-dir',
		default=None,
		help='Output directory for CSVs (default: predictions_root)',
	)
	p.add_argument(
		'--summary-csv',
		default=None,
		help='Output path for summary CSV (default: <out-dir>/summary.csv)',
	)
	p.add_argument(
		'--metrics',
		type=parse_metrics_arg,
		default=None,
		metavar='M1,M2,...',
		help='Comma-separated metrics to compute. Default: all.',
	)
	p.add_argument(
		'--max-points',
		type=int,
		default=None,
		help='If set, randomly sample at most this many points from each mesh',
	)
	p.add_argument(
		'--ext',
		default='.vtp',
		help='File extension to look for (default: .vtp)',
	)
	p.add_argument(
		'--spacing',
		default='1.0',
		help='Voxel spacing for Dice voxelization. Single float or comma-separated three floats.',
	)
	p.add_argument(
		'--centerline-dir',
		default=None,
		help='Directory containing centerline .vtp files. Required if --clip is used.',
	)
	p.add_argument(
		'--clip',
		action='store_true',
		help='Clip meshes using centerlines before computing metrics. Requires --centerline-dir.',
	)
	p.add_argument(
		'--clip-temp-dir',
		default=None,
		help='Temporary directory for clipping box files',
	)
	p.add_argument(
		'--quiet',
		action='store_true',
		help='Reduce logging',
	)
	return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	predictions_root = args.predictions_root
	out_dir = args.out_dir or predictions_root
	summary_csv = args.summary_csv or os.path.join(out_dir, 'summary.csv')
	ext = args.ext if args.ext.startswith('.') else '.' + args.ext
	clip = args.clip
	centerline_dir = args.centerline_dir
	clip_temp_dir = args.clip_temp_dir
	metrics_to_compute = args.metrics

	spacing = parse_spacing_arg(args.spacing) if getattr(args, 'spacing', None) is not None else None

	if not os.path.isdir(predictions_root):
		print(f'Error: Predictions root is not a directory: {predictions_root}', file=sys.stderr)
		return 1

	subdirs = sorted([
		d for d in os.listdir(predictions_root)
		if os.path.isdir(os.path.join(predictions_root, d))
	])
	if len(subdirs) < 2:
		print(
			f'Error: Need at least 2 subdirectories for comparison, found {len(subdirs)}',
			file=sys.stderr,
		)
		return 1

	os.makedirs(out_dir, exist_ok=True)

	if clip:
		if centerline_dir is None:
			print('Error: --centerline-dir is required when --clip is used', file=sys.stderr)
			return 1
		if not os.path.isdir(centerline_dir):
			print(f'Error: Centerline directory does not exist: {centerline_dir}', file=sys.stderr)
			return 1

	metric_keys = [k for k in RESULTS_FIELDNAMES if k != 'case']
	summary_rows = []
	pairs_processed = 0

	for i, j in itertools.combinations(range(len(subdirs)), 2):
		ref_name = subdirs[i]
		pred_name = subdirs[j]
		ref_dir = os.path.join(predictions_root, ref_name)
		pred_dir = os.path.join(predictions_root, pred_name)
		comparison_name = f'{ref_name}_vs_{pred_name}'

		if not args.quiet:
			print(f'Comparison: {comparison_name}')

		rows = run_metrics_for_pair(
			ref_dir,
			pred_dir,
			ref_name,
			pred_name,
			ext,
			args.max_points,
			spacing,
			clip,
			centerline_dir,
			clip_temp_dir,
			args.quiet,
			metrics_to_compute=metrics_to_compute,
		)

		if not rows:
			if not args.quiet:
				print(f'  No matching cases for {comparison_name}', file=sys.stderr)
			continue

		# Per-comparison CSV
		per_pair_csv = os.path.join(out_dir, f'{comparison_name}.csv')
		write_results_csv(per_pair_csv, rows)
		if not args.quiet:
			print(f'  Wrote {per_pair_csv} ({len(rows)} cases)')

		# Raw volume-per-radii and dice-per-radii (optional)
		if metrics_to_compute is None or 'volume_radii' in metrics_to_compute:
			raw_path = os.path.join(out_dir, f'volume_radii_raw_{comparison_name}.csv')
			write_volume_radii_raw_csv(raw_path, rows)
			n_raw = sum(len(r.get('volume_error_radii_raw') or []) for r in rows)
			if n_raw > 0 and not args.quiet:
				print(f'  Wrote {raw_path} ({n_raw} segments)')

		if metrics_to_compute is None or 'dice_radii' in metrics_to_compute:
			raw_path = os.path.join(out_dir, f'dice_radii_raw_{comparison_name}.csv')
			write_dice_radii_raw_csv(raw_path, rows)
			n_raw = sum(len(r.get('dice_radii_raw') or []) for r in rows)
			if n_raw > 0 and not args.quiet:
				print(f'  Wrote {raw_path} ({n_raw} segments)')

		mean_row, std_row = _rows_mean_std(rows, metric_keys)
		summary_rows.append({
			'comparison': comparison_name,
			**{f'{k}_mean': mean_row[k] for k in metric_keys},
			**{f'{k}_std': std_row[k] for k in metric_keys},
		})
		pairs_processed += 1

	if summary_rows:
		write_summary_csv(summary_csv, summary_rows)
		if not args.quiet:
			print(f'\nWrote summary to {summary_csv} ({len(summary_rows)} comparisons)')
	else:
		print('No metrics computed for any comparison pair.', file=sys.stderr)
		return 2

	return 0


if __name__ == '__main__':
	raise SystemExit(main())
