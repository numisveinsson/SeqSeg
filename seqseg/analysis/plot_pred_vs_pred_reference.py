"""Plot pred-vs-pred metric differences: reference vs others vs others amongst themselves.

Reads the per-comparison CSV files from compute_metrics_meshes_pred_vs_pred and
designates one prediction as the 'reference'. Compares:
  1) Reference vs Others: metric values from comparisons involving the reference
  2) Others vs Others: metric values from pairwise comparisons between non-reference preds

Violin plots show whether the reference is closer or further from the other preds
than they are from each other.

With --point-comparison: for each case (e.g. data0), compute |pairwise differences|
within each group. E.g. |data0_obs0 - data0_obs1| = |(ref vs pred1) - (ref vs pred2)|
for the same case. Plots the distribution of these absolute case-wise differences.

Usage:
	python -m seqseg.analysis.plot_pred_vs_pred_reference /path/to/output_dir --reference obs1
	python -m seqseg.analysis.plot_pred_vs_pred_reference /path/to/output_dir --reference obs1 --point-comparison
	python -m seqseg.analysis.plot_pred_vs_pred_reference /path/to/output_dir --reference obs1 --metrics dice,hausdorff_sym,volume_error_rel
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
	from seqseg.analysis.plotting.latex_table import (
		write_point_diff_latex_table,
		write_pred_vs_pred_reference_latex_table,
	)
	from seqseg.analysis.plotting.violin_plot_functions import (
		NATURE_COLORS_DEFAULT as NATURE_COLORS,
		apply_nature_style,
		draw_violin_ax,
		save_violin_figure,
	)
except ImportError:
	# Allow running as script: python seqseg/analysis/plot_pred_vs_pred_reference.py
	_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	if _root not in sys.path:
		sys.path.insert(0, _root)
	from seqseg.analysis.plotting.latex_table import (
		write_point_diff_latex_table,
		write_pred_vs_pred_reference_latex_table,
	)
	from seqseg.analysis.plotting.violin_plot_functions import (
		NATURE_COLORS_DEFAULT as NATURE_COLORS,
		apply_nature_style,
		draw_violin_ax,
		save_violin_figure,
	)


# Main scalar metrics to plot (excludes case, n_*, and radius-bucket columns)
DEFAULT_METRICS = [
	'hausdorff_sym',
	'hd95_sym',
	'assd',
	'dice',
	'volume_error_rel',
	'surface_area_error_rel',
	'surface_dice_t1',
	'surface_dice_t2',
	'centerline_overlap',
]

# Display names for axis labels and titles
METRIC_DISPLAY_NAMES = {
	'hausdorff_sym': 'Hausdorff (sym)',
	'hd95_sym': 'HD 95th',
	'assd': 'ASSD',
	'dice': 'Dice',
	'volume_error_rel': 'Volume error (rel)',
	'surface_area_error_rel': 'Relative Surface Area Error',
	'surface_dice_t1': 'Surface Dice τ1',
	'surface_dice_t2': 'Surface Dice τ2',
	'centerline_overlap': 'Centerline overlap',
	'mean_normal_angular_error_gt_to_pred': 'Average Normal Difference (°)',
	'max_normal_angular_error_gt_to_pred': 'Max Normal Difference (°)',
	'std_normal_angular_error_gt_to_pred': 'Std Normal Difference (°)',
}

# Dice-like metrics: [0, 1] range, ylim upper = 1
DICE_METRICS = {'dice', 'surface_dice_t1', 'surface_dice_t2', 'centerline_overlap'}
# Distance/error metrics: lower bound 0
DISTANCE_METRICS = {
	'hausdorff_sym', 'hd95_sym', 'assd', 'volume_error_rel', 'surface_area_error_rel',
	'mean_normal_angular_error_gt_to_pred', 'max_normal_angular_error_gt_to_pred', 'std_normal_angular_error_gt_to_pred',
}
# Relative error metrics: use absolute value before plotting/stats
RELATIVE_ERROR_METRICS = {'volume_error_rel', 'surface_area_error_rel'}


def _metric_label(metric: str) -> str:
	return METRIC_DISPLAY_NAMES.get(metric, metric)


def _tick_labels(reference: str) -> dict[str, str]:
	"""X-axis tick labels for the two groups."""
	if reference.lower() == 'seqseg':
		return {'Reference vs Others': 'SeqSeg vs Manual', 'Others vs Others': 'Manual vs Manual'}
	return {'Reference vs Others': f'{reference} vs Others', 'Others vs Others': 'Others vs Others'}


def _set_ylim(ax, metric: str) -> None:
	"""Set y-axis limits: upper 1 for dice (lower auto), lower 0 for distances."""
	if metric in DICE_METRICS:
		ax.set_ylim(top=1)
	elif metric in DISTANCE_METRICS:
		ax.set_ylim(bottom=0)


def _find_comparison_csvs(indir: str) -> list[tuple[str, str]]:
	"""Find *_vs_*.csv files. Returns [(comparison_name, path), ...]."""
	pattern = os.path.join(indir, '*_vs_*.csv')
	paths = glob.glob(pattern)
	results = []
	for p in paths:
		basename = os.path.basename(p)
		name = basename[:-4] if basename.endswith('.csv') else basename
		if '_vs_' in name:
			results.append((name, p))
	return sorted(results, key=lambda x: x[0])


def _parse_comparison_name(name: str) -> tuple[str, str]:
	"""Parse 'A_vs_B' -> (A, B)."""
	parts = name.split('_vs_', 1)
	if len(parts) != 2:
		raise ValueError(f'Invalid comparison name: {name}')
	return parts[0], parts[1]


def _load_all_comparisons(indir: str) -> pd.DataFrame | None:
	"""Load all per-comparison CSVs into one DataFrame with 'comparison' column."""
	pairs = _find_comparison_csvs(indir)
	if not pairs:
		return None

	dfs = []
	for comp_name, path in pairs:
		try:
			df = pd.read_csv(path)
			df = df[df['case'].astype(str).str.upper() != 'MEAN']
			df = df[df['case'].astype(str).str.upper() != 'STD']
			df['comparison'] = comp_name
			dfs.append(df)
		except Exception as e:
			print(f'Warning: Could not read {path}: {e}', file=sys.stderr)

	if not dfs:
		return None
	return pd.concat(dfs, ignore_index=True)


def _assign_group(comparison_name: str, reference: str) -> str:
	"""Assign comparison to 'Reference vs Others' or 'Others vs Others'."""
	a, b = _parse_comparison_name(comparison_name)
	if a == reference or b == reference:
		return 'Reference vs Others'
	return 'Others vs Others'


def _get_numeric_metric_columns(df: pd.DataFrame, metrics: list[str] | None) -> list[str]:
	"""Return list of numeric metric columns to plot."""
	exclude = {'case', 'comparison'}
	radius_prefixes = ('volume_error_radii_', 'dice_radii_', 'volume_radii_')
	numeric = []
	for c in df.columns:
		if c in exclude:
			continue
		if any(c.startswith(p) for p in radius_prefixes):
			continue
		if df[c].dtype in (np.float64, np.int64, np.float32, np.int32):
			numeric.append(c)

	if metrics is not None:
		numeric = [c for c in numeric if c in metrics]
	return sorted(numeric)


def _get_available_references(indir: str) -> set[str]:
	"""Return set of pred names that appear in comparison files."""
	pairs = _find_comparison_csvs(indir)
	names = set()
	for comp_name, _ in pairs:
		a, b = _parse_comparison_name(comp_name)
		names.add(a)
		names.add(b)
	return names


def _compute_point_differences(
	df: pd.DataFrame,
	metric_cols: list[str],
	reference: str,
) -> dict[str, dict[str, np.ndarray]]:
	"""For each case, compute pairwise metric differences within each group.

	Returns: {metric: {'Reference vs Others': array of diffs, 'Others vs Others': array of diffs}}
	"""
	ref_comps = [
		c for c in df['comparison'].unique()
		if _assign_group(c, reference) == 'Reference vs Others'
	]
	others_comps = [
		c for c in df['comparison'].unique()
		if _assign_group(c, reference) == 'Others vs Others'
	]

	result = {m: {'Reference vs Others': [], 'Others vs Others': []} for m in metric_cols}

	for case in df['case'].unique():
		ref_rows = df[(df['case'] == case) & (df['comparison'].isin(ref_comps))]
		others_rows = df[(df['case'] == case) & (df['comparison'].isin(others_comps))]

		for metric in metric_cols:
			# Reference vs Others: pairwise diffs between (ref vs pred_i) and (ref vs pred_j)
			ref_vals = ref_rows.set_index('comparison')[metric].dropna()
			if len(ref_vals) >= 2:
				for (c1, v1), (c2, v2) in itertools.combinations(ref_vals.items(), 2):
					result[metric]['Reference vs Others'].append(abs(v1 - v2))

			# Others vs Others: pairwise diffs between (pred_i vs pred_j) and (pred_k vs pred_l)
			others_vals = others_rows.set_index('comparison')[metric].dropna()
			if len(others_vals) >= 2:
				for (c1, v1), (c2, v2) in itertools.combinations(others_vals.items(), 2):
					result[metric]['Others vs Others'].append(abs(v1 - v2))

	for metric in metric_cols:
		for group in result[metric]:
			result[metric][group] = np.array(result[metric][group]) if result[metric][group] else np.array([])
	return result


def plot_reference_violins(
	df: pd.DataFrame,
	metric_cols: list[str],
	reference: str,
	out_path: str,
	figsize_per_metric: tuple[float, float] = (2.4, 2.0),
	format: str = 'png',
) -> None:
	"""Create violin plots: one subplot per metric, two violins (Reference vs Others, Others vs Others).
	Uses Nature journal styling: Arial, 7pt, colorblind-safe palette, 600 DPI."""
	apply_nature_style()
	n = len(metric_cols)
	ncols = min(3, n)
	nrows = (n + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_metric[0] * ncols, figsize_per_metric[1] * nrows))
	if n == 1:
		axes = np.array([axes])
	axes = axes.flatten()

	group_order = ['Reference vs Others', 'Others vs Others']
	tick_labels = _tick_labels(reference)

	for idx, metric in enumerate(metric_cols):
		ax = axes[idx]
		data = df[['group', metric]].dropna(subset=[metric])
		if data.empty:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
			continue

		vals_by_group = []
		positions = []
		labels = []
		for i, group in enumerate(group_order):
			vals = data[data['group'] == group][metric].values
			if len(vals) > 0:
				vals_by_group.append(vals)
				positions.append(i)
				labels.append(tick_labels[group])

		if vals_by_group:
			draw_violin_ax(
				ax,
				vals_by_group,
				positions,
				labels,
				NATURE_COLORS,
				_metric_label(metric),
				group_order=group_order,
				set_ylim=lambda a, m=metric: _set_ylim(a, m),
				subplot_label=chr(97 + idx),
			)
		else:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

	for j in range(idx + 1, len(axes)):
		axes[j].set_visible(False)
	fig.tight_layout()
	save_violin_figure(fig, out_path, format=format or 'png')


def plot_reference_violins_single(
	df: pd.DataFrame,
	metric_cols: list[str],
	reference: str,
	out_dir: str,
	figsize: tuple[float, float] = (3.5, 2.5),
	format: str = 'png',
) -> None:
	"""Create one violin plot per metric, saved as separate files. Nature-style."""
	apply_nature_style()
	os.makedirs(out_dir, exist_ok=True)
	group_order = ['Reference vs Others', 'Others vs Others']
	tick_labels = _tick_labels(reference)

	for metric in metric_cols:
		fig, ax = plt.subplots(figsize=figsize)
		data = df[['group', metric]].dropna(subset=[metric])
		if data.empty:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
		else:
			vals_by_group = []
			positions = []
			labels = []
			for i, group in enumerate(group_order):
				vals = data[data['group'] == group][metric].values
				if len(vals) > 0:
					vals_by_group.append(vals)
					positions.append(i)
					labels.append(tick_labels[group])

			if vals_by_group:
				draw_violin_ax(
					ax,
					vals_by_group,
					positions,
					labels,
					NATURE_COLORS,
					_metric_label(metric),
					group_order=group_order,
					set_ylim=lambda a, m=metric: _set_ylim(a, m),
				)
			else:
				ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
		fig.tight_layout()
		safe_name = re.sub(r'[^\w\-]', '_', metric)
		out_path = os.path.join(out_dir, f'reference_violin_{safe_name}.{format}')
		save_violin_figure(fig, out_path, format=format)
		print(f'Saved {out_path}')


def plot_point_diff_violins(
	point_diffs: dict[str, dict[str, np.ndarray]],
	metric_cols: list[str],
	reference: str,
	out_path: str,
	figsize_per_metric: tuple[float, float] = (2.4, 2.0),
	format: str = 'png',
) -> None:
	"""Violin plots of case-wise pairwise metric differences. Nature-style."""
	apply_nature_style()
	n = len(metric_cols)
	ncols = min(3, n)
	nrows = (n + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_metric[0] * ncols, figsize_per_metric[1] * nrows))
	if n == 1:
		axes = np.array([axes])
	axes = axes.flatten()

	group_order = ['Reference vs Others', 'Others vs Others']
	tick_labels = _tick_labels(reference)

	for idx, metric in enumerate(metric_cols):
		ax = axes[idx]
		vals_by_group = []
		positions = []
		labels = []
		for i, group in enumerate(group_order):
			vals = point_diffs.get(metric, {}).get(group, np.array([]))
			vals = np.asarray(vals)
			if len(vals) > 0:
				vals_by_group.append(vals)
				positions.append(i)
				labels.append(tick_labels[group])

		if vals_by_group:
			draw_violin_ax(
				ax,
				vals_by_group,
				positions,
				labels,
				NATURE_COLORS,
				f'{_metric_label(metric)} (|pairwise diff|)',
				group_order=group_order,
				set_ylim=lambda a, m=metric: _set_ylim(a, m),
				subplot_label=chr(97 + idx),
			)
		else:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

	for j in range(idx + 1, len(axes)):
		axes[j].set_visible(False)
	fig.tight_layout()
	save_violin_figure(fig, out_path, format=format or 'png')


def plot_point_diff_violins_single(
	point_diffs: dict[str, dict[str, np.ndarray]],
	metric_cols: list[str],
	reference: str,
	out_dir: str,
	figsize: tuple[float, float] = (3.5, 2.5),
	format: str = 'png',
) -> None:
	"""One violin plot per metric for point differences. Nature-style."""
	apply_nature_style()
	os.makedirs(out_dir, exist_ok=True)
	group_order = ['Reference vs Others', 'Others vs Others']
	tick_labels = _tick_labels(reference)

	for metric in metric_cols:
		fig, ax = plt.subplots(figsize=figsize)
		vals_by_group = []
		positions = []
		labels = []
		for i, group in enumerate(group_order):
			vals = point_diffs.get(metric, {}).get(group, np.array([]))
			vals = np.asarray(vals)
			if len(vals) > 0:
				vals_by_group.append(vals)
				positions.append(i)
				labels.append(tick_labels[group])

		if vals_by_group:
			draw_violin_ax(
				ax,
				vals_by_group,
				positions,
				labels,
				NATURE_COLORS,
				f'{_metric_label(metric)} (|pairwise diff|)',
				group_order=group_order,
				set_ylim=lambda a, m=metric: _set_ylim(a, m),
			)
		else:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
		fig.tight_layout()
		safe_name = re.sub(r'[^\w\-]', '_', metric)
		out_path = os.path.join(out_dir, f'point_diff_violin_{safe_name}.{format}')
		save_violin_figure(fig, out_path, format=format)
		print(f'Saved {out_path}')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description='Plot pred-vs-pred metrics: reference vs others vs others amongst themselves'
	)
	p.add_argument(
		'input_dir',
		help='Directory containing *_vs_*.csv files from compute_metrics_meshes_pred_vs_pred',
	)
	p.add_argument(
		'--reference',
		required=True,
		help='Name of the prediction to use as reference (e.g. obs1, obs2)',
	)
	p.add_argument(
		'--out-dir',
		default=None,
		help='Output directory for figures (default: input_dir)',
	)
	p.add_argument(
		'--out',
		default=None,
		help='Output path for combined figure (default: reference_violin_metrics.png in out-dir)',
	)
	p.add_argument(
		'--metrics',
		default=None,
		metavar='M1,M2,...',
		help='Comma-separated metrics to plot. Default: %s' % ','.join(DEFAULT_METRICS[:5]) + ',...',
	)
	p.add_argument(
		'--separate',
		action='store_true',
		help='Save one figure per metric instead of a combined grid',
	)
	p.add_argument(
		'--point-comparison',
		action='store_true',
		help='Plot pairwise metric differences per case (e.g. data0_obs0 - data0_obs1) instead of raw values',
	)
	p.add_argument(
		'--format',
		choices=['png', 'pdf'],
		default='png',
		help='Output format: png (600 DPI) or pdf (vector, preferred by Nature). Default: png',
	)
	return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	indir = args.input_dir
	out_dir = args.out_dir or indir
	reference = args.reference

	if not os.path.isdir(indir):
		print(f'Error: Not a directory: {indir}', file=sys.stderr)
		return 1

	df = _load_all_comparisons(indir)
	if df is None or df.empty:
		print(f'Error: No *_vs_*.csv files found in {indir}', file=sys.stderr)
		return 2

	available = _get_available_references(indir)
	if reference not in available:
		print(f'Error: Reference "{reference}" not found. Available: {sorted(available)}', file=sys.stderr)
		return 3

	df['group'] = df['comparison'].apply(lambda c: _assign_group(c, reference))

	metrics_to_plot = [m.strip() for m in args.metrics.split(',')] if args.metrics else DEFAULT_METRICS
	metric_cols = _get_numeric_metric_columns(df, metrics_to_plot)
	if not metric_cols:
		print('Error: No matching numeric metric columns found.', file=sys.stderr)
		return 4

	for m in metric_cols:
		if m in RELATIVE_ERROR_METRICS and m in df.columns:
			df[m] = np.abs(df[m])

	os.makedirs(out_dir, exist_ok=True)

	fmt = args.format
	if args.point_comparison:
		point_diffs = _compute_point_differences(df, metric_cols, reference)
		if args.separate:
			plot_point_diff_violins_single(point_diffs, metric_cols, reference, out_dir, format=fmt)
		else:
			out_path = args.out or os.path.join(out_dir, f'point_diff_violin_metrics_{reference}.png')
			plot_point_diff_violins(point_diffs, metric_cols, reference, out_path, format=fmt)
			base, _ = os.path.splitext(out_path)
			print(f'Saved {base}.{fmt}')
		latex_path = os.path.join(out_dir, f'latex_table_point_diff_{reference}.txt')
		write_point_diff_latex_table(
			point_diffs, metric_cols, reference, latex_path,
			metric_display_names=METRIC_DISPLAY_NAMES,
		)
		print(f'Wrote {latex_path}')
	else:
		if args.separate:
			plot_reference_violins_single(df, metric_cols, reference, out_dir, format=fmt)
		else:
			out_path = args.out or os.path.join(out_dir, f'reference_violin_metrics_{reference}.png')
			plot_reference_violins(df, metric_cols, reference, out_path, format=fmt)
			base, _ = os.path.splitext(out_path)
			print(f'Saved {base}.{fmt}')
		latex_path = os.path.join(out_dir, f'latex_table_{reference}.txt')
		write_pred_vs_pred_reference_latex_table(
			df, metric_cols, reference, latex_path,
			metric_display_names=METRIC_DISPLAY_NAMES,
		)
		print(f'Wrote {latex_path}')

	return 0


if __name__ == '__main__':
	raise SystemExit(main())
