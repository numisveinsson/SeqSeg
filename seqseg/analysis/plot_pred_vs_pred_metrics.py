"""Plot metrics from compute_metrics_meshes_pred_vs_pred output as violin plots.

Reads the per-comparison CSV files (e.g. obs1_vs_obs2.csv) and creates violin
plots showing the distribution of each metric across comparisons.

Usage:
	python -m seqseg.analysis.plot_pred_vs_pred_metrics /path/to/output_dir
	python -m seqseg.analysis.plot_pred_vs_pred_metrics /path/to/output_dir --out-dir /path/to/figures
	python -m seqseg.analysis.plot_pred_vs_pred_metrics /path/to/output_dir --metrics dice,hausdorff_sym,volume_error_rel
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

try:
	from seqseg.analysis.plotting.latex_table import write_pred_vs_pred_all_comparisons_latex_table
	from seqseg.analysis.plotting.violin_plot_functions import (
		apply_nature_style,
		draw_violin_ax,
		get_nature_colors,
		save_violin_figure,
	)
except ImportError:
	# Allow running as script: python seqseg/analysis/plot_pred_vs_pred_metrics.py
	_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	if _root not in sys.path:
		sys.path.insert(0, _root)
	from seqseg.analysis.plotting.latex_table import write_pred_vs_pred_all_comparisons_latex_table
	from seqseg.analysis.plotting.violin_plot_functions import (
		apply_nature_style,
		draw_violin_ax,
		get_nature_colors,
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

# Display names for LaTeX table row labels
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

# Abbreviations for LaTeX table column headers (short form for compact tables)
METRIC_ABBREVIATIONS = {
	'hausdorff_sym': 'HD',
	'hd95_sym': 'HD95',
	'assd': 'ASSD',
	'dice': 'Dice',
	'volume_error_rel': 'VolErr',
	'surface_area_error_rel': 'SAErr',
	'surface_dice_t1': 'SD$_{\\tau 1}$',
	'surface_dice_t2': 'SD$_{\\tau 2}$',
	'centerline_overlap': 'CL',
	'mean_normal_angular_error_gt_to_pred': 'AvgNorm',
	'max_normal_angular_error_gt_to_pred': 'MaxNorm',
	'std_normal_angular_error_gt_to_pred': 'StdNorm',
}

# Dice-like metrics: [0, 1] range, ylim upper = 1
DICE_METRICS = {'dice', 'surface_dice_t1', 'surface_dice_t2', 'centerline_overlap'}
# Distance/error metrics: lower bound 0
DISTANCE_METRICS = {
	'hausdorff_sym', 'hd95_sym', 'assd', 'volume_error_rel', 'surface_area_error_rel',
	'mean_normal_angular_error_gt_to_pred', 'max_normal_angular_error_gt_to_pred', 'std_normal_angular_error_gt_to_pred',
}
# Relative error metrics: use absolute value before plotting and calculations
RELATIVE_ERROR_METRICS = {'volume_error_rel', 'surface_area_error_rel'}


def _metric_label(metric: str) -> str:
	return METRIC_DISPLAY_NAMES.get(metric, metric)


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
		# e.g. obs1_vs_obs2.csv -> obs1_vs_obs2
		name = basename[:-4] if basename.endswith('.csv') else basename
		if '_vs_' in name:
			results.append((name, p))
	return sorted(results, key=lambda x: x[0])


def _load_all_comparisons(indir: str) -> pd.DataFrame | None:
	"""Load all per-comparison CSVs into one DataFrame with 'comparison' column."""
	pairs = _find_comparison_csvs(indir)
	if not pairs:
		return None

	dfs = []
	for comp_name, path in pairs:
		try:
			df = pd.read_csv(path)
			# Exclude MEAN/STD rows if present
			df = df[df['case'].astype(str).str.upper() != 'MEAN']
			df = df[df['case'].astype(str).str.upper() != 'STD']
			df['comparison'] = comp_name
			dfs.append(df)
		except Exception as e:
			print(f'Warning: Could not read {path}: {e}', file=sys.stderr)

	if not dfs:
		return None
	df = pd.concat(dfs, ignore_index=True)
	# Use absolute values for relative error metrics before plotting/calculations
	for m in RELATIVE_ERROR_METRICS:
		if m in df.columns:
			df[m] = np.abs(df[m])
	return df


def _get_numeric_metric_columns(df: pd.DataFrame, metrics: list[str] | None) -> list[str]:
	"""Return list of numeric metric columns to plot."""
	exclude = {'case', 'comparison'}
	# Exclude radius-bucket columns (many of them)
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


def _format_comparison_label(name: str) -> str:
	"""Format comparison name for display (e.g. obs1_vs_obs2 -> obs1 vs obs2)."""
	return name.replace('_vs_', ' vs ')


def plot_violins(
	df: pd.DataFrame,
	metric_cols: list[str],
	out_path: str,
	figsize_per_metric: tuple[float, float] = (2.4, 2.0),
	rotation: int = 15,
	fmt: str = 'png',
) -> None:
	"""Create violin plots: one subplot per metric, comparisons on x-axis. Nature-style."""
	if draw_violin_ax is None:
		_plot_violins_fallback(df, metric_cols, out_path, rotation)
		return

	apply_nature_style()
	n = len(metric_cols)
	ncols = min(3, n)
	nrows = (n + ncols - 1) // ncols
	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_metric[0] * ncols, figsize_per_metric[1] * nrows))
	if n == 1:
		axes = np.array([axes])
	axes = axes.flatten()

	comparisons = sorted(df['comparison'].unique())
	colors = get_nature_colors(len(comparisons))

	for idx, metric in enumerate(metric_cols):
		ax = axes[idx]
		data = df[['comparison', metric]].dropna(subset=[metric])
		if data.empty:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
			continue

		vals_by_comp = [(c, data[data['comparison'] == c][metric].values) for c in comparisons]
		vals_by_comp = [(c, v) for c, v in vals_by_comp if len(v) > 0]
		if not vals_by_comp:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
			continue

		comps_used, values_list = zip(*vals_by_comp)
		positions = list(range(len(comps_used)))
		labels = [_format_comparison_label(c) for c in comps_used]
		draw_violin_ax(
			ax,
			list(values_list),
			positions,
			labels,
			colors,
			_metric_label(metric),
			group_order=None,
			set_ylim=lambda a, m=metric: _set_ylim(a, m),
			add_wilcoxon=len(comps_used) == 2,
			subplot_label=chr(97 + idx),
			xtick_rotation=rotation,
		)

	for j in range(idx + 1, len(axes)):
		axes[j].set_visible(False)
	fig.tight_layout()
	save_violin_figure(fig, out_path, format=fmt)


def _plot_violins_fallback(
	df: pd.DataFrame,
	metric_cols: list[str],
	out_path: str,
	rotation: int,
) -> None:
	"""Fallback when violin_plot_functions not available."""
	import matplotlib.pyplot as plt
	n = len(metric_cols)
	ncols = min(3, n)
	nrows = (n + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
	if n == 1:
		axes = np.array([axes])
	axes = axes.flatten()
	comparisons = sorted(df['comparison'].unique())
	colors = plt.cm.Set3(np.linspace(0, 1, len(comparisons)))
	for idx, metric in enumerate(metric_cols):
		ax = axes[idx]
		data = df[['comparison', metric]].dropna(subset=[metric])
		if data.empty:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
			continue
		vals_by_comp = [(c, data[data['comparison'] == c][metric].values) for c in comparisons]
		vals_by_comp = [(c, v) for c, v in vals_by_comp if len(v) > 0]
		if not vals_by_comp:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
			continue
		comps_used, values_list = zip(*vals_by_comp)
		parts = ax.violinplot(values_list, positions=range(len(comps_used)), showmeans=True, showmedians=True)
		for i, pc in enumerate(parts['bodies']):
			pc.set_facecolor(colors[i % len(colors)])
			pc.set_alpha(0.7)
		ax.set_xticks(range(len(comps_used)))
		ax.set_xticklabels([_format_comparison_label(c) for c in comps_used], rotation=rotation, ha='right')
		ax.set_ylabel(metric)
		ax.grid(axis='y', alpha=0.3)
	for j in range(idx + 1, len(axes)):
		axes[j].set_visible(False)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150, bbox_inches='tight')
	plt.close(fig)


def plot_violins_single_figure(
	df: pd.DataFrame,
	metric_cols: list[str],
	out_dir: str,
	figsize: tuple[float, float] = (3.5, 2.5),
	rotation: int = 15,
	fmt: str = 'png',
) -> None:
	"""Create one violin plot per metric, saved as separate files. Nature-style."""
	os.makedirs(out_dir, exist_ok=True)

	if draw_violin_ax is None:
		_plot_violins_single_fallback(df, metric_cols, out_dir, rotation)
		return

	import matplotlib.pyplot as plt
	comparisons = sorted(df['comparison'].unique())
	colors = get_nature_colors(len(comparisons))
	apply_nature_style()
	for metric in metric_cols:
		fig, ax = plt.subplots(figsize=figsize)
		data = df[['comparison', metric]].dropna(subset=[metric])
		if data.empty:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
		else:
			vals_by_comp = [(c, data[data['comparison'] == c][metric].values) for c in comparisons]
			vals_by_comp = [(c, v) for c, v in vals_by_comp if len(v) > 0]
			if not vals_by_comp:
				ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
			else:
				comps_used, values_list = zip(*vals_by_comp)
				positions = list(range(len(comps_used)))
				labels = [_format_comparison_label(c) for c in comps_used]
				draw_violin_ax(
					ax,
					list(values_list),
					positions,
					labels,
					colors,
					_metric_label(metric),
					group_order=None,
					set_ylim=lambda a, m=metric: _set_ylim(a, m),
					add_wilcoxon=len(comps_used) == 2,
					xtick_rotation=rotation,
				)
		fig.tight_layout()
		safe_name = re.sub(r'[^\w\-]', '_', metric)
		out_path = os.path.join(out_dir, f'violin_{safe_name}.{fmt}')
		save_violin_figure(fig, out_path, format=fmt)
		print(f'Saved {out_path}')


def _plot_violins_single_fallback(
	df: pd.DataFrame,
	metric_cols: list[str],
	out_dir: str,
	rotation: int,
) -> None:
	"""Fallback when violin_plot_functions not available."""
	import matplotlib.pyplot as plt
	comparisons = sorted(df['comparison'].unique())
	colors = plt.cm.Set3(np.linspace(0, 1, len(comparisons)))
	for metric in metric_cols:
		fig, ax = plt.subplots(figsize=(10, 6))
		data = df[['comparison', metric]].dropna(subset=[metric])
		if data.empty:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
		else:
			vals_by_comp = [(c, data[data['comparison'] == c][metric].values) for c in comparisons]
			vals_by_comp = [(c, v) for c, v in vals_by_comp if len(v) > 0]
			if vals_by_comp:
				comps_used, values_list = zip(*vals_by_comp)
				parts = ax.violinplot(values_list, positions=range(len(comps_used)), showmeans=True, showmedians=True)
				for i, pc in enumerate(parts['bodies']):
					pc.set_facecolor(colors[i % len(colors)])
					pc.set_alpha(0.7)
				ax.set_xticks(range(len(comps_used)))
				ax.set_xticklabels([_format_comparison_label(c) for c in comps_used], rotation=rotation, ha='right')
				ax.set_ylabel(metric)
				ax.grid(axis='y', alpha=0.3)
		fig.tight_layout()
		safe_name = re.sub(r'[^\w\-]', '_', metric)
		fig.savefig(os.path.join(out_dir, f'violin_{safe_name}.png'), dpi=150, bbox_inches='tight')
		plt.close(fig)
		print(f'Saved {out_dir}/violin_{safe_name}.png')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description='Plot pred-vs-pred metrics as violin plots'
	)
	p.add_argument(
		'input_dir',
		help='Directory containing *_vs_*.csv files from compute_metrics_meshes_pred_vs_pred',
	)
	p.add_argument(
		'--out-dir',
		default=None,
		help='Output directory for figures (default: input_dir)',
	)
	p.add_argument(
		'--out',
		default=None,
		help='Output path for combined figure (default: violin_metrics.png in out-dir)',
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
		'--rotation',
		type=int,
		default=15,
		help='Rotation for x-axis labels (default: 15)',
	)
	p.add_argument(
		'--format',
		choices=['png', 'pdf'],
		default='png',
		help='Output format: png (600 DPI) or pdf. Default: png',
	)
	return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	indir = args.input_dir
	out_dir = args.out_dir or indir
	metrics_arg = args.metrics
	separate = args.separate

	if not os.path.isdir(indir):
		print(f'Error: Not a directory: {indir}', file=sys.stderr)
		return 1

	df = _load_all_comparisons(indir)
	if df is None or df.empty:
		print(f'Error: No *_vs_*.csv files found in {indir}', file=sys.stderr)
		return 2

	metrics_to_plot = [m.strip() for m in metrics_arg.split(',')] if metrics_arg else DEFAULT_METRICS
	metric_cols = _get_numeric_metric_columns(df, metrics_to_plot)
	if not metric_cols:
		print('Error: No matching numeric metric columns found.', file=sys.stderr)
		return 3

	os.makedirs(out_dir, exist_ok=True)

	fmt = args.format
	if separate:
		plot_violins_single_figure(df, metric_cols, out_dir, rotation=args.rotation, fmt=fmt)
	else:
		out_path = args.out or os.path.join(out_dir, f'violin_metrics.{fmt}')
		plot_violins(df, metric_cols, out_path, rotation=args.rotation, fmt=fmt)
		print(f'Saved {out_path}')

	if write_pred_vs_pred_all_comparisons_latex_table is not None:
		latex_path = os.path.join(out_dir, 'latex_table_all_comparisons.txt')
		write_pred_vs_pred_all_comparisons_latex_table(
			df, metric_cols, latex_path,
			metric_display_names=METRIC_DISPLAY_NAMES,
			metric_abbreviations=METRIC_ABBREVIATIONS,
		)
		print(f'Wrote {latex_path}')

	return 0


if __name__ == '__main__':
	raise SystemExit(main())
