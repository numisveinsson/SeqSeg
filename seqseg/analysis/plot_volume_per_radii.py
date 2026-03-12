"""Plot volume-per-radii metrics from summary.csv (from compute_metrics_meshes_comparison.py).

Creates grouped bar plots: one bar group per method (pred_folder), with each bucket
as a differently colored bar within the group.

Usage:
	python -m seqseg.analysis.plot_volume_per_radii summary.csv
	python -m seqseg.analysis.plot_volume_per_radii summary.csv --metric rel --out volume_radii.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np
import pandas as pd

try:
	from seqseg.analysis.volume_per_radii import format_bucket_label_for_display
except ImportError:
	try:
		from analysis.volume_per_radii import format_bucket_label_for_display
	except ImportError:

		def format_bucket_label_for_display(label: str, latex: bool = False) -> str:
			parts = label.split('_')
			if len(parts) >= 2:
				if parts[1] == 'inf':
					return f'[{float(parts[0]) * 10:.1f}, ∞)'
				lo, hi = float(parts[0]) * 10, float(parts[1]) * 10
				return f'[{lo:.1f}, {hi:.1f})'
			return label


def _format_method_display_name(folder: str) -> str:
	"""Map pred_folder to professional display name."""
	folder = str(folder).strip()
	if folder == 'global_pred':
		return 'EGNN (ours)'
	if folder == 'mc':
		return 'Marching Cubes'
	# Taubin variants: taubin_on_mc_04_50 -> Taubin -0.4/50, taubin_on_mc_025_400 -> Taubin -0.25/400
	m = re.match(r'.*taubin.*_(\d+)_(\d+)$', folder, re.IGNORECASE)
	if m:
		a, b = m.group(1), m.group(2)
		# Convert 04 -> 0.4, 025 -> 0.25, 02 -> 0.2
		if a.startswith('0') and len(a) >= 2:
			val = int(a) / (10 ** (len(a) - 1))
		else:
			val = int(a)
		return f'Taubin {val}/{b}'
	if 'taubin' in folder.lower():
		return 'Taubin'
	return folder


def _find_volume_radii_columns(df: pd.DataFrame, metric: str) -> list[str]:
	"""Find volume-per-radii columns in the summary DataFrame.

	metric: 'rel' for volume_error_radii_*_rel, 'gt' for volume_gt_radii_*, 'pred' for volume_pred_radii_*
	Returns columns ending with _mean (we use mean for the bar values).
	"""
	if metric == 'rel':
		prefix = 'volume_error_radii_'
		suffix = '_rel_mean'
	else:
		prefix = f'volume_{metric}_radii_'
		suffix = '_mean'

	cols = [c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)]
	return sorted(cols)


def _extract_bucket_label(col: str, metric: str) -> str:
	"""Extract bucket label from column name, e.g. '0.0_0.1' or '1.6_inf'."""
	if metric == 'rel':
		# volume_error_radii_0.0_0.1_rel_mean -> 0.0_0.1
		prefix = 'volume_error_radii_'
		suffix = '_rel_mean'
	else:
		prefix = f'volume_{metric}_radii_'
		suffix = '_mean'
	return col[len(prefix) : -len(suffix)]


def plot_volume_per_radii(
	summary_csv: str,
	metric: str = 'rel',
	out_path: str | None = None,
	figsize: tuple[float, float] = (10, 6),
	ylabel: str | None = None,
) -> None:
	"""Plot volume-per-radii as grouped bar chart.

	Parameters
	----------
	summary_csv : str
		Path to summary.csv from compute_metrics_meshes_comparison.py
	metric : str
		'rel' = relative volume error (pred-gt)/gt, 'gt' = GT volume, 'pred' = pred volume
	out_path : str, optional
		Path to save the figure
	figsize : tuple
		Figure size (width, height)
	ylabel : str, optional
		Y-axis label (default: auto based on metric)
	"""
	df = pd.read_csv(summary_csv)
	if 'pred_folder' not in df.columns:
		raise ValueError(
			f"summary.csv must have 'pred_folder' column. Found: {list(df.columns)[:5]}..."
		)

	cols = _find_volume_radii_columns(df, metric)
	if not cols:
		raise ValueError(
			f"No volume-per-radii columns found for metric '{metric}'. "
			f"Ensure summary.csv was generated with --centerline-dir and volume_radii metrics."
		)

	# Order: MC, Taubin, EGNN
	df = df.sort_values('pred_folder', key=lambda s: s.str.strip().str.lower().map(
		lambda f: (0 if f == 'mc' else 1 if 'taubin' in f else 2 if f == 'global_pred' else 3)
	))
	methods_raw = df['pred_folder'].astype(str).tolist()
	methods = [_format_method_display_name(m) for m in methods_raw]
	bucket_labels = [_extract_bucket_label(c, metric) for c in cols]
	display_labels = [format_bucket_label_for_display(b) + ' mm radius' for b in bucket_labels]

	# Build data matrix: rows = methods, cols = buckets
	data = df[cols].values
	# Optional: std for error bars
	std_cols = [c.replace('_mean', '_std') for c in cols]
	has_std = all(c in df.columns for c in std_cols)
	std_data = df[std_cols].values if has_std else None

	n_methods = len(methods)
	n_buckets = len(bucket_labels)

	def _ylim_from_data(plot_data, plot_std, bottom_fixed=None, top_fixed=None, padding_frac=0.1):
		"""Compute ylim from data range with optional padding."""
		lo = float(np.nanmin(plot_data))
		hi = float(np.nanmax(plot_data))
		if plot_std is not None:
			lo = min(lo, float(np.nanmin(plot_data - plot_std)))
			hi = max(hi, float(np.nanmax(plot_data + plot_std)))
		span = hi - lo if hi != lo else 1.0
		padding = span * padding_frac
		ylim_bottom = bottom_fixed if bottom_fixed is not None else lo - padding
		ylim_top = top_fixed if top_fixed is not None else hi + padding
		return ylim_bottom, ylim_top

	# Bar layout: grouped bars
	bar_width = 0.8 / n_buckets
	x = np.arange(n_methods)
	offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, n_buckets)

	def _draw_bars(ax, plot_data, plot_std, ylim_bottom, ylim_top, ax_ylabel):
		colors = plt.cm.tab10(np.linspace(0, 1, n_buckets))
		for i, (col, bl, dl) in enumerate(zip(cols, bucket_labels, display_labels)):
			vals = plot_data[:, i]
			err = plot_std[:, i] if has_std and plot_std is not None else None
			ax.bar(
				x + offsets[i],
				vals,
				bar_width,
				label=dl,
				color=colors[i],
				yerr=err,
				capsize=2,
				error_kw={'elinewidth': 1},
			)
		ax.set_xlabel('Method')
		ax.set_ylabel(ax_ylabel)
		ax.set_xticks(x)
		ax.set_xticklabels(methods, rotation=45, ha='right')
		ax.legend(loc='best', fontsize=9)
		ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
		if ylim_bottom is not None and ylim_top is not None:
			ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

	# Plot 1: signed values
	fig, ax = plt.subplots(figsize=figsize)
	if ylabel is not None:
		ax_ylabel = ylabel
	elif metric == 'rel':
		ax_ylabel = 'Relative volume error (pred − gt) / gt'
	elif metric == 'gt':
		ax_ylabel = 'Volume GT (mm³)'
	else:
		ax_ylabel = 'Volume pred (mm³)'
	ylim_bottom, ylim_top = _ylim_from_data(data, std_data)
	_draw_bars(ax, data, std_data, ylim_bottom=ylim_bottom, ylim_top=ylim_top, ax_ylabel=ax_ylabel)
	fig.tight_layout()
	if out_path:
		fig.savefig(out_path, dpi=150, bbox_inches='tight')
		print(f'Saved figure to {out_path}')
	else:
		plt.show()

	# Plot 2 (only for rel): absolute values, y bottom 0
	if metric == 'rel':
		data_abs = np.abs(data)
		ylim_bottom_abs, ylim_top_abs = _ylim_from_data(data_abs, std_data, bottom_fixed=0)
		fig2, ax2 = plt.subplots(figsize=figsize)
		_draw_bars(ax2, data_abs, std_data, ylim_bottom=ylim_bottom_abs, ylim_top=ylim_top_abs, ax_ylabel='|Relative volume error|')
		fig2.tight_layout()
		if out_path:
			base, ext = os.path.splitext(out_path)
			out_path_abs = f'{base}_abs{ext}'
			fig2.savefig(out_path_abs, dpi=150, bbox_inches='tight')
			print(f'Saved figure to {out_path_abs}')
		else:
			plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description='Plot volume-per-radii from summary.csv as grouped bar chart'
	)
	p.add_argument(
		'summary_csv',
		help='Path to summary.csv from compute_metrics_meshes_comparison.py (--predictions-root mode)',
	)
	p.add_argument(
		'--metric',
		choices=['rel', 'gt', 'pred'],
		default='rel',
		help='Metric to plot: rel=relative error, gt=GT volume, pred=pred volume (default: rel)',
	)
	p.add_argument(
		'--out', '-o',
		default=None,
		help='Output path for figure (default: display only)',
	)
	p.add_argument(
		'--figsize',
		default='10,6',
		help='Figure size as width,height (default: 10,6)',
	)
	return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	if not os.path.isfile(args.summary_csv):
		print(f'Error: File not found: {args.summary_csv}', file=sys.stderr)
		return 1
	try:
		figsize = tuple(float(x) for x in args.figsize.split(','))
	except ValueError:
		print(f'Error: --figsize must be width,height (e.g. 10,6)', file=sys.stderr)
		return 1

	plot_volume_per_radii(
		args.summary_csv,
		metric=args.metric,
		out_path=args.out,
		figsize=figsize,
	)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
