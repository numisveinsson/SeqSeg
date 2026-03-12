"""Analyze metrics CSVs by case categories.

Reads a folder of .csv files (from compute_metrics_meshes.py, excluding summary.csv),
a JSON file mapping case names to categories, and computes mean and std per category
for each metric in each CSV.

Usage:
	python -m seqseg.analysis.analyze_by_names /path/to/csv_folder /path/to/categories.json [--out-dir output_dir]
	python -m seqseg.analysis.analyze_by_names /path/to/csv_folder /path/to/categories.json --metrics hd95_gt_to_pred,assd,dice
	python -m seqseg.analysis.analyze_by_names /path/to/csv_folder /path/to/categories.json --predictions-names predictions_names.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
	from .volume_per_radii import (
		get_volume_per_radii_fieldnames,
		DEFAULT_RADIUS_BUCKETS,
		format_bucket_label_for_display,
	)
	_VOLUME_RADII_FIELDNAMES = get_volume_per_radii_fieldnames(DEFAULT_RADIUS_BUCKETS)
except ImportError:
	_VOLUME_RADII_FIELDNAMES = []
	format_bucket_label_for_display = None

# Metric columns from compute_metrics_meshes (excluding 'case')
METRIC_COLUMNS = [
	'hausdorff_sym', 'hd95_sym', 'hausdorff_gt_to_pred', 'hd95_gt_to_pred',
	'hausdorff_pred_to_gt', 'hd95_pred_to_gt', 'mean_gt_to_pred', 'mean_pred_to_gt',
	'assd', 'n_sampled_gt', 'n_sampled_pred', 'dice', 'n_voxels_gt', 'n_voxels_pred',
	'centerline_overlap', 'mean_normal_angular_error_gt_to_pred',
	'std_normal_angular_error_gt_to_pred', 'max_normal_angular_error_gt_to_pred',
	'volume_gt', 'volume_pred', 'volume_error_abs', 'volume_error_rel',
] + _VOLUME_RADII_FIELDNAMES + [
	'surface_area_gt', 'surface_area_pred', 'surface_area_error_abs', 'surface_area_error_rel',
	'surface_dice_t1', 'surface_dice_t2',
]

# Compact display names and direction (↓ lower better, ↑ higher better) for LaTeX
METRIC_DISPLAY: Dict[str, Tuple[str, str]] = {
	'hausdorff_sym': ('Haus. sym', '$\\downarrow$'),
	'hd95_sym': ('HD95 sym', '$\\downarrow$'),
	'hausdorff_gt_to_pred': ('Haus. G$\\rightarrow$P', '$\\downarrow$'),
	'hd95_gt_to_pred': ('HD95 G$\\rightarrow$P', '$\\downarrow$'),
	'hausdorff_pred_to_gt': ('Haus. P$\\rightarrow$G', '$\\downarrow$'),
	'hd95_pred_to_gt': ('HD95 P$\\rightarrow$G', '$\\downarrow$'),
	'mean_gt_to_pred': ('Mean dist. G$\\rightarrow$P', '$\\downarrow$'),
	'mean_pred_to_gt': ('Mean dist. P$\\rightarrow$G', '$\\downarrow$'),
	'assd': ('ASSD', '$\\downarrow$'),
	'n_sampled_gt': ('N samp. GT', ''),
	'n_sampled_pred': ('N samp. pred', ''),
	'dice': ('Dice', '$\\uparrow$'),
	'n_voxels_gt': ('N vox. GT', ''),
	'n_voxels_pred': ('N vox. pred', ''),
	'centerline_overlap': ('CL overlap', '$\\uparrow$'),
	'mean_normal_angular_error_gt_to_pred': ('Norm. ang. mean', '$\\downarrow$'),
	'std_normal_angular_error_gt_to_pred': ('Norm. ang. std', '$\\downarrow$'),
	'max_normal_angular_error_gt_to_pred': ('Norm. ang. max', '$\\downarrow$'),
	'volume_gt': ('Vol. GT', ''),
	'volume_pred': ('Vol. pred', ''),
	'volume_error_abs': ('Vol. err. abs', '$\\downarrow$'),
	'volume_error_rel': ('Vol. err. rel', '$\\downarrow$'),
	'surface_area_gt': ('Surf. GT', ''),
	'surface_area_pred': ('Surf. pred', ''),
	'surface_area_error_abs': ('Surf. err. abs', '$\\downarrow$'),
	'surface_area_error_rel': ('Surf. err. rel', '$\\downarrow$'),
	'surface_dice_t1': ('SurfDice $t_1$', '$\\uparrow$'),
	'surface_dice_t2': ('SurfDice $t_2$', '$\\uparrow$'),
}


def _is_volume_radii_metric(name: str) -> bool:
	"""True if metric matches volume-per-radii pattern (any bucket config)."""
	return (
		(name.startswith('volume_error_radii_') and name.endswith('_rel'))
		or name.startswith('volume_gt_radii_')
		or name.startswith('volume_pred_radii_')
		or name.startswith('n_segments_radii_')
	)


def _closest_to_zero_is_better(name: str) -> bool:
	"""True if optimal value is 0 (negative and positive both bad)."""
	return (
		name in ('volume_error_abs', 'volume_error_rel',
		         'surface_area_error_abs', 'surface_area_error_rel')
		or (name.startswith('volume_error_radii_') and name.endswith('_rel'))
	)


def metric_display_header(metric_key: str) -> str:
	"""Return LaTeX header for metric: 'Name $\\downarrow$' or 'Name $\\uparrow$'."""
	if metric_key in METRIC_DISPLAY:
		name, arrow = METRIC_DISPLAY[metric_key]
		return f'{name} {arrow}'.strip() if arrow else name
	# Volume-per-radii: compact label e.g. "Verr [0.6,1.6)↓"
	if metric_key.startswith('volume_error_radii_') and metric_key.endswith('_rel'):
		label = metric_key[len('volume_error_radii_'):-len('_rel')]
		parts = label.split('_')
		rng = (f'[{parts[0]},$\\infty$)' if len(parts) >= 2 and parts[1] == 'inf'
		       else f'[{parts[0]},{parts[1]})' if len(parts) >= 2 else label)
		return f'Verr {rng} $\\downarrow$'
	if metric_key.startswith('volume_gt_radii_'):
		label = metric_key[len('volume_gt_radii_'):]
		return f'Vol.GT {label.replace("_", "-")}'
	if metric_key.startswith('volume_pred_radii_'):
		label = metric_key[len('volume_pred_radii_'):]
		return f'Vol.pred {label.replace("_", "-")}'
	if metric_key.startswith('n_segments_radii_'):
		return metric_key.replace('_', '\\_')
	name = metric_key.replace('_', '\\_')
	return name


def load_categories(path: str) -> Dict[str, str]:
	"""Load categories JSON. Expects {category: [case1, case2, ...]}.
	Returns case -> category mapping."""
	with open(path) as f:
		data = json.load(f)
	case_to_cat: Dict[str, str] = {}
	for cat, cases in data.items():
		for c in cases:
			case_to_cat[c] = cat
	return case_to_cat


def load_csv_rows(path: str) -> List[dict]:
	"""Load CSV rows, skipping MEAN/STD summary rows if present."""
	rows = []
	with open(path) as f:
		reader = csv.DictReader(f)
		fieldnames = reader.fieldnames or []
		for r in reader:
			case = r.get('case', '')
			if case in ('MEAN', 'STD', ''):
				continue
			rows.append(r)
	return rows


def get_csv_metric_columns(path: str) -> List[str]:
	"""Return metric column names from CSV header, matching actual columns in the file.
	Uses CSV columns instead of METRIC_COLUMNS so volume-per-radii columns (e.g. 0.6_1.6
	vs 0.6_1.0) match the CSV regardless of bucket configuration."""
	with open(path) as f:
		reader = csv.DictReader(f)
		fieldnames = reader.fieldnames or []
	# Filter to metric columns: in METRIC_COLUMNS or match volume-per-radii pattern
	return [
		col for col in fieldnames
		if col != 'case' and (col in METRIC_COLUMNS or _is_volume_radii_metric(col))
	]


def numeric_value(v) -> Optional[float]:
	"""Extract numeric value; return None if not a valid number (NaN/empty/invalid are ignored)."""
	if v is None or v == '':
		return None
	s = str(v).strip().lower()
	if s in ('nan', 'inf', '-inf'):
		return None
	try:
		x = float(v)
		return x if math.isfinite(x) else None
	except (ValueError, TypeError):
		return None


def mean_std(values: List[float]) -> Tuple[float, float]:
	"""Return (mean, std). std is nan if len < 2."""
	arr = np.asarray(values, dtype=float)
	mean = float(np.mean(arr))
	std = float(np.std(arr)) if arr.size >= 2 else float('nan')
	return mean, std


def analyze_csv_by_categories(
	csv_path: str,
	case_to_cat: Dict[str, str],
	columns: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
	"""For each category, compute mean and std for each metric.
	Returns {category: {metric: {'mean': x, 'std': y}, ...}, ...}
	columns: if provided, only compute these metrics (e.g. from --metrics).
	"""
	rows = load_csv_rows(csv_path)
	by_cat: Dict[str, List[dict]] = {}
	for r in rows:
		case = r.get('case', '')
		cat = case_to_cat.get(case)
		if cat is None:
			cat = '_uncategorized'
		by_cat.setdefault(cat, []).append(r)

	cols = columns if columns is not None else get_csv_metric_columns(csv_path)
	result: Dict[str, Dict[str, Dict[str, float]]] = {}
	for cat, cat_rows in by_cat.items():
		result[cat] = {}
		for col in cols:
			vals = []
			for r in cat_rows:
				v = numeric_value(r.get(col))
				if v is not None:
					vals.append(v)
			if vals:
				m, s = mean_std(vals)
				result[cat][col] = {'mean': m, 'std': s}
			# else: omit metric for this category (all entries were NaN/invalid)
	return result


def write_category_summary(
	out_path: str,
	summary: Dict[str, Dict[str, Dict[str, float]]],
	metrics_filter: Optional[List[str]] = None,
) -> None:
	"""Write per-category mean/std to CSV."""
	cats = sorted(summary.keys())
	if not cats:
		return
	# Use metrics from summary when no filter (summary keys match actual CSV columns)
	if metrics_filter is not None:
		allowed = metrics_filter
	else:
		allowed = []
		seen = set()
		for m in METRIC_COLUMNS:
			if m not in seen and any(summary[c].get(m) for c in cats):
				seen.add(m)
				allowed.append(m)
		for c in cats:
			for m in summary[c].keys():
				if m not in seen:
					seen.add(m)
					allowed.append(m)
	metrics = [k for k in allowed if any(summary[c].get(k) for c in cats)]
	fieldnames = ['category'] + [f'{m}_mean' for m in metrics] + [f'{m}_std' for m in metrics]

	with open(out_path, 'w', newline='') as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for cat in cats:
			row = {'category': cat}
			for m in metrics:
				d = summary[cat].get(m, {'mean': float('nan'), 'std': float('nan')})
				row[f'{m}_mean'] = d['mean'] if not math.isnan(d['mean']) else ''
				row[f'{m}_std'] = d['std'] if not math.isnan(d['std']) else ''
			w.writerow(row)


def _fmt_num(x: float, decimals: int = 3) -> str:
	"""Format number for LaTeX: use scientific notation when compact (|x|<0.01 or |x|>=1e3)."""
	if math.isnan(x):
		return '---'
	if abs(x) < 0.01 or abs(x) >= 1000:
		s = f'{x:.2e}'
		# Compact exponent: 1.23e-02 -> 1.23e-2 (only when exponent is 1-9, not 10+)
		s = re.sub(r'e([+-])0([1-9])$', r'e\1\2', s)
		return s
	return f'{x:.{decimals}f}'


def format_metric_mean_std(mean: float, std: float, decimals: int = 3) -> str:
	"""Format as 'mean ± std' for LaTeX. Uses scientific notation when compact."""
	if math.isnan(mean):
		return '---'
	if math.isnan(std) or std == 0:
		return _fmt_num(mean, decimals)
	return f'{_fmt_num(mean, decimals)} $\\pm$ {_fmt_num(std, decimals)}'


def write_latex_table(
	out_path: str,
	all_summaries: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
	metrics_subset: Optional[List[str]] = None,
	predictions_names: Optional[Dict[str, str]] = None,
	means_only: bool = False,
) -> None:
	"""Write LaTeX table: category (multirow) | method | metric1 | metric2 | ...

	all_summaries: {method_name: {category: {metric: {mean, std}, ...}, ...}, ...}
	predictions_names: {stem: display_name} for LaTeX method column
	means_only: if True, show only mean values (no ± std)
	"""
	pred_names = predictions_names or {}
	# Method order: follow predictions_names.json order (by first matching key), then any others sorted
	def _order_key(stem: str) -> Tuple[int, str]:
		if stem in pred_names:
			keys_list = list(pred_names.keys())
			return (keys_list.index(stem), stem)
		matches = [k for k in pred_names if k in stem]
		if matches:
			keys_list = list(pred_names.keys())
			idx = min(keys_list.index(k) for k in matches)
			return (idx, stem)
		return (len(pred_names), stem)  # Unmatched go after all named methods
	methods_order = sorted(all_summaries.keys(), key=_order_key)
	cats_ordered = []
	seen_cats = set()
	for method in sorted(all_summaries.keys()):
		for c in sorted(all_summaries[method].keys()):
			if c not in seen_cats and c != '_uncategorized':
				seen_cats.add(c)
				cats_ordered.append(c)
	if '_uncategorized' in seen_cats:
		cats_ordered.append('_uncategorized')

	metrics = metrics_subset
	if metrics is None:
		# Collect metrics from actual summary data (matches CSV columns, e.g. volume_error_radii_0.6_1.6_rel)
		metrics = []
		seen = set()
		for method in all_summaries.values():
			for cat_data in method.values():
				for m in METRIC_COLUMNS:
					if m in seen:
						continue
					d = cat_data.get(m, {'mean': float('nan'), 'std': float('nan')})
					mean = d.get('mean', float('nan'))
					if isinstance(mean, (int, float)) and not math.isnan(mean):
						seen.add(m)
						metrics.append(m)
				for m in cat_data.keys():
					if m not in seen:
						d = cat_data.get(m, {'mean': float('nan'), 'std': float('nan')})
						mean = d.get('mean', float('nan'))
						if isinstance(mean, (int, float)) and not math.isnan(mean):
							seen.add(m)
							metrics.append(m)
		# Preserve METRIC_COLUMNS order for known metrics, append extras (e.g. volume radii with custom buckets)
		metrics = [m for m in METRIC_COLUMNS if m in metrics] + [m for m in metrics if m not in METRIC_COLUMNS]
		if not metrics:
			metrics = list(METRIC_COLUMNS)

	# Bold rules: volume error = closest to 0; dice/overlap = highest; others = lowest
	HIGHER_IS_BETTER = {'dice', 'centerline_overlap', 'surface_dice_t1', 'surface_dice_t2'}

	def _display_name(stem: str) -> str:
		# Exact match first
		if stem in pred_names:
			return pred_names[stem]
		# If stem contains a key from predictions_names, use that display name
		# Prefer longest match (e.g. "per_global_pred_taubin" over "per_global_pred")
		matches = [(k, pred_names[k]) for k in pred_names if k in stem]
		if matches:
			return max(matches, key=lambda x: len(x[0]))[1]
		return stem

	def _higher_is_better(metric: str) -> bool:
		return 'dice' in metric.lower() or metric in HIGHER_IS_BETTER

	rows: List[Tuple[str, str, str, Dict[str, str], Dict[str, float]]] = []
	for cat in cats_ordered:
		methods_with_cat = [m for m in methods_order if cat in all_summaries[m]]
		for method in methods_with_cat:
			summary = all_summaries[method][cat]
			vals = {}
			raw_means = {}
			for m in metrics:
				d = summary.get(m, {'mean': float('nan'), 'std': float('nan')})
				mean_val = float(d.get('mean', float('nan')))
				std_val = float(d.get('std', float('nan')))
				vals[m] = _fmt_num(mean_val) if means_only else format_metric_mean_std(mean_val, std_val)
				raw_means[m] = mean_val
			rows.append((cat, method, _display_name(method), vals, raw_means))

	# For each (category, metric), find which method(s) have the best value
	best_per_cat_metric: Dict[Tuple[str, str], set] = {}
	for cat in cats_ordered:
		for metric in metrics:
			pairs = []
			for c, meth, _, _, raw_means in rows:
				if c != cat:
					continue
				mean_val = raw_means.get(metric, float('nan'))
				if isinstance(mean_val, (int, float)) and not math.isnan(mean_val):
					pairs.append((meth, mean_val))
			if not pairs:
				continue
			if _closest_to_zero_is_better(metric):
				best_abs = min(abs(p[1]) for p in pairs)
				best_methods = {p[0] for p in pairs if abs(p[1]) == best_abs}
			elif _higher_is_better(metric):
				best_val = max(p[1] for p in pairs)
				best_methods = {p[0] for p in pairs if p[1] == best_val}
			else:
				best_val = min(p[1] for p in pairs)
				best_methods = {p[0] for p in pairs if p[1] == best_val}
			best_per_cat_metric[(cat, metric)] = best_methods

	if not rows:
		return

	# LaTeX output (requires \\usepackage{multirow} in preamble)
	lines = []
	lines.append('% Requires \\usepackage{multirow} in your LaTeX preamble')
	lines.append('')
	lines.append('\\begin{table}[htbp]')
	lines.append('\\centering')
	lines.append('\\caption{Mean per category and method.}' if means_only else '\\caption{Mean $\\pm$ std per category and method.}')
	lines.append(f'\\begin{{tabular}}{{l|l|{"c" * len(metrics)}}}')
	lines.append('\\hline')
	header = 'Category & Method & ' + ' & '.join(metric_display_header(m) for m in metrics) + ' \\\\'
	lines.append(header)
	lines.append('\\hline')

	def _cell_with_bold(cat: str, meth: str, metric: str, val_str: str) -> str:
		best = best_per_cat_metric.get((cat, metric), set())
		if meth in best and val_str != '---':
			return f'\\textbf{{{val_str}}}'
		return val_str

	# Group rows by category for multirow
	prev_cat = None
	cat_row_count = 0
	cat_start_idx = 0
	for i, (cat, meth_stem, meth_display, vals, _) in enumerate(rows):
		if cat != prev_cat:
			if prev_cat is not None:
				# Write previous category's rows with multirow on first
				for j in range(cat_start_idx, i):
					c, mstem, mdisp, v, _ = rows[j]
					metric_str = ' & '.join(_cell_with_bold(c, mstem, m, v[m]) for m in metrics)
					if j == cat_start_idx:
						multirow = f'\\multirow{{{cat_row_count}}}{{*}}{{{c.replace("_", " ")}}}'
						mdisp_tex = mdisp.replace('_', '\\_')
						lines.append(f'{multirow} & {mdisp_tex} & {metric_str} \\\\')
					else:
						mdisp_tex = mdisp.replace('_', '\\_')
						lines.append(f' & {mdisp_tex} & {metric_str} \\\\')
				lines.append('\\hline')
			prev_cat = cat
			cat_start_idx = i
			cat_row_count = 0
		cat_row_count += 1

	# Last category
	if prev_cat is not None:
		for j in range(cat_start_idx, len(rows)):
			c, mstem, mdisp, v, _ = rows[j]
			metric_str = ' & '.join(_cell_with_bold(c, mstem, m, v[m]) for m in metrics)
			if j == cat_start_idx:
				multirow = f'\\multirow{{{cat_row_count}}}{{*}}{{{c.replace("_", " ")}}}'
				mdisp_tex = mdisp.replace('_', '\\_')
				lines.append(f'{multirow} & {mdisp_tex} & {metric_str} \\\\')
			else:
				mdisp_tex = mdisp.replace('_', '\\_')
				lines.append(f' & {mdisp_tex} & {metric_str} \\\\')
		lines.append('\\hline')

	lines.append('\\end{tabular}')
	lines.append('\\end{table}')

	with open(out_path, 'w') as f:
		f.write('\n'.join(lines))


def main(argv: Optional[List[str]] = None) -> int:
	p = argparse.ArgumentParser(
		description='Compute mean and std per category for metrics in CSV files'
	)
	p.add_argument('csv_dir', help='Folder containing .csv files (summary.csv is ignored)')
	p.add_argument('categories_json', help='JSON file mapping case names to categories')
	p.add_argument('--out-dir', default=None, help='Output directory for summary CSVs (default: same as csv_dir)')
	p.add_argument('--metrics', default=None, metavar='M1,M2,...', help='Comma-separated metrics to keep (default: all). E.g. --metrics hd95_gt_to_pred,assd,dice')
	p.add_argument('--predictions-names', default=None, metavar='JSON', help='JSON file mapping CSV stems to display names, e.g. {"mesh_metrics_run1": "Model A"}. Used in LaTeX table.')
	p.add_argument('--quiet', action='store_true', help='Reduce logging')
	args = p.parse_args(argv)

	csv_dir = args.csv_dir
	categories_path = args.categories_json
	out_dir = args.out_dir or csv_dir
	metrics_filter: Optional[List[str]] = None
	if args.metrics:
		metrics_filter = [m.strip() for m in args.metrics.split(',') if m.strip()]
		invalid = [
			m for m in metrics_filter
			if m not in METRIC_COLUMNS and not _is_volume_radii_metric(m)
		]
		if invalid:
			print(f'Error: Unknown metrics: {invalid}. Valid: {", ".join(METRIC_COLUMNS)}', file=sys.stderr)
			return 2

	if not os.path.isdir(csv_dir):
		print(f'Error: CSV directory not found: {csv_dir}', file=sys.stderr)
		return 2
	if not os.path.isfile(categories_path):
		print(f'Error: Categories file not found: {categories_path}', file=sys.stderr)
		return 2

	predictions_names: Dict[str, str] = {}
	if args.predictions_names:
		if not os.path.isfile(args.predictions_names):
			print(f'Error: Predictions names file not found: {args.predictions_names}', file=sys.stderr)
			return 2
		with open(args.predictions_names) as f:
			predictions_names = json.load(f)
		if not args.quiet:
			print(f'Loaded {len(predictions_names)} prediction display names from {args.predictions_names}')

	case_to_cat = load_categories(categories_path)
	if not args.quiet:
		print(f'Loaded {len(case_to_cat)} case->category mappings from {categories_path}')

	csv_files = [
		f for f in os.listdir(csv_dir)
		if f.lower().endswith('.csv') and f.lower() != 'summary.csv'
	]
	csv_files.sort()

	if not csv_files:
		print(f'No .csv files found in {csv_dir} (excluding summary.csv)', file=sys.stderr)
		return 2

	os.makedirs(out_dir, exist_ok=True)

	all_summaries: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

	for fname in csv_files:
		csv_path = os.path.join(csv_dir, fname)
		stem, _ = os.path.splitext(fname)
		out_name = f'{stem}_by_category.csv'
		out_path = os.path.join(out_dir, out_name)

		summary = analyze_csv_by_categories(csv_path, case_to_cat, columns=metrics_filter)
		write_category_summary(out_path, summary, metrics_filter)
		all_summaries[stem] = summary
		if not args.quiet:
			print(f'Wrote {out_path}')

	# Write LaTeX tables
	latex_path = os.path.join(out_dir, 'latex_table.txt')
	write_latex_table(latex_path, all_summaries, metrics_subset=metrics_filter, predictions_names=predictions_names)
	if not args.quiet:
		print(f'Wrote {latex_path}')
	latex_means_path = os.path.join(out_dir, 'latex_table_means.txt')
	write_latex_table(latex_means_path, all_summaries, metrics_subset=metrics_filter, predictions_names=predictions_names, means_only=True)
	if not args.quiet:
		print(f'Wrote {latex_means_path}')

	return 0


if __name__ == '__main__':
	raise SystemExit(main())
