"""LaTeX table output for journal-ready tables.

Provides functions to write metrics as rows in LaTeX format, suitable for
inclusion in manuscripts (e.g. Nature, IEEE).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


def _fmt_num(x: float, decimals: int = 3) -> str:
	"""Format number for LaTeX; use scientific notation when compact."""
	if math.isnan(x):
		return '---'
	if abs(x) >= 1000 or (0 < abs(x) < 0.001 and x != 0):
		return f'{x:.2e}'
	return f'{x:.{decimals}f}'


def format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
	"""Format as 'mean ± std' for LaTeX."""
	if math.isnan(mean):
		return '---'
	if math.isnan(std) or std == 0:
		return _fmt_num(mean, decimals)
	return f'{_fmt_num(mean, decimals)} $\\pm$ {_fmt_num(std, decimals)}'


def _escape_latex(s: str) -> str:
	"""Escape special LaTeX characters in table cell text."""
	return s.replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')


def _format_p_value(p: float) -> str:
	"""Format p-value for LaTeX: p < 0.001 for very small, else p = X.XXX."""
	if math.isnan(p) or p < 0:
		return '---'
	if p < 0.001:
		return '$p < 0.001$'
	return f'$p = {_fmt_num(p, 3)}$'


def write_pred_vs_pred_reference_latex_table(
	df: pd.DataFrame,
	metric_cols: list[str],
	reference: str,
	out_path: str,
	metric_display_names: Optional[dict[str, str]] = None,
	means_only: bool = False,
	include_pvalues: bool = True,
) -> None:
	"""Write LaTeX table with metrics as rows, groups as columns.

	Computes mean ± std per metric for 'Reference vs Others' and 'Others vs Others'.
	P-values from Mann-Whitney U test (two-sided). Output is journal-ready (e.g. Nature, IEEE).

	Args:
		df: DataFrame with 'group' column and metric columns.
		metric_cols: List of metric column names to include.
		reference: Reference prediction name (used for column headers).
		out_path: Path to write .txt file containing LaTeX.
		metric_display_names: Optional {metric_key: display_name} for row labels.
		means_only: If True, show only mean values (no ± std).
		include_pvalues: If True, add p-value column (Mann-Whitney U test).
	"""
	display_names = metric_display_names or {}
	group_order = ['Reference vs Others', 'Others vs Others']

	# Column headers: use reference-specific labels
	if reference.lower() == 'seqseg':
		col_ref = 'SeqSeg vs Manual'
		col_others = 'Manual vs Manual'
	else:
		col_ref = f'{reference} vs Others'
		col_others = 'Others vs Others'

	# Compute mean ± std and p-value per metric per group
	rows: list[tuple[str, str, str, str]] = []
	for metric in metric_cols:
		if metric not in df.columns:
			continue
		label = display_names.get(metric, metric.replace('_', ' ').title())
		label_tex = _escape_latex(label)

		vals_ref = df[df['group'] == group_order[0]][metric].dropna().values
		vals_others = df[df['group'] == group_order[1]][metric].dropna().values

		cells = []
		for vals in (vals_ref, vals_others):
			if len(vals) == 0:
				cells.append('---')
			else:
				mean_val = float(np.mean(vals))
				std_val = float(np.std(vals))
				if means_only:
					cells.append(_fmt_num(mean_val))
				else:
					cells.append(format_mean_std(mean_val, std_val))

		# Mann-Whitney U (two-sided)
		if include_pvalues and len(vals_ref) >= 2 and len(vals_others) >= 2:
			try:
				_, p = mannwhitneyu(vals_ref, vals_others, alternative='two-sided')
				p_str = _format_p_value(float(p))
			except Exception:
				p_str = '---'
		else:
			p_str = '---'

		if len(cells) == 2:
			rows.append((label_tex, cells[0], cells[1], p_str))

	if not rows:
		return

	# Build LaTeX
	lines = []
	lines.append('% Pred-vs-pred reference comparison. Metrics as rows.')
	lines.append('% P-values: Mann-Whitney U test (two-sided).')
	lines.append('% Copy into your LaTeX document. Ensure \\usepackage{booktabs} in preamble.')
	lines.append('')
	lines.append('\\begin{table}[htbp]')
	lines.append('\\centering')
	caption = 'Mean per metric.' if means_only else 'Mean $\\pm$ std per metric.'
	if include_pvalues:
		caption += ' P-values from Mann-Whitney U test.'
	lines.append(f'\\caption{{{caption}}}')
	lines.append('\\label{tab:pred_vs_pred_reference}')
	col_spec = 'lccc' if include_pvalues else 'lcc'
	lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
	lines.append('\\toprule')
	header = f'Metric & {_escape_latex(col_ref)} & {_escape_latex(col_others)}'
	if include_pvalues:
		header += ' & $p$-value'
	lines.append(header + ' \\\\')
	lines.append('\\midrule')
	for label_tex, cell_ref, cell_others, p_str in rows:
		row = f'{label_tex} & {cell_ref} & {cell_others}'
		if include_pvalues:
			row += f' & {p_str}'
		lines.append(row + ' \\\\')
	lines.append('\\bottomrule')
	lines.append('\\end{tabular}')
	lines.append('\\end{table}')

	with open(out_path, 'w') as f:
		f.write('\n'.join(lines))


def write_point_diff_latex_table(
	point_diffs: dict[str, dict[str, np.ndarray]],
	metric_cols: list[str],
	reference: str,
	out_path: str,
	metric_display_names: Optional[dict[str, str]] = None,
	means_only: bool = False,
	include_pvalues: bool = True,
) -> None:
	"""Write LaTeX table for point-wise pairwise differences.

	Metrics as rows; columns are 'Reference vs Others' and 'Others vs Others'
	with mean ± std of |pairwise differences| per case. P-values from Mann-Whitney U test.

	Args:
		point_diffs: {metric: {group: array of values}} from _compute_point_differences.
		metric_cols: List of metric names.
		reference: Reference name for column headers.
		out_path: Output .txt path.
		metric_display_names: Optional display names for metrics.
		means_only: If True, show only mean.
		include_pvalues: If True, add p-value column (Mann-Whitney U test).
	"""
	display_names = metric_display_names or {}
	group_order = ['Reference vs Others', 'Others vs Others']

	if reference.lower() == 'seqseg':
		col_ref = 'SeqSeg vs Manual'
		col_others = 'Manual vs Manual'
	else:
		col_ref = f'{reference} vs Others'
		col_others = 'Others vs Others'

	rows: list[tuple[str, str, str, str]] = []
	for metric in metric_cols:
		label = display_names.get(metric, metric.replace('_', ' ').title())
		label_tex = _escape_latex(label) + ' (|pairwise diff|)'

		vals_ref = np.asarray(point_diffs.get(metric, {}).get(group_order[0], np.array([])))
		vals_others = np.asarray(point_diffs.get(metric, {}).get(group_order[1], np.array([])))

		cells = []
		for vals in (vals_ref, vals_others):
			if len(vals) == 0:
				cells.append('---')
			else:
				mean_val = float(np.mean(vals))
				std_val = float(np.std(vals))
				if means_only:
					cells.append(_fmt_num(mean_val))
				else:
					cells.append(format_mean_std(mean_val, std_val))

		if include_pvalues and len(vals_ref) >= 2 and len(vals_others) >= 2:
			try:
				_, p = mannwhitneyu(vals_ref, vals_others, alternative='two-sided')
				p_str = _format_p_value(float(p))
			except Exception:
				p_str = '---'
		else:
			p_str = '---'

		rows.append((label_tex, cells[0], cells[1], p_str))

	if not rows:
		return

	lines = []
	lines.append('% Point-wise pairwise metric differences. Metrics as rows.')
	lines.append('% P-values: Mann-Whitney U test (two-sided).')
	lines.append('% Copy into your LaTeX document. Ensure \\usepackage{booktabs} in preamble.')
	lines.append('')
	lines.append('\\begin{table}[htbp]')
	lines.append('\\centering')
	caption = 'Mean |pairwise diff| per metric.' if means_only else 'Mean $\\pm$ std |pairwise diff| per metric.'
	if include_pvalues:
		caption += ' P-values from Mann-Whitney U test.'
	lines.append(f'\\caption{{{caption}}}')
	lines.append('\\label{tab:point_diff_reference}')
	col_spec = 'lccc' if include_pvalues else 'lcc'
	lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
	lines.append('\\toprule')
	header = f'Metric & {_escape_latex(col_ref)} & {_escape_latex(col_others)}'
	if include_pvalues:
		header += ' & $p$-value'
	lines.append(header + ' \\\\')
	lines.append('\\midrule')
	for label_tex, cell_ref, cell_others, p_str in rows:
		row = f'{label_tex} & {cell_ref} & {cell_others}'
		if include_pvalues:
			row += f' & {p_str}'
		lines.append(row + ' \\\\')
	lines.append('\\bottomrule')
	lines.append('\\end{tabular}')
	lines.append('\\end{table}')

	with open(out_path, 'w') as f:
		f.write('\n'.join(lines))


def write_pred_vs_pred_all_comparisons_latex_table(
	df: pd.DataFrame,
	metric_cols: list[str],
	out_path: str,
	metric_display_names: Optional[dict[str, str]] = None,
	metric_abbreviations: Optional[dict[str, str]] = None,
	comparison_display_names: Optional[dict[str, str]] = None,
	means_only: bool = False,
	include_pvalues: bool = True,
) -> None:
	"""Write LaTeX table with comparisons as rows, metrics as columns.

	Computes mean ± std per metric per comparison. P-values from Kruskal-Wallis
	test (tests whether distributions differ across comparisons). Output is
	journal-ready (e.g. Nature, IEEE).

	Args:
		df: DataFrame with 'comparison' column and metric columns.
		metric_cols: List of metric column names to include.
		out_path: Path to write .txt file containing LaTeX.
		metric_display_names: Optional {metric_key: display_name} for caption abbreviations.
		metric_abbreviations: Optional {metric_key: abbrev} for column headers; falls back to display_names.
		comparison_display_names: Optional {comparison_key: display_name} for row labels.
		means_only: If True, show only mean values (no ± std).
		include_pvalues: If True, add p-value column (Kruskal-Wallis test).
	"""
	display_names = metric_display_names or {}
	abbrevs = metric_abbreviations or display_names
	comp_display = comparison_display_names or {}

	comparisons = sorted(df['comparison'].unique())

	# Filter to metrics that exist
	metric_cols = [m for m in metric_cols if m in df.columns]
	if not metric_cols:
		return

	# Compute mean ± std per comparison per metric; each row = one comparison
	rows: list[tuple[str, list[str]]] = []
	for comp in comparisons:
		comp_label = comp_display.get(comp, comp.replace('_vs_', ' vs '))
		comp_label_tex = _escape_latex(comp_label)

		cells = []
		for metric in metric_cols:
			vals = df[df['comparison'] == comp][metric].dropna().values
			if len(vals) == 0:
				cells.append('---')
			else:
				mean_val = float(np.mean(vals))
				std_val = float(np.std(vals))
				if means_only:
					cells.append(_fmt_num(mean_val))
				else:
					cells.append(format_mean_std(mean_val, std_val))

		rows.append((comp_label_tex, cells))

	# Column headers: abbreviations (or full names if no abbreviations)
	col_headers = []
	for metric in metric_cols:
		abbrev = abbrevs.get(metric, display_names.get(metric, metric.replace('_', ' ').title()))
		# Don't escape if already contains LaTeX math (e.g. SD$_{\tau 1}$)
		col_headers.append(abbrev if '$' in abbrev else _escape_latex(abbrev))

	# Compute Kruskal-Wallis p-value per metric (across comparisons) for a summary row
	pvalue_row: list[str] = []
	if include_pvalues:
		for metric in metric_cols:
			vals_by_comp = [
				df[df['comparison'] == comp][metric].dropna().values
				for comp in comparisons
			]
			non_empty = [v for v in vals_by_comp if len(v) >= 1]
			if len(non_empty) >= 2:
				try:
					_, p = kruskal(*non_empty)
					pvalue_row.append(_format_p_value(float(p)))
				except Exception:
					pvalue_row.append('---')
			else:
				pvalue_row.append('---')

	# Build LaTeX
	lines = []
	lines.append('% Pred-vs-pred all comparisons. Comparisons as rows, metrics as columns.')
	lines.append('% P-values: Kruskal-Wallis test (across comparisons, per metric).')
	lines.append('% Copy into your LaTeX document. Ensure \\usepackage{booktabs} in preamble.')
	lines.append('')
	# Build caption with abbreviation descriptions
	caption = 'Mean per metric per comparison.' if means_only else 'Mean $\\pm$ std per metric per comparison.'
	if include_pvalues:
		caption += ' P-values from Kruskal-Wallis test (per metric, across comparisons).'
	abbrev_desc_pairs = [
		f'{abbrevs.get(m, m)}: {_escape_latex(display_names.get(m, m.replace("_", " ").title()))}'
		for m in metric_cols
	]
	if abbrev_desc_pairs:
		caption += ' Abbreviations: ' + '; '.join(abbrev_desc_pairs) + '.'
	lines.append('\\begin{table}[htbp]')
	lines.append('\\centering')
	lines.append(f'\\caption{{{caption}}}')
	lines.append('\\label{tab:pred_vs_pred_all_comparisons}')
	col_spec = 'l' + 'c' * len(metric_cols)
	lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
	lines.append('\\toprule')
	lines.append('Comparison & ' + ' & '.join(col_headers) + ' \\\\')
	lines.append('\\midrule')
	for comp_label_tex, cells in rows:
		lines.append(comp_label_tex + ' & ' + ' & '.join(cells) + ' \\\\')
	if include_pvalues and pvalue_row:
		lines.append('\\midrule')
		lines.append('$p$ (Kruskal-Wallis) & ' + ' & '.join(pvalue_row) + ' \\\\')
	lines.append('\\bottomrule')
	lines.append('\\end{tabular}')
	lines.append('\\end{table}')

	with open(out_path, 'w') as f:
		f.write('\n'.join(lines))
