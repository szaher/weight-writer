"""JSON, HTML, and SVG reports for model weight differences."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import html
import json
import math
import os
import re
from .compare import ModelDiff, TensorDiff

METRIC_LABELS = {
    "mean_abs_diff": "Mean absolute difference",
    "max_abs_diff": "Maximum absolute difference",
    "l2_diff": "L2 difference",
    "relative_l2": "Relative L2 difference",
    "changed_fraction": "Changed element fraction",
    "bytes_changed_per_element": "Changed bytes per element",
    "bytes_changed": "Changed bytes",
}

METRIC_PRIORITY = [
    "relative_l2",
    "mean_abs_diff",
    "changed_fraction",
    "bytes_changed_per_element",
    "bytes_changed",
]

_LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|layer|h)\.(\d+)\.")


@dataclass(frozen=True)
class HeatmapCell:
    row: int | str
    column: str
    value: float
    weight: int
    tensor_count: int


def metric_value(tensor: TensorDiff, metric: str) -> float | None:
    if metric == "changed_fraction":
        return tensor.changed_fraction
    if metric == "bytes_changed_per_element":
        return tensor.bytes_changed / tensor.numel if tensor.numel else None
    value = getattr(tensor, metric, None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def available_metrics(diff: ModelDiff) -> list[str]:
    available = []
    for metric in METRIC_PRIORITY:
        if any(metric_value(tensor, metric) is not None for tensor in diff.tensors):
            available.append(metric)
    return available


def choose_metric(diff: ModelDiff, requested: str | None = None) -> str:
    if requested:
        if requested not in METRIC_LABELS:
            raise ValueError(f"Unknown metric {requested!r}")
        return requested
    available = available_metrics(diff)
    if not available:
        # Models with no tensor names in common still deserve a report showing
        # the missing tensors.
        return "bytes_changed"
    return available[0]


def write_json_report(diff: ModelDiff, output: str | os.PathLike[str]) -> None:
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(diff.to_dict(), handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_html_report(
    diff: ModelDiff,
    output: str | os.PathLike[str],
    *,
    metric: str | None = None,
    svg_path: str | os.PathLike[str] | None = None,
    top_rows: int | None = None,
) -> str:
    selected_metric = choose_metric(diff, metric)
    heatmap = render_layer_heatmap_svg(diff, selected_metric)

    counts = _status_counts(diff)
    changed_tensors = [
        tensor for tensor in diff.tensors if tensor.status != "identical"
    ]
    changed_tensors.sort(
        key=lambda tensor: (
            metric_value(tensor, selected_metric) is None,
            -(metric_value(tensor, selected_metric) or 0.0),
            tensor.name,
        )
    )
    if top_rows is not None:
        changed_tensors = changed_tensors[:top_rows]

    title = "LLM weight difference report"
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        """
:root { color-scheme: light dark; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
body { margin: 0; padding: 32px; background: #f6f7f9; color: #17202a; }
main { max-width: 1400px; margin: 0 auto; }
h1 { margin: 0 0 8px; font-size: 28px; }
h2 { margin-top: 36px; font-size: 20px; }
.paths { margin: 0; color: #52606d; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; overflow-wrap: anywhere; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 24px; }
.card { background: white; border: 1px solid #d9e2ec; border-radius: 10px; padding: 16px; }
.card strong { display: block; font-size: 24px; margin-bottom: 4px; }
.card span { color: #52606d; font-size: 13px; }
.heatmap { background: white; border: 1px solid #d9e2ec; border-radius: 12px; padding: 16px; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9e2ec; border-radius: 10px; overflow: hidden; font-size: 13px; }
th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid #e6ebf1; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { background: #edf2f7; position: sticky; top: 0; }
tbody tr:hover { background: #f0f5fa; }
tbody tr:last-child td { border-bottom: 0; }
.name { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; max-width: 520px; overflow: hidden; text-overflow: ellipsis; }
.identical { color: #249f77; }
.changed { color: #b93d3d; }
.mismatch { color: #94470b; background: #fff4e5; }
svg { max-width: none; }
@media (prefers-color-scheme: dark) {
 body { background: #101418; color: #e6edf3; }
 .card, .heatmap, table { background: #171d24; border-color: #303a45; }
 th { background: #1f2731; }
 tbody tr:hover { background: #1d2630; }
 .paths { color: #98a5b3; }
}
""",
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="paths">A: {html.escape(diff.path_a)}<br>B: {html.escape(diff.path_b)}</p>',
        '<div class="cards">',
        _card(len(diff.tensors), "Compared tensors"),
        _card(counts["identical"], "Identical"),
        _card(counts["changed"], "Changed"),
        _card(counts["mismatch"], "Shape/dtype mismatch"),
        _card(len(diff.missing_in_a), "Missing in A"),
        _card(len(diff.missing_in_b), "Missing in B"),
        "</div>",
        "<h2>Weight difference heatmap</h2>",
        '<div class="heatmap">',
        f'<p><strong>{html.escape(METRIC_LABELS[selected_metric])}</strong>, aggregated by layer and parameter type.</p>',
        heatmap,
        "</div>",
        "<h2>Detailed tensor differences</h2>",
        "<table><thead><tr>",
        "<th>Tensor</th><th>Shape</th><th>Dtype</th><th>Status</th>",
        "<th>Bytes changed</th><th>Elements changed</th><th>Mean |Δ|</th>",
        "<th>Max |Δ|</th><th>Relative L2</th><th>Cosine</th>",
        "</tr></thead><tbody>",
    ]

    for tensor in changed_tensors:
        parts.append(_table_row(tensor, selected_metric))

    if not changed_tensors:
        if diff.tensors:
            message = "All compared tensors are identical."
        else:
            message = (
                "No tensors with matching names were found. "
                "The selected models use incompatible tensor layouts and are not directly comparable."
            )
        parts.append(
            f'<tr><td colspan="10" style="text-align:center">{html.escape(message)}</td></tr>'
        )

    parts.extend(["</tbody></table>"])
    if diff.missing_in_a or diff.missing_in_b:
        parts.append("<h2>Missing tensors</h2><ul>")
        for name in diff.missing_in_a:
            parts.append(f"<li>Missing in A: <code>{html.escape(name)}</code></li>")
        for name in diff.missing_in_b:
            parts.append(f"<li>Missing in B: <code>{html.escape(name)}</code></li>")
        parts.append("</ul>")

    parts.extend(["</main>", "</body>", "</html>", ""])
    document = "\n".join(parts)
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(document)
    if svg_path is not None:
        with open(svg_path, "w", encoding="utf-8") as handle:
            handle.write(heatmap)
    return selected_metric


def render_layer_heatmap_svg(diff: ModelDiff, metric: str) -> str:
    cells = _aggregate_by_layer(diff, metric)
    if not cells:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="120" '
            'role="img" aria-label="No heatmap data">'
            '<rect width="100%" height="100%" fill="#f0f3f6"/><text x="24" y="54" '
            'font-family="sans-serif" font-size="16" fill="#52606d">No numeric differences available</text></svg>'
        )

    rows = sorted({cell.row for cell in cells}, key=_row_sort_key)
    columns = sorted({cell.column for cell in cells})
    row_lookup = {row: index for index, row in enumerate(rows)}
    column_lookup = {column: index for index, column in enumerate(columns)}

    row_height = 22
    column_width = 118
    label_width = 190
    top_height = 132
    width = label_width + len(columns) * column_width + 30
    height = top_height + len(rows) * row_height + 48
    values = [cell.value for cell in cells]
    maximum = max(values)
    minimum = min(values)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(METRIC_LABELS[metric])} heatmap">',
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        f'<text x="{label_width}" y="28" font-family="sans-serif" font-size="17" '
        f'font-weight="600" fill="#17202a">{html.escape(METRIC_LABELS[metric])}</text>',
        f'<text x="{label_width}" y="49" font-family="sans-serif" font-size="12" '
        f'fill="#52606d">Aggregated by layer and parameter type</text>',
    ]

    for column_index, column in enumerate(columns):
        x = label_width + column_index * column_width
        label = _truncate(column, 17)
        parts.append(
            f'<text x="{x + 6}" y="78" font-family="ui-monospace, monospace" '
            f'font-size="10" fill="#52606d">{html.escape(label)}</text>'
        )

    for row_index, row in enumerate(rows):
        y = top_height + row_index * row_height
        label = f"layer {row}" if isinstance(row, int) else str(row)
        parts.append(
            f'<text x="{label_width - 12}" y="{y + 15}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11" fill="#52606d">{html.escape(_truncate(label, 25))}</text>'
        )

    for cell in cells:
        x = label_width + column_lookup[cell.column] * column_width
        y = top_height + row_lookup[cell.row] * row_height
        color = _value_color(cell.value, minimum, maximum)
        title = (
            f"Layer {cell.row} / {cell.column}: {_format_number(cell.value)} "
            f"over {cell.tensor_count} tensor(s), {cell.weight:,} elements"
        )
        parts.append(
            f'<rect x="{x + 2}" y="{y + 2}" width="{column_width - 5}" '
            f'height="{row_height - 5}" rx="3" fill="{color}">'
            f"<title>{html.escape(title)}</title></rect>"
        )

    legend_x = label_width
    legend_y = height - 27
    for index in range(80):
        t = index / 79
        color = _value_color(minimum + t * (maximum - minimum), minimum, maximum) if maximum != minimum else "#c7d0da"
        parts.append(
            f'<rect x="{legend_x + index * 7}" y="{legend_y}" width="7" height="12" fill="{color}"/>'
        )
    parts.append(
        f'<text x="{legend_x}" y="{legend_y - 6}" font-family="sans-serif" font-size="10" '
        f'fill="#52606d">{html.escape(_format_number(minimum))}</text>'
    )
    parts.append(
        f'<text x="{legend_x + 560}" y="{legend_y - 6}" text-anchor="end" font-family="sans-serif" '
        f'font-size="10" fill="#52606d">{html.escape(_format_number(maximum))}</text>'
    )
    parts.append("</svg>\n")
    return "".join(parts)


def _aggregate_by_layer(diff: ModelDiff, metric: str) -> list[HeatmapCell]:
    grouped: dict[tuple[int | str, str], list[tuple[float, int]]] = defaultdict(list)
    for tensor in diff.tensors:
        if tensor.status == "shape_or_dtype_mismatch":
            continue
        value = metric_value(tensor, metric)
        if value is None:
            continue
        row, column = _split_layer_name(tensor.name)
        grouped[(row, column)].append((value, tensor.numel))

    cells = []
    for (row, column), entries in grouped.items():
        if metric == "max_abs_diff":
            value = max(entry_value for entry_value, _ in entries)
        elif metric == "bytes_changed":
            value = sum(entry_value for entry_value, _ in entries)
        elif metric == "l2_diff":
            value = math.sqrt(sum(entry_value ** 2 for entry_value, _ in entries))
        else:
            total_weight = sum(weight for _, weight in entries)
            value = (
                sum(entry_value * weight for entry_value, weight in entries) / total_weight
                if total_weight
                else 0.0
            )
        cells.append(
            HeatmapCell(
                row=row,
                column=column,
                value=value,
                weight=sum(weight for _, weight in entries),
                tensor_count=len(entries),
            )
        )
    return cells


def _split_layer_name(name: str) -> tuple[int | str, str]:
    match = _LAYER_PATTERN.search(name)
    if match:
        return int(match.group(1)), name[match.end() :]
    parts = name.split(".")
    if len(parts) > 1:
        return parts[0], ".".join(parts[1:])
    return "model", name


def _status_counts(diff: ModelDiff) -> dict[str, int]:
    counts = {"identical": 0, "changed": 0, "mismatch": 0}
    for tensor in diff.tensors:
        if tensor.status == "identical":
            counts["identical"] += 1
        elif tensor.status == "shape_or_dtype_mismatch":
            counts["mismatch"] += 1
        else:
            counts["changed"] += 1
    return counts


def _table_row(tensor: TensorDiff, selected_metric: str) -> str:
    status_class = (
        "identical" if tensor.status == "identical" else
        "mismatch" if tensor.status == "shape_or_dtype_mismatch" else "changed"
    )
    metric = metric_value(tensor, selected_metric)
    intensity = _relative_intensity(metric, tensor)
    if intensity is None:
        mean_cell = f"<td>{_format_number(tensor.mean_abs_diff)}</td>"
    else:
        background = f"hsl(8, 78%, {94 - intensity * 18:.0f}%)"
        mean_cell = f'<td style="background:{background}">{_format_number(tensor.mean_abs_diff)}</td>'
    return (
        "<tr>"
        f'<td class="name" title="{html.escape(tensor.name)}">{html.escape(tensor.name)}</td>'
        f"<td>{html.escape(' × '.join(map(str, tensor.shape)))}</td>"
        f"<td>{html.escape(tensor.dtype)}</td>"
        f'<td class="{status_class}">{html.escape(tensor.status)}</td>'
        f"<td>{tensor.bytes_changed:,}</td>"
        f"<td>{_format_count(tensor.elements_changed)}</td>"
        f"{mean_cell}"
        f"<td>{_format_number(tensor.max_abs_diff)}</td>"
        f"<td>{_format_number(tensor.relative_l2)}</td>"
        f"<td>{_format_number(tensor.cosine_similarity)}</td>"
        "</tr>"
    )


def _relative_intensity(value: float | None, tensor: TensorDiff) -> float | None:
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    return min(1.0, math.log1p(value) / 6.0)


def _value_color(value: float, minimum: float, maximum: float) -> str:
    if not math.isfinite(value) or value <= 0:
        return "#e6ebf1"
    if maximum <= minimum:
        normalized = 0.5
    else:
        transformed = math.log1p(value)
        transformed_min = math.log1p(max(0.0, minimum))
        transformed_max = math.log1p(maximum)
        normalized = (transformed - transformed_min) / max(transformed_max - transformed_min, 1e-12)
    hue = 215 - normalized * 215
    lightness = 70 - normalized * 22
    return f"hsl({hue:.1f}, 72%, {lightness:.1f}%)"


def _row_sort_key(row: int | str) -> tuple[int, int | str]:
    if isinstance(row, int):
        return (0, row)
    return (1, row)


def _truncate(value: str, maximum: int) -> str:
    return value if len(value) <= maximum else value[: maximum - 1] + "…"


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if not math.isfinite(value):
        return "non-finite"
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 1_000_000:
        return f"{value:.3e}"
    return f"{value:.6g}"


def _format_count(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _card(value: int, label: str) -> str:
    return f'<div class="card"><strong>{value:,}</strong><span>{html.escape(label)}</span></div>'


__all__ = [
    "HeatmapCell",
    "METRIC_LABELS",
    "available_metrics",
    "choose_metric",
    "metric_value",
    "render_layer_heatmap_svg",
    "write_html_report",
    "write_json_report",
]
