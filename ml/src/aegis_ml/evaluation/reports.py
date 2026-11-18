"""Self-contained HTML evaluation reports logged as MLflow artifacts.

Inline SVG and no external CSS, so the report opens from the MLflow artifact
browser on any machine and still renders when it is emailed to a compliance
reviewer who has no access to the cluster.
"""
from __future__ import annotations

import html
import json
import pathlib

CSS = (
    "body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:28px;"
    "color:#1a2330}h1{color:#15294B}h2{color:#1F4E8C;margin-top:26px}"
    "table{border-collapse:collapse;margin:10px 0}td,th{border:1px solid #c9d2df;"
    "padding:5px 9px;text-align:left}th{background:#eaf0f8}"
    ".pass{color:#1d7a40;font-weight:600}.fail{color:#b3261e;font-weight:600}"
)


def _poly(points, width, height, colour, xlim, ylim):
    (x0, x1), (y0, y1) = xlim, ylim
    sx = (width - 48) / max(x1 - x0, 1e-9)
    sy = (height - 40) / max(y1 - y0, 1e-9)
    pts = " ".join(
        f"{40 + (x - x0) * sx:.1f},{height - 28 - (y - y0) * sy:.1f}"
        for x, y in points
    )
    return f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.8"/>'


def line_chart(series: dict, xlim, ylim, xlabel="", ylabel="", width=470, height=300):
    """series: {label: [(x, y), ...]}. Two or three lines maximum - this is a
    report, not a dashboard."""
    palette = ["#2E6FD9", "#b3261e", "#1d7a40"]
    body = "".join(
        _poly(pts, width, height, palette[i % 3], xlim, ylim)
        for i, pts in enumerate(series.values())
    )
    legend = "".join(
        f'<text x="{width - 150}" y="{24 + 14 * i}" font-size="11" '
        f'fill="{palette[i % 3]}">{html.escape(name)}</text>'
        for i, name in enumerate(series)
    )
    return (
        f'<svg width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="#fff" stroke="#c9d2df"/>'
        f"{body}{legend}"
        f'<text x="{width / 2}" y="{height - 6}" font-size="11" text-anchor="middle">'
        f"{html.escape(xlabel)}</text>"
        f'<text x="12" y="{height / 2}" font-size="11" transform="rotate(-90 12 '
        f'{height / 2})" text-anchor="middle">{html.escape(ylabel)}</text></svg>'
    )


def _rows(d: dict) -> str:
    out = []
    for k, v in d.items():
        v = f"{v:.4f}" if isinstance(v, float) else v
        out.append(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>")
    return "<table>" + "".join(out) + "</table>"


def _grid(rows: list[dict]) -> str:
    if not rows:
        return "<p>no rows</p>"
    cols = list(rows[0])
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(f'{r[c]:.4f}' if isinstance(r[c], float) else str(r[c]))}</td>"
            for c in cols
        )
        + "</tr>"
        for r in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def write_report(path, title, sections, gates=(), out_json=None):
    """sections: [{'heading', 'metrics'?: dict, 'rows'?: list[dict], 'svg'?: str,
    'note'?: str}]"""
    parts = [f"<h1>{html.escape(title)}</h1>"]
    if gates:
        parts.append(
            "<h2>Gates</h2><table><tr><th>Gate</th><th>Measured</th>"
            "<th>Result</th></tr>"
            + "".join(
                f"<tr><td>{html.escape(n)}</td><td>{html.escape(d)}</td>"
                f"<td class='{'pass' if ok else 'fail'}'>"
                f"{'PASS' if ok else 'FAIL'}</td></tr>"
                for n, ok, d in gates
            )
            + "</table>"
        )
    for s in sections:
        parts.append(f"<h2>{html.escape(s['heading'])}</h2>")
        if s.get("note"):
            parts.append(f"<p>{html.escape(s['note'])}</p>")
        if s.get("metrics"):
            parts.append(_rows(s["metrics"]))
        if s.get("rows"):
            parts.append(_grid(s["rows"]))
        if s.get("svg"):
            parts.append(s["svg"])
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}"
        f"</title><style>{CSS}</style>" + "".join(parts),
        encoding="utf-8",
    )
    if out_json:
        payload = {
            "title": title,
            "gates": [{"gate": n, "pass": ok, "detail": d} for n, ok, d in gates],
            "sections": [
                {k: v for k, v in s.items() if k != "svg"} for s in sections
            ],
        }
        pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_json).write_text(
            json.dumps(payload, indent=2, default=float), encoding="utf-8"
        )
    return str(p)
