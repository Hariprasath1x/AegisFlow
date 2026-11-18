"""Unit tests for HTML and JSON evaluation report generation."""
from __future__ import annotations

import json
import pathlib
import pytest

from aegis_ml.evaluation.reports import (
    CSS,
    _grid,
    _poly,
    _rows,
    line_chart,
    write_report,
)


def test_poly_polyline_generation():
    """_poly maps coordinates into SVG space and generates polyline element."""
    points = [(0.0, 0.0), (1.0, 1.0)]
    poly_str = _poly(points, width=470, height=300, colour="#2E6FD9", xlim=(0, 1), ylim=(0, 1))
    assert poly_str.startswith('<polyline points="')
    assert 'stroke="#2E6FD9"' in poly_str
    assert 'fill="none"' in poly_str


def test_line_chart_svg_structure_and_escaping():
    """line_chart generates valid SVG and escapes labels and series names."""
    series = {
        "ROC <Curve> & 'Test'": [(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)],
        "Baseline": [(0.0, 0.0), (1.0, 1.0)],
    }
    svg = line_chart(
        series,
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False Positive Rate (<FPR>)",
        ylabel="True Positive Rate (<TPR>)",
    )
    assert svg.startswith('<svg width="470" height="300">')
    assert svg.endswith("</svg>")
    # Verify HTML escaping
    assert "ROC &lt;Curve&gt; &amp; &#x27;Test&#x27;" in svg
    assert "&lt;FPR&gt;" in svg
    assert "&lt;TPR&gt;" in svg


def test_rows_formatting_and_escaping():
    """_rows creates a table from a key-value dictionary and formats floats."""
    data = {
        "<Metric>": 0.123456,
        "Name": "Model_A & B",
        "Integer": 42,
    }
    table = _rows(data)
    assert "<table>" in table
    assert "<th>&lt;Metric&gt;</th><td>0.1235</td>" in table
    assert "<td>Model_A &amp; B</td>" in table
    assert "<td>42</td>" in table


def test_grid_formatting_and_empty():
    """_grid renders tabular records or '<p>no rows</p>' if empty."""
    assert _grid([]) == "<p>no rows</p>"

    rows = [
        {"col1": "val1 & A", "col2": 0.54321},
        {"col1": "val2", "col2": 0.99999},
    ]
    table = _grid(rows)
    assert "<table><tr><th>col1</th><th>col2</th></tr>" in table
    assert "<td>val1 &amp; A</td><td>0.5432</td>" in table


def test_write_report_html_and_json(tmp_path: pathlib.Path):
    """write_report creates parent directories and writes self-contained HTML and JSON."""
    report_html = tmp_path / "subdir" / "m3_fraud.html"
    report_json = tmp_path / "subdir" / "m3_fraud.json"

    gates = [
        ("recall_at_1pct_fpr", True, "recall=0.58 >= 0.50 [pass]"),
        ("ece", False, "ece=0.08 <= 0.05 [FAIL]"),
    ]
    sections = [
        {
            "heading": "Champion: <XGBoost>",
            "note": "Note with <html> & dangerous 'quotes'",
            "metrics": {"recall": 0.58123},
        },
        {
            "heading": "Reliability",
            "rows": [{"bin": "[0.0,0.1)", "n": 100, "observed": 0.05}],
            "svg": "<svg>dummy</svg>",
        },
    ]

    result_path = write_report(
        path=report_html,
        title="Fraud Evaluation (<M3>)",
        sections=sections,
        gates=gates,
        out_json=report_json,
    )

    assert result_path == str(report_html)
    assert report_html.exists()
    assert report_json.exists()

    html_content = report_html.read_text(encoding="utf-8")
    assert "<!doctype html>" in html_content
    assert "<title>Fraud Evaluation (&lt;M3&gt;)</title>" in html_content
    assert CSS in html_content
    # Gates table check
    assert "<th>Gate</th><th>Measured</th><th>Result</th>" in html_content
    assert "<td class='pass'>PASS</td>" in html_content
    assert "<td class='fail'>FAIL</td>" in html_content
    # Escaping check
    assert "Champion: &lt;XGBoost&gt;" in html_content
    assert "Note with &lt;html&gt; &amp; dangerous &#x27;quotes&#x27;" in html_content
    # SVG check
    assert "<svg>dummy</svg>" in html_content

    # JSON structure check
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["title"] == "Fraud Evaluation (<M3>)"
    assert len(payload["gates"]) == 2
    assert payload["gates"][0] == {
        "gate": "recall_at_1pct_fpr",
        "pass": True,
        "detail": "recall=0.58 >= 0.50 [pass]",
    }
    assert payload["gates"][1] == {
        "gate": "ece",
        "pass": False,
        "detail": "ece=0.08 <= 0.05 [FAIL]",
    }
    # Sections in JSON must exclude svg
    assert len(payload["sections"]) == 2
    assert "svg" not in payload["sections"][1]
    assert payload["sections"][1]["rows"][0]["n"] == 100
