from __future__ import annotations

import html
from typing import Any, Iterable, Mapping, Optional


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def markdown_table(rows: list[Mapping[str, Any]], *, columns: list[str]) -> str:
    if not rows:
        return "_(no rows)_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |")
    return "\n".join([header, sep, *body])


def html_table(rows: list[Mapping[str, Any]], *, columns: list[str]) -> str:
    if not rows:
        return "<p><em>(no rows)</em></p>"
    ths = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{html.escape(_fmt(r.get(c)))}</td>" for c in columns)
        trs.append(f"<tr>{tds}</tr>")
    return f"""
<div class="table-wrap">
<table>
  <thead><tr>{ths}</tr></thead>
  <tbody>
    {''.join(trs)}
  </tbody>
</table>
</div>
""".strip()


def html_page(*, title: str, body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #586174;
      --line: #e5e7eb;
      --head: #f3f4f6;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
    }}
    .container {{
      max-width: 1200px;
      margin: 24px auto;
      padding: 0 16px 32px;
    }}
    h1, h2, h3 {{ margin: 0.7em 0 0.28em; line-height: 1.25; }}
    h1 {{ font-size: 2rem; }}
    h2 {{ font-size: 1.28rem; border-bottom: 1px solid var(--line); padding-bottom: 6px; }}
    h3 {{ font-size: 1rem; color: #111827; }}
    p, ul, pre {{ margin-top: 0.55em; margin-bottom: 0.85em; }}
    ul {{ padding-left: 20px; }}
    li + li {{ margin-top: 4px; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{
      background: #eef2ff;
      color: #1e3a8a;
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 0.92em;
    }}
    pre {{
      background: #0b1020;
      color: #e5ecff;
      padding: 12px;
      border-radius: 10px;
      overflow: auto;
      font-size: 0.9rem;
      border: 1px solid #1f2a44;
    }}
    .muted {{ color: var(--muted); }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 760px;
      margin: 0;
      background: var(--panel);
      font-size: 0.94rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      background: var(--head);
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tr:nth-child(even) td {{ background: #fafafa; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: var(--panel);
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .card h3 {{ margin-top: 0; margin-bottom: 6px; }}
    .spark {{ width: 100%; height: 72px; display: block; margin: 4px 0 2px; }}
  </style>
</head>
<body>
<main class="container">
{body_html}
</main>
</body>
</html>
"""


def sparkline_svg(
    ys: list[float],
    *,
    width: int = 260,
    height: int = 70,
    stroke: str = "#1f77b4",
    stroke_width: int = 2,
) -> str:
    """
    Render a simple inline SVG sparkline.
    """
    if not ys:
        return '<svg class="spark" viewBox="0 0 260 70"></svg>'

    # Downsample to keep HTML size reasonable.
    max_points = 200
    if len(ys) > max_points:
        step = max(1, len(ys) // max_points)
        ys = ys[::step]

    y_min = min(ys)
    y_max = max(ys)
    if y_max == y_min:
        y_max = y_min + 1.0

    def x(i: int) -> float:
        return (i / max(1, len(ys) - 1)) * (width - 2) + 1

    def y(v: float) -> float:
        # invert y axis
        return (height - 2) - ((v - y_min) / (y_max - y_min)) * (height - 2) + 1

    pts_list = [f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(ys)]
    # Insert line breaks every ~50 points to avoid huge single lines.
    pts = "\n    ".join(" ".join(pts_list[i : i + 50]) for i in range(0, len(pts_list), 50))
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" role="img" aria-label="equity curve sparkline">'
        f'<polyline fill="none" stroke="{html.escape(stroke)}" stroke-width="{stroke_width}" points="\n    {pts}\n  " />'
        f"</svg>"
    )

