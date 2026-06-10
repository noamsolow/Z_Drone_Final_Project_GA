from pathlib import Path
from xml.sax.saxutils import escape


MODELS = [
    ("scale only", 47.75, "#d1d5db"),
    ("depth linear", 32.11, "#cbd5e1"),
    ("linear+bb", 14.50, "#94a3b8"),
    ("improved linear", 12.55, "#64748b"),
    ("RF", 7.46, "#475569"),
    ("RF+jitter", 6.3614, "#0f766e"),
    ("ensemble", 5.9725, "#059669"),
]


def text(
    x: float,
    y: float,
    value: str,
    size: int,
    fill: str,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr}/>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, rx: float = 0, stroke: str | None = None) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{stroke_attr}/>'


def main() -> None:
    output_path = Path(__file__).resolve().parent / "synthetic_model_progress_mae.svg"

    width = 1280
    height = 760
    left = 250
    right = 90
    top = 120
    bottom = 72
    chart_w = width - left - right
    row_h = 78
    bar_h = 38
    max_mae = 50.0

    bg = "#ffffff"
    grid = "#e5e7eb"
    axis = "#9ca3af"
    text_main = "#111827"
    text_muted = "#6b7280"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, bg),
        text(left, 58, "Synthetic Model Progress", 34, text_main, 700),
        text(left, 88, "Test MAE [m]   lower is better", 18, text_muted, 400),
    ]

    chart_top = top
    chart_bottom = top + row_h * len(MODELS)

    for tick in range(0, 51, 10):
        x = left + chart_w * (tick / max_mae)
        parts.append(line(x, chart_top - 8, x, chart_bottom + 8, grid, 1, "4 6"))
        parts.append(text(x, chart_bottom + 34, f"{tick}", 18, text_muted, 400, "middle"))

    parts.append(line(left, chart_bottom + 8, left + chart_w, chart_bottom + 8, axis, 1.2))

    for idx, (label, value, color) in enumerate(MODELS):
        y = chart_top + idx * row_h + (row_h - bar_h) / 2
        cy = y + bar_h / 2
        bar_w = chart_w * (value / max_mae)

        parts.append(text(left - 22, cy + 7, label, 24, text_main, 600, "end"))
        parts.append(rect(left, y, chart_w, bar_h, "#f8fafc", 8))
        parts.append(rect(left, y, bar_w, bar_h, color, 8))
        parts.append(text(left + bar_w + 14, cy + 7, f"{value:.2f}", 22, text_main, 700))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
