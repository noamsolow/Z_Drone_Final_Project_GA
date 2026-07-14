"""Create a slide-ready SVG explaining the exact Attempt 8 ensemble inputs."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "plots" / "attempt8_ensemble_feature_pipeline.svg"


def text(x: float, y: float, value: str, size: int, color: str, *, weight: str = "normal", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Aptos, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def card(x: int, y: int, w: int, h: int, fill: str, title: str, count: str, lines: list[str]) -> list[str]:
    output = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}"/>',
        text(x + 24, y + 38, title, 22, "#ffffff", weight="700"),
        text(x + w - 24, y + 38, count, 16, "#ffffff", weight="700", anchor="end"),
    ]
    for index, line in enumerate(lines):
        output.append(text(x + 24, y + 73 + index * 25, line, 15, "#ffffff"))
    return output


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#ff8748"/></marker></defs>',
        '<rect width="1600" height="900" fill="#e9edf5"/>',
        text(80, 72, "ATTEMPT 8 · FINAL MODEL", 17, "#ff8748", weight="700"),
        text(80, 122, "From RGB Features to Metric Drone Distance", 42, "#0b2463", weight="700"),
        text(80, 158, "The final RF-XGBoost ensemble converts target-centric cues into estimated Z-distance in meters.", 20, "#4b5563"),
        '<line x1="80" y1="186" x2="1520" y2="186" stroke="#d6dde9" stroke-width="2"/>',
        text(80, 223, "1. ONE RGB IMAGE", 16, "#ff8748", weight="700"),
        '<rect x="80" y="245" width="255" height="132" rx="16" fill="#ffffff"/>',
        '<rect x="100" y="265" width="215" height="72" rx="8" fill="#78a8d6"/>',
        '<rect x="190" y="286" width="45" height="31" fill="none" stroke="#ff8748" stroke-width="4"/>',
        text(207, 360, "drone detection + depth map", 15, "#0b2463", weight="700", anchor="middle"),
        '<path d="M350 310 L425 310" stroke="#ff8748" stroke-width="5" marker-end="url(#arrow)"/>',
        text(450, 223, "2. FEATURE EXTRACTION", 16, "#ff8748", weight="700"),
    ]

    lines += card(425, 245, 400, 160, "#0b2463", "Target depth cue", "1 feature", [
        "inner-50% median relative depth",
        "inside the detected drone box",
        "aggregated over jittered boxes",
    ])
    lines += card(425, 425, 400, 255, "#2e5e9e", "Bounding-box geometry", "16 features", [
        "width, height, normalized width, normalized height",
        "area ratio, aspect ratio, center X, center Y",
        "each represented by jitter median and jitter std",
        "= 8 geometry signals x 2 robust statistics",
    ])
    lines += card(425, 700, 400, 125, "#173b7a", "Environment metadata", "4 features", [
        "weather: clear sky / light rain (one-hot)",
        "time: 10 AM / 8 PM (one-hot)",
    ])
    lines += [
        '<path d="M845 325 L905 325" stroke="#ff8748" stroke-width="5" marker-end="url(#arrow)"/>',
        '<path d="M845 552 L905 552" stroke="#ff8748" stroke-width="5" marker-end="url(#arrow)"/>',
        '<path d="M845 762 C880 762, 875 595, 905 595" fill="none" stroke="#ff8748" stroke-width="5" marker-end="url(#arrow)"/>',
        '<rect x="915" y="402" width="235" height="210" rx="18" fill="#ffffff" stroke="#d6dde9" stroke-width="2"/>',
        text(1032, 447, "21-DIMENSION", 16, "#ff8748", weight="700", anchor="middle"),
        text(1032, 488, "FEATURE VECTOR", 25, "#0b2463", weight="700", anchor="middle"),
        text(1032, 525, "1 depth + 16 geometry", 15, "#4b5563", anchor="middle"),
        text(1032, 550, "+ 4 metadata features", 15, "#4b5563", anchor="middle"),
        text(1032, 578, "per detected drone", 14, "#4b5563", anchor="middle"),
        '<path d="M1165 507 L1220 507" stroke="#ff8748" stroke-width="5" marker-end="url(#arrow)"/>',
        text(1215, 223, "3. ENSEMBLE REGRESSION", 16, "#ff8748", weight="700"),
        '<rect x="1225" y="245" width="285" height="165" rx="16" fill="#0b2463"/>',
        text(1368, 288, "Random Forest", 24, "#ffffff", weight="700", anchor="middle"),
        text(1368, 321, "jitter-aggregated features", 15, "#ffffff", anchor="middle"),
        text(1368, 370, "17% blend weight", 20, "#ff8748", weight="700", anchor="middle"),
        '<rect x="1225" y="445" width="285" height="165" rx="16" fill="#2e5e9e"/>',
        text(1368, 488, "XGBoost", 24, "#ffffff", weight="700", anchor="middle"),
        text(1368, 521, "jitter-aggregated features", 15, "#ffffff", anchor="middle"),
        text(1368, 570, "83% blend weight", 20, "#ff8748", weight="700", anchor="middle"),
        '<path d="M1368 625 L1368 690" stroke="#ff8748" stroke-width="5" marker-end="url(#arrow)"/>',
        '<rect x="1195" y="710" width="345" height="115" rx="16" fill="#ff8748"/>',
        text(1368, 752, "OUTPUT", 16, "#ffffff", weight="700", anchor="middle"),
        text(1368, 790, "Metric Z-distance (m)", 28, "#ffffff", weight="700", anchor="middle"),
        text(80, 865, "Attempt 8 final ensemble: RF weight 0.17 + XGBoost weight 0.83. The depth model supplies relative cues; regression learns metric scale.", 15, "#4b5563"),
        '</svg>',
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
