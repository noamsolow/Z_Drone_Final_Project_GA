"""Build a slide-ready SVG for the Attempt 8 per-drone quadratic calibration."""

from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "artifacts" / "reports"
PLOTS = ROOT / "artifacts" / "plots"
PREDICTIONS = REPORTS / "final_calibrated_predictions.csv"
OUTPUT = PLOTS / "attempt8_quadratic_calibration_story.svg"

COEFFICIENTS = {
    "Kongsberg": (-7.702050914186054, 0.7069468495995243, -0.0017934399937629285),
    "Vestfold": (-14.409283549074965, 1.2581037096260697, -0.004219570307932922),
}
COLORS = {"Kongsberg": "#0b2463", "Vestfold": "#ff8748"}


def label(x: float, y: float, value: str, size: int, color: str, *, bold: bool = False, anchor: str = "start") -> str:
    weight = "700" if bold else "normal"
    return f'<text x="{x}" y="{y}" font-family="Aptos, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>'


def polynomial(x: float, coefficients: tuple[float, float, float]) -> float:
    intercept, linear, quadratic = coefficients
    return max(0.0, intercept + linear * x + quadratic * x * x)


def point(x: float, y: float, chart: tuple[float, float, float, float], maximum: float) -> tuple[float, float]:
    left, top, width, height = chart
    return left + x / maximum * width, top + height - y / maximum * height


def path_for_curve(coefficients: tuple[float, float, float], chart: tuple[float, float, float, float], maximum: float) -> str:
    commands: list[str] = []
    for index in range(101):
        x = maximum * index / 100
        px, py = point(x, polynomial(x, coefficients), chart, maximum)
        commands.append(("M" if index == 0 else "L") + f"{px:.1f},{py:.1f}")
    return " ".join(commands)


def main() -> None:
    with PREDICTIONS.open(newline="", encoding="utf-8") as source:
        records = list(csv.DictReader(source))
    maximum = max(max(float(row["raw_predicted_distance_m"]), float(row["calibrated_distance_m"]), float(row["true_distance_m"])) for row in records)
    maximum = max(90.0, (int(maximum / 10) + 1) * 10)
    chart = (85.0, 265.0, 735.0, 510.0)
    left, top, width, height = chart
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#ff8748"/></marker></defs>',
        '<rect width="1600" height="900" fill="#e9edf5"/>',
        label(85, 72, "SIM-TO-REAL CALIBRATION", 17, "#ff8748", bold=True),
        label(85, 123, "A quadratic correction removes systematic real-domain bias.", 39, "#0b2463", bold=True),
        label(85, 160, "The raw synthetic-trained ensemble overpredicted Nenrus distances; a small real calibration set learns the output-scale mapping.", 19, "#4b5563"),
        '<line x1="85" y1="190" x2="1515" y2="190" stroke="#d6dde9" stroke-width="2"/>',
        label(85, 224, "RAW ENSEMBLE PREDICTION → CALIBRATED METRIC DISTANCE", 16, "#ff8748", bold=True),
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="10" fill="#ffffff"/>',
    ]

    for tick in range(0, int(maximum) + 1, 10):
        x, y = point(tick, tick, chart, maximum)
        lines.extend([
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" stroke="#e6ebf3" stroke-width="1"/>',
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#e6ebf3" stroke-width="1"/>',
            label(x, top + height + 27, str(tick), 13, "#4b5563", anchor="middle"),
            label(left - 12, y + 5, str(tick), 13, "#4b5563", anchor="end"),
        ])
    x0, y0 = point(0, 0, chart, maximum)
    x1, y1 = point(maximum, maximum, chart, maximum)
    lines.extend([
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#8a96aa" stroke-width="3" stroke-dasharray="10 8"/>',
        label(left + width - 8, top + 27, "ideal: corrected = raw", 14, "#6b7280", anchor="end"),
        label(left + width / 2, top + height + 60, "Raw ensemble prediction (m)", 17, "#0b2463", bold=True, anchor="middle"),
        f'<text x="28" y="{top + height / 2}" font-family="Aptos, Arial, sans-serif" font-size="17" font-weight="700" fill="#0b2463" transform="rotate(-90 28 {top + height / 2})" text-anchor="middle">Calibrated metric distance (m)</text>',
    ])

    # Real predictions demonstrate the actual mapping; the lines are the exact selected quadratic functions.
    for row in records:
        drone_type = row["drone_type"]
        px, py = point(float(row["raw_predicted_distance_m"]), float(row["calibrated_distance_m"]), chart, maximum)
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="{COLORS[drone_type]}" fill-opacity="0.30"/>')
    for drone_type, coefficients in COEFFICIENTS.items():
        lines.append(f'<path d="{path_for_curve(coefficients, chart, maximum)}" fill="none" stroke="{COLORS[drone_type]}" stroke-width="5"/>')

    lines.extend([
        '<rect x="875" y="250" width="640" height="170" rx="16" fill="#ffffff"/>',
        label(905, 291, "Selected: per-drone quadratic calibration", 24, "#0b2463", bold=True),
        label(905, 328, "Kongsberg: z = -7.70 + 0.7069r - 0.001793r²", 18, "#0b2463", bold=True),
        label(905, 365, "Vestfold: z = -14.41 + 1.2581r - 0.004220r²", 18, "#ff8748", bold=True),
        label(905, 399, "r = raw ensemble prediction, z = corrected metric distance", 15, "#4b5563"),
        '<rect x="875" y="450" width="292" height="155" rx="16" fill="#0b2463"/>',
        label(1021, 489, "Before calibration", 16, "#ffffff", bold=True, anchor="middle"),
        label(1021, 543, "23.80 m", 40, "#ffffff", bold=True, anchor="middle"),
        label(1021, 575, "raw real-domain MAE", 15, "#ffffff", anchor="middle"),
        '<rect x="1195" y="450" width="320" height="155" rx="16" fill="#ff8748"/>',
        label(1355, 489, "After 20% calibration", 16, "#ffffff", bold=True, anchor="middle"),
        label(1355, 543, "3.28 m", 40, "#ffffff", bold=True, anchor="middle"),
        label(1355, 575, "held-out MAE (repeated splits)", 15, "#ffffff", anchor="middle"),
        '<path d="M1169 527 L1188 527" stroke="#ff8748" stroke-width="5" marker-end="url(#arrow)"/>',
        '<rect x="875" y="640" width="640" height="135" rx="16" fill="#ffffff"/>',
        label(905, 681, "Why it works", 22, "#0b2463", bold=True),
        label(905, 713, "The real error was systematic overprediction (99.8%), not random noise.", 17, "#111827"),
        label(905, 742, "The quadratic mapping corrects scale bias without retraining the RF-XGBoost ensemble.", 17, "#111827"),
        '<circle cx="103" cy="808" r="8" fill="#0b2463"/><text x="120" y="814" font-family="Aptos, Arial, sans-serif" font-size="15" fill="#4b5563">Kongsberg quadratic</text>',
        '<circle cx="330" cy="808" r="8" fill="#ff8748"/><text x="347" y="814" font-family="Aptos, Arial, sans-serif" font-size="15" fill="#4b5563">Vestfold quadratic</text>',
        '<line x1="575" y1="808" x2="610" y2="808" stroke="#8a96aa" stroke-width="3" stroke-dasharray="8 6"/><text x="620" y="814" font-family="Aptos, Arial, sans-serif" font-size="15" fill="#4b5563">identity reference</text>',
        label(85, 858, "Curves: final functions fit on all 489 Nenrus rows. Performance claim: repeated split evaluation with 20% calibration data and the remaining rows held out.", 14, "#4b5563"),
        '</svg>',
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
