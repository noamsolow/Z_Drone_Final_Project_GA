from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "poster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 4724, 6496  # 80:110 cm ratio, good-quality Canva import draft.
MARGIN = 170
GAP = 58
COL_W = int((W - 2 * MARGIN - 2 * GAP) / 3)

NAVY = "#0B1F3A"
BLUE = "#1E66D0"
TEAL = "#0E8E9D"
GREEN = "#16845B"
ORANGE = "#D7632B"
RED = "#B54132"
INK = "#1E293B"
MUTED = "#5C6B7A"
LIGHT = "#F5F8FB"
LINE = "#D9E3EC"
WHITE = "#FFFFFF"

FONT_REG = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_ITALIC = "C:/Windows/Fonts/segoeuii.ttf"


def font(size, bold=False, italic=False):
    path = FONT_BOLD if bold else FONT_ITALIC if italic else FONT_REG
    return ImageFont.truetype(path, size=size)


F_TITLE = font(105, True)
F_SUB = font(42)
F_META = font(29)
F_SEC = font(42, True)
F_BODY = font(28)
F_SMALL = font(23)
F_TINY = font(19)
F_NUM = font(74, True)
F_NUM_BIG = font(92, True)
F_LABEL = font(26, True)


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw, text, fnt, max_w):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if text_size(draw, candidate, fnt)[0] <= max_w:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(draw, xy, text, fnt, fill=INK, max_w=500, line_gap=10):
    x, y = xy
    for para in text.split("\n"):
        if not para.strip():
            y += fnt.size
            continue
        for line in wrap_text(draw, para.strip(), fnt, max_w):
            draw.text((x, y), line, font=fnt, fill=fill)
            y += fnt.size + line_gap
        y += line_gap
    return y


def panel(draw, x, y, w, h, title, accent=BLUE):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=26, fill=WHITE, outline=LINE, width=3)
    draw.rounded_rectangle((x, y, x + w, y + 18), radius=20, fill=accent)
    draw.text((x + 32, y + 36), title, font=F_SEC, fill=NAVY)
    return x + 32, y + 98, w - 64


def bullet_list(draw, x, y, items, max_w, fnt=F_BODY, fill=INK, bullet_fill=TEAL, gap=12):
    for item in items:
        draw.ellipse((x, y + 10, x + 13, y + 23), fill=bullet_fill)
        y = draw_wrapped(draw, (x + 28, y), item, fnt, fill=fill, max_w=max_w - 28, line_gap=8)
        y += gap
    return y


def metric_card(draw, x, y, w, h, number, label, color, sub=None, big=False):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=24, fill="#FFFFFF", outline=color, width=4)
    draw.text((x + 26, y + 24), number, font=F_NUM_BIG if big else F_NUM, fill=color)
    draw_wrapped(draw, (x + 28, y + 116), label, F_LABEL, fill=NAVY, max_w=w - 56, line_gap=5)
    if sub:
        draw_wrapped(draw, (x + 28, y + h - 58), sub, F_SMALL, fill=MUTED, max_w=w - 56, line_gap=4)


def load_cover(path, size):
    img = Image.open(ROOT / path).convert("RGB")
    return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def load_contain(path, size, bg=WHITE):
    img = Image.open(ROOT / path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def draw_bar_chart(draw, x, y, w, h, labels, values, colors, title, unit="m MAE"):
    draw.text((x, y), title, font=F_LABEL, fill=NAVY)
    plot_y = y + 55
    max_v = max(values)
    bar_gap = 22
    bar_h = int((h - 80 - (len(values) - 1) * bar_gap) / len(values))
    label_w = 330
    for i, (lab, val, col) in enumerate(zip(labels, values, colors)):
        by = plot_y + i * (bar_h + bar_gap)
        draw.text((x, by + bar_h // 2 - 15), lab, font=F_SMALL, fill=INK)
        bx = x + label_w
        bw = int((w - label_w - 130) * val / max_v)
        draw.rounded_rectangle((bx, by, bx + bw, by + bar_h), radius=14, fill=col)
        draw.text((bx + bw + 18, by + bar_h // 2 - 16), f"{val:.2f} {unit}", font=F_SMALL, fill=NAVY)


def draw_pipeline(draw, x, y, w):
    steps = [
        ("RGB\nImage", BLUE),
        ("Drone\nBBox", TEAL),
        ("Relative\nDepth", TEAL),
        ("Feature\nVector", NAVY),
        ("RF + XGB\nEnsemble", BLUE),
        ("Metric\nz_hat", GREEN),
    ]
    box_w = int((w - 5 * 32) / 6)
    box_h = 118
    for i, (label, col) in enumerate(steps):
        bx = x + i * (box_w + 32)
        draw.rounded_rectangle((bx, y, bx + box_w, y + box_h), radius=22, fill="#F8FBFD", outline=col, width=4)
        parts = label.split("\n")
        total_h = len(parts) * 28
        ty = y + (box_h - total_h) // 2 - 2
        for p in parts:
            tw = text_size(draw, p, F_SMALL)[0]
            draw.text((bx + (box_w - tw) / 2, ty), p, font=F_SMALL, fill=NAVY)
            ty += 32
        if i < len(steps) - 1:
            ax = bx + box_w + 7
            ay = y + box_h // 2
            draw.line((ax, ay, ax + 18, ay), fill=col, width=5)
            draw.polygon([(ax + 18, ay - 10), (ax + 34, ay), (ax + 18, ay + 10)], fill=col)
    return y + box_h


def draw_timeline(draw, x, y, w):
    rows = [
        ("Attempt 1", "Linear baselines", "14.50 m"),
        ("Attempt 2", "Local depth representation", "20.92 m CV"),
        ("Attempt 3", "Noisy-bbox aggregation RF", "5.21 m"),
        ("Attempt 4", "Detector-like jitter + RF/XGB", "7.63 m"),
        ("Attempt 5", "Nenrus calibration", "3.07 m holdout"),
    ]
    line_x = x + 32
    draw.line((line_x, y + 20, line_x, y + 410), fill=LINE, width=6)
    for i, (name, desc, result) in enumerate(rows):
        cy = y + i * 88 + 20
        color = [BLUE, TEAL, BLUE, NAVY, GREEN][i]
        draw.ellipse((line_x - 15, cy - 15, line_x + 15, cy + 15), fill=color)
        draw.text((x + 75, cy - 26), name, font=F_LABEL, fill=NAVY)
        draw.text((x + 245, cy - 24), desc, font=F_SMALL, fill=INK)
        tw = text_size(draw, result, F_LABEL)[0]
        draw.rounded_rectangle((x + w - tw - 70, cy - 33, x + w - 28, cy + 18), radius=16, fill="#EEF7F4")
        draw.text((x + w - tw - 50, cy - 24), result, font=F_LABEL, fill=GREEN if i == 4 else BLUE)


def paste_caption(base, draw, img, x, y, w, h, caption):
    base.paste(img, (x, y))
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, outline=LINE, width=3)
    draw.rectangle((x, y + h - 38, x + w, y + h), fill=(11, 31, 58, 220))
    draw.text((x + 18, y + h - 32), caption, font=F_TINY, fill=WHITE)


def main():
    base = Image.new("RGB", (W, H), LIGHT)
    draw = ImageDraw.Draw(base)

    # Header
    draw.rectangle((0, 0, W, 610), fill=NAVY)
    draw.rectangle((0, 570, W, 610), fill=TEAL)
    draw.text((MARGIN, 96), "AirDepth", font=font(128, True), fill=WHITE)
    draw.text((MARGIN, 238), "Monocular Drone Distance Estimation from Relative Depth and Visual Scale Cues", font=F_TITLE, fill=WHITE)
    draw_wrapped(
        draw,
        (MARGIN, 372),
        "Estimating metric drone distance from a single RGB image using monocular depth, bounding-box geometry, ensemble regression, and real-domain calibration.",
        F_SUB,
        fill="#DDEAF6",
        max_w=W - 2 * MARGIN - 640,
        line_gap=8,
    )
    draw.text((MARGIN, 520), "Authors: [Names]   |   Supervisor: [Name]   |   Institution: [Logo]", font=F_META, fill="#DDEAF6")
    # QR placeholder
    qr_x, qr_y = W - MARGIN - 300, 118
    draw.rounded_rectangle((qr_x, qr_y, qr_x + 300, qr_y + 300), radius=22, fill=WHITE)
    for i in range(0, 300, 40):
        draw.line((qr_x + i, qr_y, qr_x + i, qr_y + 300), fill="#DDEAF6", width=2)
        draw.line((qr_x, qr_y + i, qr_x + 300, qr_y + i), fill="#DDEAF6", width=2)
    draw.text((qr_x + 58, qr_y + 126), "QR / Repo", font=F_LABEL, fill=NAVY)

    y0 = 690
    col_x = [MARGIN, MARGIN + COL_W + GAP, MARGIN + 2 * (COL_W + GAP)]

    # Section 1
    x, y, cw = panel(draw, col_x[0], y0, COL_W, 835, "1. The Problem", BLUE)
    draw_wrapped(
        draw,
        (x, y),
        "Knowing how far a drone really is from a single camera frame is a metric localization problem, not just a detection problem.",
        F_BODY,
        max_w=cw,
    )
    bullet_list(
        draw,
        x,
        y + 145,
        [
            "Input: one RGB image and a detected drone bounding box.",
            "Output: estimated optical-axis distance z_hat in meters.",
            "No stereo, LiDAR, radar, GPS, RGB-D sensing, or cooperative target signal.",
        ],
        cw,
    )
    img1 = load_cover(
        "attempts/attempt2/test_single_image_context_window_maps/outputs_HighresScreenshot00309_depth_30_clear_sky_10AM/HighresScreenshot00309_depth_30_clear_sky_10AM_original_with_bbox.png",
        (cw, 275),
    )
    paste_caption(base, draw, img1, x, y + 500, cw, 275, "Drone image with target bounding box")

    # Section 2
    x, y, cw = panel(draw, col_x[1], y0, COL_W, 835, "2. Core Challenge", ORANGE)
    draw_wrapped(
        draw,
        (x, y),
        "Monocular depth models produce relative depth maps. Their values are useful visual cues, but they are not calibrated in meters.",
        F_BODY,
        max_w=cw,
    )
    draw_bar_chart(
        draw,
        x,
        y + 175,
        cw,
        370,
        ["Scale-only", "Depth-only linear", "Depth + geometry + metadata"],
        [47.75, 32.11, 14.50],
        [RED, ORANGE, TEAL],
        "Attempt 1 baseline error",
    )
    draw_wrapped(
        draw,
        (x, y + 610),
        "Takeaway: metric distance cannot be recovered reliably from a single raw relative-depth value. Visual scale and context are required.",
        F_BODY,
        fill=NAVY,
        max_w=cw,
    )

    # Section 3
    x, y, cw = panel(draw, col_x[2], y0, COL_W, 835, "3. Dataset", TEAL)
    draw_wrapped(
        draw,
        (x, y),
        "Synthetic controlled scenes provide ground-truth metric distance and balanced conditions for supervised calibration.",
        F_BODY,
        max_w=cw,
    )
    stats = [
        ("14.8k-15.1k", "images"),
        ("20-150 m", "distance range"),
        ("12", "distance values"),
        ("2 x 2", "weather / time"),
        ("48", "exact strata"),
    ]
    sx, sy = x, y + 160
    for i, (num, lab) in enumerate(stats):
        bx = sx + (i % 2) * (cw // 2 + 14)
        by = sy + (i // 2) * 142
        bw = cw // 2 - 14 if i < 4 else cw
        draw.rounded_rectangle((bx, by, bx + bw, by + 112), radius=20, fill="#F8FBFD", outline=LINE, width=2)
        draw.text((bx + 22, by + 18), num, font=F_LABEL, fill=BLUE)
        draw.text((bx + 22, by + 58), lab, font=F_SMALL, fill=MUTED)
    draw_wrapped(
        draw,
        (x, y + 600),
        "Controlled labels let us measure MAE, relative error, and behavior by distance, weather, and time of day.",
        F_BODY,
        max_w=cw,
    )

    # Section 4 - wide center pipeline
    x4 = MARGIN
    y4 = y0 + 900
    w4 = W - 2 * MARGIN
    px, py, pcw = panel(draw, x4, y4, w4, 840, "4. Pipeline Overview", BLUE)
    draw_pipeline(draw, px, py, pcw)
    draw_wrapped(
        draw,
        (px, py + 155),
        "AirDepth combines relative-depth cues, bounding-box geometry, scene metadata, and supervised regression. Camera intrinsics can optionally back-project the metric Z estimate into a camera-centric 3D point.",
        F_BODY,
        max_w=pcw,
    )
    img2 = load_cover(
        "attempts/attempt2/test_single_image_context_window_maps/outputs_HighresScreenshot00309_depth_30_clear_sky_10AM/HighresScreenshot00309_depth_30_clear_sky_10AM_original_with_bbox.png",
        (720, 340),
    )
    img3 = load_cover(
        "attempts/attempt2/test_single_image_context_window_maps/outputs_HighresScreenshot00309_depth_30_clear_sky_10AM/HighresScreenshot00309_depth_30_clear_sky_10AM_bbox_only_depth_map_with_boxes.png",
        (720, 340),
    )
    paste_caption(base, draw, img2, px, py + 345, 720, 340, "RGB + bounding box")
    paste_caption(base, draw, img3, px + 760, py + 345, 720, 340, "Relative depth inside drone crop")
    fx = px + 1560
    draw.text((fx, py + 355), "Feature Vector", font=F_LABEL, fill=NAVY)
    bullet_list(
        draw,
        fx,
        py + 410,
        [
            "Relative depth inside the target region",
            "BBox width, height, area, aspect ratio, center",
            "Weather and time metadata when available",
            "RF + XGBoost ensemble prediction",
        ],
        pcw - 1560,
        fnt=F_SMALL,
    )

    # Section 5
    y5 = y4 + 900
    x, y, cw = panel(draw, col_x[0], y5, COL_W, 1125, "5. Research Progression", NAVY)
    draw_timeline(draw, x, y + 10, cw)
    draw_wrapped(
        draw,
        (x, y + 470),
        "Progression moved from simple calibration toward detector-like robustness and domain-aware calibration. Each stage isolated one modeling assumption before changing the next one.",
        F_BODY,
        max_w=cw,
    )
    draw.rounded_rectangle((x, y + 690, x + cw, y + 955), radius=22, fill="#F8FBFD", outline=LINE, width=2)
    draw.text((x + 28, y + 718), "Most important modeling lesson", font=F_LABEL, fill=NAVY)
    draw_wrapped(
        draw,
        (x + 28, y + 770),
        "The final system is geometry-first and depth-assisted: image scale is the strongest signal, while relative depth improves it as a correction cue.",
        F_BODY,
        fill=INK,
        max_w=cw - 56,
    )

    # Section 6
    x, y, cw = panel(draw, col_x[1], y5, COL_W, 1125, "6. Key Results", GREEN)
    metric_card(draw, x, y, cw, 205, "7.63 m", "Synthetic RF/XGBoost ensemble MAE", BLUE, "9.90% relative error", big=True)
    metric_card(draw, x, y + 240, cw, 205, "26.11 m", "Raw real-domain Nenrus MAE", RED, "100% overprediction", big=True)
    metric_card(draw, x, y + 480, cw, 205, "3.03 m", "Calibrated real-domain Nenrus MAE", GREEN, "97.14% within 10 m; R2 = 0.956", big=True)
    draw_wrapped(
        draw,
        (x, y + 740),
        "Interpretation: strong in-domain performance did not transfer directly to real-drone imagery. Calibration turned a systematic scale error into usable metric predictions.",
        F_BODY,
        max_w=cw,
    )

    # Section 7
    x, y, cw = panel(draw, col_x[2], y5, COL_W, 1125, "7. Calibration Strategy", GREEN)
    draw_wrapped(
        draw,
        (x, y),
        "The frozen synthetic-trained ensemble overpredicted every Nenrus image because real-drone bounding boxes were smaller at the same true distance.",
        F_BODY,
        max_w=cw,
    )
    draw_bar_chart(
        draw,
        x,
        y + 190,
        cw,
        290,
        ["Raw Nenrus", "20% calibration holdout", "Full-data calibration fit"],
        [26.11, 3.07, 3.03],
        [RED, GREEN, GREEN],
        "Real-domain calibration effect",
    )
    bullet_list(
        draw,
        x,
        y + 560,
        [
            "Per-drone quadratic calibration maps raw ensemble output to metric distance.",
            "It does not retrain RF or XGBoost.",
            "Honest 20% labelled calibration split: 3.07 m MAE, 96.54% within 10 m.",
            "Full-data fit: 3.03 m MAE, 97.14% within 10 m.",
        ],
        cw,
        fnt=F_SMALL,
    )

    # Section 8 and bottom visual
    y8 = y5 + 1190
    h8 = H - y8 - 255
    x, y, cw = panel(draw, MARGIN, y8, W - 2 * MARGIN, h8, "8. Conclusions", NAVY)
    conclusions = [
        "Monocular relative depth alone is not enough for metric drone distance.",
        "Bounding-box geometry is the strongest distance cue; relative depth improves it as a correction signal.",
        "Strong synthetic performance does not guarantee real-world transfer because apparent object scale changes across domains.",
        "A lightweight calibration layer can convert a strong synthetic-trained estimator into a usable real-domain distance predictor.",
    ]
    bullet_list(draw, x, y, conclusions, cw // 2 - 40, fnt=F_BODY, bullet_fill=GREEN)
    # Simplified final chart
    chart_x = x + cw // 2 + 80
    draw.text((chart_x, y), "Poster-level result summary", font=F_LABEL, fill=NAVY)
    vals = [7.63, 26.11, 3.03]
    labs = ["Synthetic\nensemble", "Raw real\nNenrus", "Calibrated\nNenrus"]
    cols = [BLUE, RED, GREEN]
    max_v = max(vals)
    bx0 = chart_x
    by0 = y + 90
    bw = 250
    scale_h = 470
    for i, (lab, val, col) in enumerate(zip(labs, vals, cols)):
        bx = bx0 + i * 330
        bh = int(scale_h * val / max_v)
        draw.rounded_rectangle((bx, by0 + scale_h - bh, bx + bw, by0 + scale_h), radius=22, fill=col)
        draw.text((bx + 26, by0 + scale_h - bh - 54), f"{val:.2f} m", font=F_LABEL, fill=col)
        ty = by0 + scale_h + 26
        for part in lab.split("\n"):
            tw = text_size(draw, part, F_SMALL)[0]
            draw.text((bx + (bw - tw) / 2, ty), part, font=F_SMALL, fill=NAVY)
            ty += 30
    draw.text((chart_x, y + 710), "MAE lower is better. Calibration result shown with real-domain context.", font=F_SMALL, fill=MUTED)

    evidence_y = y + 800
    draw.line((x, evidence_y - 35, x + cw, evidence_y - 35), fill=LINE, width=3)
    draw.text((x, evidence_y), "Real-domain calibration evidence", font=F_LABEL, fill=NAVY)
    draw_wrapped(
        draw,
        (x, evidence_y + 48),
        "Raw Nenrus predictions were systematically too large. After per-drone calibration, predictions align with the true metric scale while preserving the frozen RF/XGBoost model.",
        F_SMALL,
        fill=INK,
        max_w=cw // 2 - 60,
        line_gap=5,
    )
    evidence_img = load_cover(
        "attempts/attempt5/studies/study04/artifacts/plots/raw_vs_calibrated_predicted_vs_true.png",
        (cw // 2 - 40, 520),
    )
    paste_caption(base, draw, evidence_img, x + cw // 2 + 40, evidence_y, cw // 2 - 40, 520, "Raw vs calibrated predictions on Nenrus")

    bottom_note_y = evidence_y + 610
    draw.rounded_rectangle((x, bottom_note_y, x + cw, bottom_note_y + 150), radius=24, fill="#F8FBFD", outline=LINE, width=2)
    draw.text((x + 34, bottom_note_y + 30), "Methodological note", font=F_LABEL, fill=NAVY)
    draw_wrapped(
        draw,
        (x + 315, bottom_note_y + 30),
        "3.07 m MAE is the repeated holdout result using 20% labelled real-domain calibration data. 3.03 m MAE is the final full-data calibration fit.",
        F_BODY,
        fill=INK,
        max_w=cw - 360,
        line_gap=6,
    )

    support_y = bottom_note_y + 205
    support_w = (cw - 55) // 2
    support_h = 510
    img4 = load_contain(
        "attempts/attempt5/studies/study04/artifacts/plots/mae_before_after_by_distance.png",
        (support_w, support_h),
    )
    img5 = load_contain(
        "attempts/attempt5/studies/study04/artifacts/plots/tolerance_rates_before_after.png",
        (support_w, support_h),
    )
    paste_caption(base, draw, img4, x, support_y, support_w, support_h, "MAE before and after calibration by distance")
    paste_caption(base, draw, img5, x + support_w + 55, support_y, support_w, support_h, "Tolerance rates before and after calibration")

    # Footer
    draw.rectangle((0, H - 145, W, H), fill=NAVY)
    draw.text((MARGIN, H - 100), "Computer Vision Final Project  |  Monocular Drone 3D Localization  |  attempt6 excluded from this poster", font=F_SMALL, fill="#DDEAF6")
    draw.text((W - MARGIN - 630, H - 100), "Contact: [email]  |  Code/Report: [QR]", font=F_SMALL, fill="#DDEAF6")

    out = OUT_DIR / "airdepth_poster_80x110cm_draft.png"
    base.save(out, dpi=(150, 150), optimize=True)
    print(out)


if __name__ == "__main__":
    main()
