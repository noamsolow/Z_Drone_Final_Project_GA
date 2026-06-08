from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"C:\Users\depthlev\Desktop\Z_Drone_Final_Project_GA")
OUT_DIR = ROOT / "artifacts" / "poster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 4000, 5500
OUT_PATH = OUT_DIR / "airdepth_canva_template_dataset_section_v3.png"

PAGE_BG = (241, 247, 249)
TITLE_BG = (32, 76, 96)
SECTION_BG = (253, 255, 255)
SECTION_SOFT = (231, 243, 247)
DATA_BG = (255, 255, 255)
NAVY = (22, 52, 68)
SLATE = (79, 112, 128)
MUTED = (109, 137, 149)
LINE = (156, 190, 203)
LINE_SOFT = (205, 225, 232)
TEAL = (35, 158, 173)
TEAL_SOFT = (205, 238, 242)
GREEN = (55, 156, 113)
GREEN_SOFT = (213, 242, 229)
WHITE = (255, 255, 255)
TABLE_HEADER = (216, 238, 244)
TABLE_ALT = (237, 247, 250)


def font(name="regular", size=48):
    pools = {
        "regular": [
            r"C:\Windows\Fonts\Aptos.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ],
        "bold": [
            r"C:\Windows\Fonts\Aptos-Bold.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ],
        "semibold": [
            r"C:\Windows\Fonts\Aptos-SemiBold.ttf",
            r"C:\Windows\Fonts\seguisb.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ],
    }[name]
    for path in pools:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font("bold", 104)
F_SUB = font("regular", 44)
F_META = font("regular", 28)
F_SECTION = font("bold", 48)
F_PLACE = font("semibold", 54)
F_BODY = font("regular", 32)
F_BODY_B = font("semibold", 34)
F_CALLOUT = font("bold", 82)
F_CALLOUT_LABEL = font("semibold", 30)
F_SMALL_B = font("semibold", 25)
F_TABLE = font("regular", 37)
F_TABLE_B = font("semibold", 37)
F_TABLE_HEAD = font("bold", 40)


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def center_text(draw, box, text, fnt, fill=NAVY, spacing=10):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, lh in zip(lines, heights):
        tw, _ = text_size(draw, line, fnt)
        draw.text((x1 + (x2 - x1 - tw) / 2, y), line, font=fnt, fill=fill)
        y += lh + spacing


def wrap_text(draw, text, fnt, max_width):
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if text_size(draw, trial, fnt)[0] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def paragraph(draw, xy, text, fnt, max_width, fill=NAVY, line_gap=10):
    x, y = xy
    for line in wrap_text(draw, text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += text_size(draw, line, fnt)[1] + line_gap
    return y


def rounded(draw, box, radius=46, fill=SECTION_BG, outline=LINE_SOFT, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def section(draw, box, title=None, fill=SECTION_BG):
    rounded(draw, box, radius=52, fill=fill, outline=LINE, width=3)
    if title:
        x1, y1, _, _ = box
        draw.text((x1 + 44, y1 + 36), title, font=F_SECTION, fill=NAVY)
        draw.line((x1 + 44, y1 + 104, x1 + 250, y1 + 104), fill=TEAL, width=7)


def yolo_bbox(label_path, img_w, img_h):
    parts = Path(label_path).read_text(encoding="utf-8").strip().split()
    _, cx, cy, bw, bh = [float(v) for v in parts]
    return (
        (cx - bw / 2) * img_w,
        (cy - bh / 2) * img_h,
        (cx + bw / 2) * img_w,
        (cy + bh / 2) * img_h,
    )


def drone_panel(image_path, label_path, size, tag):
    src = Image.open(image_path).convert("RGB")
    iw, ih = src.size
    bx1, by1, bx2, by2 = yolo_bbox(label_path, iw, ih)
    bw, bh = bx2 - bx1, by2 - by1
    crop_w = max(bw * 9.5, 620)
    crop_h = max(bh * 9.5, 395)
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    x1 = max(0, int(cx - crop_w / 2))
    y1 = max(0, int(cy - crop_h / 2))
    x2 = min(iw, int(cx + crop_w / 2))
    y2 = min(ih, int(cy + crop_h / 2))
    crop = src.crop((x1, y1, x2, y2))
    panel = ImageOps.fit(crop, size, method=Image.Resampling.LANCZOS)

    sx, sy = size[0] / (x2 - x1), size[1] / (y2 - y1)
    rx1, ry1 = (bx1 - x1) * sx, (by1 - y1) * sy
    rx2, ry2 = (bx2 - x1) * sx, (by2 - y1) * sy

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=32, outline=(184, 212, 222, 255), width=4)
    od.rectangle((rx1, ry1, rx2, ry2), outline=(28, 229, 213, 255), width=6)
    od.rounded_rectangle((18, 16, size[0] - 18, 66), radius=20, fill=(20, 52, 68, 218))
    tw, _ = text_size(od, tag, F_SMALL_B)
    od.text(((size[0] - tw) / 2, 27), tag, font=F_SMALL_B, fill=WHITE)
    return Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB")


def draw_table(draw, box):
    x1, y1, x2, y2 = box
    rounded(draw, box, radius=34, fill=WHITE, outline=TEAL, width=5)
    rows = [
        ("Images", "15,064 source RGB images"),
        ("Weather", "Clear sky: 7,547 (50.10%)\nLight rain: 7,517 (49.90%)"),
        ("Time", "10AM: 9,998 (66.37%)\n8PM: 5,066 (33.63%)"),
        ("Distance", "20-60 m: 6,301 (41.83%)\n70-100 m: 5,027 (33.37%)\n115-150 m: 3,736 (24.80%)"),
    ]
    header_h = 78
    draw.rounded_rectangle((x1, y1, x2, y1 + header_h), radius=34, fill=TABLE_HEADER)
    draw.rectangle((x1, y1 + 32, x2, y1 + header_h), fill=TABLE_HEADER)

    c1, c2 = x1 + 34, x1 + 335
    draw.text((c1, y1 + 20), "Split", font=F_TABLE_HEAD, fill=NAVY)
    draw.text((c2, y1 + 20), "Count and share", font=F_TABLE_HEAD, fill=NAVY)

    row_h = (y2 - y1 - header_h) / len(rows)
    y = y1 + header_h
    for i, (split, value) in enumerate(rows):
        if i % 2 == 1:
            draw.rectangle((x1, int(y), x2, int(y + row_h)), fill=TABLE_ALT)
        draw.text((c1, y + 34), split, font=F_TABLE_B, fill=NAVY)
        value_y = y + 25
        for line in value.split("\n"):
            draw.text((c2, value_y), line, font=F_TABLE, fill=NAVY)
            value_y += 54
        draw.line((x1, int(y + row_h), x2, int(y + row_h)), fill=LINE_SOFT, width=3)
        y += row_h


def main():
    img = Image.new("RGB", (W, H), PAGE_BG)
    draw = ImageDraw.Draw(img)

    margin = 60
    # Page boundary
    draw.rounded_rectangle((margin, 45, W - margin, H - 45), radius=18, fill=PAGE_BG, outline=LINE_SOFT, width=2)

    title_band = (margin, 45, W - margin, 880)
    draw.rounded_rectangle(title_band, radius=18, fill=TITLE_BG)
    draw.rectangle((margin, 820, W - margin, 880), fill=TITLE_BG)
    center_text(
        draw,
        (140, 205, W - 140, 350),
        "AirDepth: Monocular Drone Distance Estimation",
        F_TITLE,
        fill=(239, 252, 255),
    )
    center_text(draw, (140, 365, W - 140, 460), "from Relative Depth and Visual Scale Cues", F_SUB, fill=(196, 222, 231))
    center_text(draw, (140, 570, W - 140, 630), "Authors | Supervisor | Institution Logo", F_META, fill=(183, 211, 222))

    body_top, body_bot = 920, 5190
    left_x1, left_x2 = 88, 1608
    mid_x1, mid_x2 = 1640, 2832
    right_x1, right_x2 = 2864, 3912

    boxes = {
        "problem": (left_x1, 1040, left_x2, 2010),
        "challenge": (left_x1, 2040, left_x2, 2640),
        "dataset": (left_x1, 2670, left_x2, 5135),
        "pipeline": (mid_x1, 980, mid_x2, 3120),
        "progression": (mid_x1, 3160, mid_x2, 5120),
        "results": (right_x1, 1020, right_x2, 2940),
        "calibration": (right_x1, 2975, right_x2, 3790),
        "conclusions": (right_x1, 3825, right_x2, 5120),
    }

    for key, box in boxes.items():
        if key != "dataset":
            section(draw, box, fill=SECTION_SOFT)

    center_text(draw, boxes["problem"], "The Problem", F_PLACE, fill=SLATE)
    center_text(draw, boxes["challenge"], "Core Challenge", F_PLACE, fill=SLATE)
    center_text(draw, boxes["pipeline"], "Pipeline", F_PLACE, fill=SLATE)
    center_text(draw, boxes["progression"], "Research\nProgression", F_PLACE, fill=SLATE)
    center_text(draw, boxes["results"], "Key Results", F_PLACE, fill=SLATE)
    center_text(draw, boxes["calibration"], "Calibration\nStrategy", F_PLACE, fill=SLATE)
    center_text(draw, boxes["conclusions"], "Conclusions", F_PLACE, fill=SLATE)

    # Dataset section
    dataset = boxes["dataset"]
    section(draw, dataset, "Our Dataset", fill=DATA_BG)
    dx1, dy1, dx2, dy2 = dataset
    inner_x = dx1 + 46
    inner_w = dx2 - dx1 - 92

    callout = (inner_x, dy1 + 132, dx2 - 46, dy1 + 290)
    draw.rounded_rectangle(callout, radius=34, fill=TEAL_SOFT, outline=(159, 218, 226), width=2)
    draw.text((inner_x + 34, dy1 + 165), "15,064", font=F_CALLOUT, fill=NAVY)
    draw.text((inner_x + 370, dy1 + 170), "synthetic RGB images", font=F_CALLOUT_LABEL, fill=NAVY)
    draw.text((inner_x + 370, dy1 + 218), "generated in Unreal Engine", font=F_BODY, fill=MUTED)

    y = dy1 + 326
    y = paragraph(
        draw,
        (inner_x, y),
        "Ground-truth metric distance across 12 distances and 48 exact weather-time-distance strata.",
        F_BODY_B,
        inner_w,
        fill=NAVY,
        line_gap=10,
    )
    y += 4
    y = paragraph(
        draw,
        (inner_x, y),
        "Dataset preparation used automated bbox alignment, folder structuring, ray-based validity checks, and varied camera viewpoints.",
        F_BODY,
        inner_w,
        fill=MUTED,
        line_gap=10,
    )

    pill_y = y + 20
    pill_x = inner_x
    for label, bg, fg in [
        ("bbox automation", TEAL_SOFT, NAVY),
        ("ray checks", GREEN_SOFT, (28, 95, 72)),
        ("view angles", (222, 236, 249), NAVY),
    ]:
        tw, _ = text_size(draw, label, F_SMALL_B)
        draw.rounded_rectangle((pill_x, pill_y, pill_x + tw + 36, pill_y + 50), radius=25, fill=bg, outline=LINE_SOFT, width=2)
        draw.text((pill_x + 18, pill_y + 13), label, font=F_SMALL_B, fill=fg)
        pill_x += tw + 55

    panel_w, panel_h = 675, 385
    img_y = pill_y + 84
    examples = [
        (
            Path(r"C:\Users\depthlev\Desktop\droneImages\dataset\depth_20\clear_sky\10AM\HighresScreenshot00277_depth_20_clear_sky_10AM.png"),
            Path(r"C:\Users\depthlev\Desktop\droneImages\dataset\depth_20\clear_sky\10AM\HighresScreenshot00277_depth_20_clear_sky_10AM.txt"),
            "20 m | clear sky | 10AM",
            inner_x,
        ),
        (
            Path(r"C:\Users\depthlev\Desktop\droneImages\dataset\depth_50\light_rain\8PM\HighresScreenshot00001_depth_50_light_rain_8PM.png"),
            Path(r"C:\Users\depthlev\Desktop\droneImages\dataset\depth_50\light_rain\8PM\HighresScreenshot00001_depth_50_light_rain_8PM.txt"),
            "50 m | light rain | 8PM",
            inner_x + panel_w + 42,
        ),
    ]
    for image_path, label_path, tag, px in examples:
        img.paste(drone_panel(image_path, label_path, (panel_w, panel_h), tag), (px, img_y))

    table_top = img_y + panel_h + 48
    draw_table(draw, (inner_x, table_top, dx2 - 46, dy2 - 52))

    footer = (margin, 5220, W - margin, H - 45)
    draw.rounded_rectangle(footer, radius=18, fill=(225, 238, 243), outline=LINE_SOFT, width=2)
    center_text(draw, footer, "QR Code | Contact | Repository", F_META, fill=MUTED)

    img.save(OUT_PATH, quality=95)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
