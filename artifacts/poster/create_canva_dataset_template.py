from pathlib import Path
import csv

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"C:\Users\depthlev\Desktop\Z_Drone_Final_Project_GA")
OUT_DIR = ROOT / "artifacts" / "poster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 3200, 4400
OUT_PATH = OUT_DIR / "airdepth_canva_template_dataset_section.png"

BG = (20, 101, 126)
BG_DARK = (14, 78, 98)
HEADER = (229, 250, 252)
TEXT = (245, 252, 255)
MUTED = (196, 229, 235)
INK = (9, 39, 50)
LINE = (7, 43, 54)
TEAL = (42, 196, 204)
GREEN = (80, 202, 145)
CARD = (234, 250, 251)
CARD_2 = (214, 242, 245)
WHITE = (255, 255, 255)


def font(name="regular", size=48):
    candidates = {
        "regular": [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ],
        "bold": [
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ],
        "semibold": [
            r"C:\Windows\Fonts\seguisb.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ],
    }[name]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font("bold", 90)
F_SUB = font("regular", 38)
F_SECTION = font("bold", 50)
F_PLACE = font("semibold", 52)
F_BODY = font("regular", 31)
F_BODY_B = font("semibold", 31)
F_SMALL = font("regular", 25)
F_SMALL_B = font("semibold", 25)
F_TABLE = font("regular", 23)
F_TABLE_B = font("semibold", 23)


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def center_text(draw, box, text, fnt, fill=TEXT, spacing=8):
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
    lines = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if text_size(draw, trial, fnt)[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def paragraph(draw, xy, text, fnt, max_width, fill=TEXT, line_gap=9):
    x, y = xy
    for line in wrap_text(draw, text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += text_size(draw, line, fnt)[1] + line_gap
    return y


def section(draw, box, title=None, radius=70, fill=BG, outline=LINE, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    if title:
        x1, y1, x2, _ = box
        draw.text((x1 + 44, y1 + 34), title, font=F_SECTION, fill=HEADER)


def yolo_bbox(label_path, img_w, img_h):
    with open(label_path, "r", encoding="utf-8") as f:
        line = f.readline().strip().split()
    _, cx, cy, bw, bh = [float(v) for v in line]
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return x1, y1, x2, y2


def make_drone_panel(image_path, label_path, size, tag):
    src = Image.open(image_path).convert("RGB")
    iw, ih = src.size
    bx1, by1, bx2, by2 = yolo_bbox(label_path, iw, ih)
    bw, bh = bx2 - bx1, by2 - by1
    crop_w = max(bw * 9, 520)
    crop_h = max(bh * 9, 330)
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    x1 = max(0, int(cx - crop_w / 2))
    y1 = max(0, int(cy - crop_h / 2))
    x2 = min(iw, int(cx + crop_w / 2))
    y2 = min(ih, int(cy + crop_h / 2))
    crop = src.crop((x1, y1, x2, y2))
    panel = ImageOps.fit(crop, size, method=Image.Resampling.LANCZOS)

    sx = size[0] / (x2 - x1)
    sy = size[1] / (y2 - y1)
    rx1 = (bx1 - x1) * sx
    ry1 = (by1 - y1) * sy
    rx2 = (bx2 - x1) * sx
    ry2 = (by2 - y1) * sy

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=34, outline=(255, 255, 255, 175), width=3)
    od.rectangle((rx1, ry1, rx2, ry2), outline=(63, 242, 222, 255), width=5)
    od.rounded_rectangle((18, 16, size[0] - 18, 62), radius=18, fill=(3, 38, 48, 205))
    tw, _ = text_size(od, tag, F_SMALL_B)
    od.text(((size[0] - tw) / 2, 24), tag, font=F_SMALL_B, fill=WHITE)
    return Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB")


def draw_table(draw, box):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=26, fill=CARD, outline=(154, 215, 222), width=2)
    rows = [
        ("Source images", "15,064", "100%"),
        ("Clear sky", "7,547", "50.10%"),
        ("Light rain", "7,517", "49.90%"),
        ("10AM", "9,998", "66.37%"),
        ("8PM", "5,066", "33.63%"),
        ("Near 20-60m", "6,301", "41.83%"),
        ("Mid 70-100m", "5,027", "33.37%"),
        ("Far 115-150m", "3,736", "24.80%"),
    ]
    header_h = 48
    draw.rounded_rectangle((x1, y1, x2, y1 + header_h), radius=26, fill=(184, 233, 239))
    draw.rectangle((x1, y1 + 24, x2, y1 + header_h), fill=(184, 233, 239))
    col1 = x1 + 24
    col2 = x1 + 440
    col3 = x1 + 625
    draw.text((col1, y1 + 13), "Split", font=F_TABLE_B, fill=INK)
    draw.text((col2, y1 + 13), "Count", font=F_TABLE_B, fill=INK)
    draw.text((col3, y1 + 13), "%", font=F_TABLE_B, fill=INK)
    row_h = (y2 - y1 - header_h) / len(rows)
    y = y1 + header_h
    for i, (split, count, pct) in enumerate(rows):
        if i % 2 == 1:
            draw.rectangle((x1, int(y), x2, int(y + row_h)), fill=CARD_2)
        draw.text((col1, y + 10), split, font=F_TABLE, fill=INK)
        draw.text((col2, y + 10), count, font=F_TABLE_B, fill=INK)
        draw.text((col3, y + 10), pct, font=F_TABLE_B, fill=INK)
        y += row_h


def main():
    img = Image.new("RGB", (W, H), (247, 250, 250))
    draw = ImageDraw.Draw(img)

    outer = (48, 36, W - 48, H - 36)
    draw.rectangle(outer, fill=BG, outline=LINE, width=4)

    # Title band
    title_band = (48, 36, W - 48, 775)
    draw.rectangle(title_band, fill=BG_DARK, outline=LINE, width=3)
    title = "AirDepth: Monocular Drone Distance Estimation"
    subtitle = "from Relative Depth and Visual Scale Cues"
    center_text(draw, (120, 170, W - 120, 415), title, F_TITLE, HEADER)
    center_text(draw, (120, 405, W - 120, 505), subtitle, F_SUB, MUTED)
    center_text(draw, (120, 560, W - 120, 660), "Authors | Supervisor | Institution Logo", F_SMALL, (170, 220, 228))

    body_top, body_bot = 775, 4190
    left = (48, body_top, 1288, body_bot)
    middle = (1288, body_top, 2288, body_bot)
    right = (2288, body_top, W - 48, body_bot)
    for x in [1288, 2288]:
        draw.line((x, body_top, x, body_bot), fill=LINE, width=3)

    # Template sections
    problem = (60, 870, 1280, 1800)
    challenge = (60, 1820, 1280, 2380)
    dataset = (60, 2405, 1280, 4165)
    pipeline = (1302, 820, 2274, 2625)
    progression = (1302, 2650, 2274, 4135)
    results = (2302, 850, 3138, 2380)
    calib = (2302, 2400, 3138, 3060)
    conclusions = (2302, 3085, 3138, 4135)

    for box in [problem, challenge, pipeline, progression, results, calib, conclusions]:
        section(draw, box)

    center_text(draw, problem, "The Problem", F_PLACE)
    center_text(draw, challenge, "Core Challenge", F_PLACE)
    center_text(draw, pipeline, "Pipeline", F_PLACE)
    center_text(draw, progression, "Research\nProgression", F_PLACE)
    center_text(draw, results, "Key Results", F_PLACE)
    center_text(draw, calib, "Calibration\nStrategy", F_PLACE)
    center_text(draw, conclusions, "Conclusions", F_PLACE)

    # Dataset section content
    section(draw, dataset, "Our Dataset")
    dx1, dy1, dx2, dy2 = dataset
    y = dy1 + 112
    y = paragraph(
        draw,
        (dx1 + 44, y),
        "~15,064 synthetic images generated in Unreal Engine with ground-truth metric distance across 12 distances and 48 condition strata.",
        F_BODY_B,
        dx2 - dx1 - 88,
        fill=TEXT,
        line_gap=10,
    )
    y += 8
    y = paragraph(
        draw,
        (dx1 + 44, y),
        "Preparation included automated bbox alignment, folder organization, ray-based validity checks, and varied camera viewpoints.",
        F_BODY,
        dx2 - dx1 - 88,
        fill=MUTED,
        line_gap=9,
    )

    # Compact process pills
    pill_y = y + 16
    pill_x = dx1 + 44
    for label, color in [
        ("bbox automation", TEAL),
        ("ray checks", GREEN),
        ("view angles", (111, 202, 240)),
    ]:
        tw, th = text_size(draw, label, F_SMALL_B)
        draw.rounded_rectangle((pill_x, pill_y, pill_x + tw + 34, pill_y + 44), radius=22, fill=color)
        draw.text((pill_x + 17, pill_y + 10), label, font=F_SMALL_B, fill=INK)
        pill_x += tw + 48

    # Drone examples
    img_y = pill_y + 72
    panel_w, panel_h = 535, 325
    examples = [
        (
            Path(r"C:\Users\depthlev\Desktop\droneImages\dataset\depth_20\clear_sky\10AM\HighresScreenshot00277_depth_20_clear_sky_10AM.png"),
            Path(r"C:\Users\depthlev\Desktop\droneImages\dataset\depth_20\clear_sky\10AM\HighresScreenshot00277_depth_20_clear_sky_10AM.txt"),
            "20 m | clear sky | 10AM",
            dx1 + 44,
        ),
        (
            Path(r"C:\Users\depthlev\Desktop\droneImages\dataset\depth_50\light_rain\8PM\HighresScreenshot00001_depth_50_light_rain_8PM.png"),
            Path(r"C:\Users\depthlev\Desktop\droneImages\dataset\depth_50\light_rain\8PM\HighresScreenshot00001_depth_50_light_rain_8PM.txt"),
            "50 m | light rain | 8PM",
            dx1 + 44 + panel_w + 44,
        ),
    ]
    for image_path, label_path, tag, px in examples:
        panel = make_drone_panel(image_path, label_path, (panel_w, panel_h), tag)
        img.paste(panel, (px, img_y))

    draw_table(draw, (dx1 + 44, img_y + panel_h + 40, dx2 - 44, dy2 - 48))

    # Bottom reserved band
    draw.rectangle((48, body_bot, W - 48, H - 36), fill=BG_DARK, outline=LINE, width=3)
    center_text(draw, (110, body_bot + 35, W - 110, H - 55), "QR Code | Contact | Repository", F_SMALL, (170, 220, 228))

    img.save(OUT_PATH, quality=95)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
