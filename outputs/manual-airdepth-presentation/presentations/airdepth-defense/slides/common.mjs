const ROOT = "c:/Users/noams/OneDrive/Desktop/school/final project/Z_Drone_Final_Project_GA";

const C = {
  bg: "#E9EDF5",
  navy: "#0B2463",
  navy2: "#173B7A",
  steel: "#2E5E9E",
  mist: "#B9C5D8",
  lightMist: "#D6DDE9",
  orange: "#FF8748",
  cream: "#E8DED1",
  white: "#FFFFFF",
  ink: "#111827",
  muted: "#4B5563",
  green: "#2F8A67",
  red: "#C9533D",
};

function bg(slide, ctx) {
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 720, fill: C.bg, line: ctx.line() });
}

function footer(slide, ctx, n) {
  ctx.addText(slide, {
    text: String(n).padStart(2, "0"),
    x: 1190, y: 666, w: 46, h: 24,
    fontSize: 12, bold: true, color: C.mist, align: "right", valign: "mid",
  });
  ctx.addShape(slide, { x: 70, y: 678, w: 1040, h: 2, fill: "#D8DEEA", line: ctx.line() });
}

function title(slide, ctx, kicker, claim, sub = "") {
  ctx.addText(slide, {
    text: kicker.toUpperCase(),
    x: 70, y: 42, w: 210, h: 20,
    fontSize: 11, bold: true, color: C.orange, align: "left", valign: "mid",
  });
  ctx.addText(slide, {
    text: claim,
    x: 70, y: 68, w: 870, h: 52,
    fontSize: 32, bold: true, color: C.navy, typeface: ctx.fonts.title,
  });
  if (sub) {
    ctx.addText(slide, {
      text: sub,
      x: 70, y: 116, w: 880, h: 36,
      fontSize: 16, color: C.ink,
    });
  }
}

function pill(slide, ctx, text, x, y, w, fill = C.navy, color = C.white) {
  ctx.addShape(slide, { geometry: "roundRect", x, y, w, h: 34, fill, line: ctx.line() });
  ctx.addText(slide, {
    text, x, y: y + 5, w, h: 22, fontSize: 13, bold: true, color, align: "center", valign: "mid",
  });
}

function num(slide, ctx, value, x, y, fill = C.navy, size = 54) {
  ctx.addShape(slide, { geometry: "ellipse", x, y, w: size, h: size, fill, line: ctx.line() });
  ctx.addText(slide, {
    text: String(value).padStart(2, "0"),
    x, y: y + size / 2 - 11, w: size, h: 22,
    fontSize: size > 48 ? 14 : 12, bold: true, color: C.white, align: "center", valign: "mid",
  });
}

function stepCard(slide, ctx, i, label, body, x, y, w, accent = C.navy) {
  ctx.addShape(slide, { geometry: "roundRect", x, y, w, h: 64, fill: C.white, line: ctx.line("#00000000", 0) });
  num(slide, ctx, i, x - 24, y + 8, accent, 48);
  ctx.addText(slide, { text: label, x: x + 40, y: y + 12, w: w - 60, h: 18, fontSize: 13, bold: true, color: C.ink });
  ctx.addText(slide, { text: body, x: x + 40, y: y + 32, w: w - 70, h: 25, fontSize: 11.5, color: C.ink });
}

function metric(slide, ctx, value, label, x, y, w, fill = C.white, accent = C.navy) {
  ctx.addShape(slide, { geometry: "roundRect", x, y, w, h: 94, fill, line: ctx.line("#00000000", 0) });
  ctx.addText(slide, { text: value, x: x + 18, y: y + 17, w: w - 36, h: 32, fontSize: 30, bold: true, color: accent, align: "center" });
  ctx.addText(slide, { text: label, x: x + 18, y: y + 54, w: w - 36, h: 28, fontSize: 12.5, color: C.ink, align: "center" });
}

function tinyBar(slide, ctx, label, value, max, x, y, w, color) {
  const barW = Math.max(6, (value / max) * w);
  ctx.addText(slide, { text: label, x, y: y - 2, w: 170, h: 18, fontSize: 12, bold: true, color: C.ink });
  ctx.addShape(slide, { geometry: "roundRect", x: x + 180, y, w, h: 18, fill: "#DDE3EF", line: ctx.line() });
  ctx.addShape(slide, { geometry: "roundRect", x: x + 180, y, w: barW, h: 18, fill: color, line: ctx.line() });
  ctx.addText(slide, { text: `${value.toFixed(value < 10 ? 2 : 1)}m`, x: x + 190 + barW, y: y - 2, w: 70, h: 18, fontSize: 12, bold: true, color });
}

async function image(slide, ctx, rel, x, y, w, h, fit = "cover") {
  return ctx.addImage(slide, { path: `${ROOT}/${rel}`, x, y, w, h, fit });
}

function arrow(slide, ctx, x, y, w, h, color) {
  ctx.addShape(slide, { x, y: y + h * 0.35, w: w * 0.72, h: h * 0.3, fill: color, line: ctx.line() });
  ctx.addShape(slide, { geometry: "triangle", x: x + w * 0.62, y, w: w * 0.38, h, fill: color, line: ctx.line() });
}

function stage(slide, ctx, i, label, x, y, color) {
  num(slide, ctx, i, x + 38, y, color, 52);
  ctx.addText(slide, { text: label, x, y: y + 65, w: 130, h: 38, fontSize: 13, bold: true, color: C.navy, align: "center" });
}

function pencil(slide, ctx, x, y, color, label, detail, icon = "") {
  ctx.addShape(slide, { geometry: "roundRect", x: x + 16, y, w: 64, h: 44, fill: C.white, line: ctx.line() });
  if (icon) ctx.addText(slide, { text: icon, x: x + 16, y: y + 8, w: 64, h: 25, fontSize: 19, bold: true, color: C.navy, align: "center" });
  ctx.addShape(slide, { x, y: y + 44, w: 96, h: 66, fill: color, line: ctx.line() });
  ctx.addShape(slide, { geometry: "triangle", x, y: y + 106, w: 96, h: 50, fill: color, line: ctx.line() });
  pill(slide, ctx, label, x - 24, y + 168, 144, color, color === C.cream ? C.navy : C.white);
  ctx.addText(slide, { text: detail, x: x - 36, y: y + 215, w: 168, h: 62, fontSize: 12, color: C.ink, align: "center" });
}

function barChart(slide, ctx, items, x, y, w, h, max) {
  const gap = 18;
  const barH = (h - gap * (items.length - 1)) / items.length;
  items.forEach((it, idx) => {
    const yy = y + idx * (barH + gap);
    ctx.addText(slide, { text: it.label, x, y: yy + 5, w: 190, h: 24, fontSize: 13, bold: true, color: C.ink });
    ctx.addShape(slide, { geometry: "roundRect", x: x + 210, y: yy, w: w - 280, h: barH, fill: "#DCE3EF", line: ctx.line() });
    ctx.addShape(slide, { geometry: "roundRect", x: x + 210, y: yy, w: Math.max(8, (it.value / max) * (w - 280)), h: barH, fill: it.color, line: ctx.line() });
    ctx.addText(slide, { text: it.valueText || `${it.value.toFixed(2)}m`, x: x + w - 58, y: yy + 5, w: 70, h: 24, fontSize: 13, bold: true, color: it.color });
  });
}

async function slide1(slide, ctx) {
  bg(slide, ctx);
  ctx.addText(slide, { text: "AirDepth", x: 74, y: 76, w: 500, h: 70, fontSize: 58, bold: true, color: C.navy, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: "Monocular Drone Distance Estimation", x: 78, y: 150, w: 560, h: 34, fontSize: 24, color: C.ink });
  ctx.addText(slide, { text: "Recovering useful metric scale from one RGB image", x: 80, y: 205, w: 470, h: 58, fontSize: 18, color: C.muted });
  metric(slide, ctx, "6.01m", "synthetic MAE", 76, 405, 170, C.white, C.navy);
  metric(slide, ctx, "3.28m", "real MAE after calibration", 268, 405, 210, C.white, C.orange);
  metric(slide, ctx, "RGB only", "no LiDAR / stereo / GPS", 500, 405, 210, C.white, C.steel);
  ctx.addShape(slide, { x: 760, y: 75, w: 450, h: 500, fill: C.white, line: ctx.line() });
  await image(slide, ctx, "experiments/background_drone_exp/data/images/synthetic_buildings/drone1.png", 780, 95, 410, 250);
  ctx.addShape(slide, { x: 812, y: 238, w: 74, h: 46, fill: "#00000000", line: ctx.line(C.orange, 4) });
  pill(slide, ctx, "single RGB frame", 810, 372, 170, C.navy);
  arrow(slide, ctx, 1000, 374, 74, 30, C.orange);
  pill(slide, ctx, "distance in meters", 1090, 372, 178, C.orange);
  footer(slide, ctx, 1);
}

async function slide2(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Problem", "Detection is not enough; the system needs distance.", "For tracking, avoidance, and counter-UAV response, a 2D box is only half the answer.");
  await image(slide, ctx, "experiments/background_drone_exp/data/images/real_buildings/image1.png", 82, 178, 500, 310);
  ctx.addShape(slide, { x: 312, y: 300, w: 72, h: 48, fill: "#00000000", line: ctx.line(C.orange, 4) });
  stepCard(slide, ctx, 1, "What the camera gives", "RGB pixels and a 2D target location.", 710, 185, 430, C.navy);
  stepCard(slide, ctx, 2, "What the system needs", "Metric Z-distance from the observer.", 710, 282, 430, C.steel);
  stepCard(slide, ctx, 3, "Why it matters", "Distance supports response timing and 3D localization.", 710, 379, 430, C.orange);
  footer(slide, ctx, 2);
}

async function slide3(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Challenge", "Scale ambiguity is the core monocular obstacle.", "The image projection loses absolute metric scale.");
  ctx.addShape(slide, { x: 80, y: 190, w: 470, h: 320, fill: C.white, line: ctx.line() });
  ctx.addShape(slide, { x: 730, y: 190, w: 470, h: 320, fill: C.white, line: ctx.line() });
  ctx.addText(slide, { text: "Small + near", x: 150, y: 230, w: 330, h: 28, fontSize: 26, bold: true, color: C.navy, align: "center" });
  ctx.addText(slide, { text: "Large + far", x: 800, y: 230, w: 330, h: 28, fontSize: 26, bold: true, color: C.navy, align: "center" });
  ctx.addShape(slide, { geometry: "ellipse", x: 285, y: 315, w: 60, h: 60, fill: C.orange, line: ctx.line() });
  ctx.addShape(slide, { geometry: "ellipse", x: 930, y: 315, w: 60, h: 60, fill: C.steel, line: ctx.line() });
  ctx.addShape(slide, { x: 180, y: 408, w: 260, h: 18, fill: C.lightMist, line: ctx.line() });
  ctx.addShape(slide, { x: 830, y: 408, w: 260, h: 18, fill: C.lightMist, line: ctx.line() });
  ctx.addText(slide, { text: "Similar image footprint", x: 400, y: 560, w: 480, h: 32, fontSize: 28, bold: true, color: C.orange, align: "center" });
  ctx.addText(slide, { text: "AirDepth learns the mapping to meters from labeled examples.", x: 352, y: 602, w: 580, h: 24, fontSize: 17, color: C.ink, align: "center" });
  footer(slide, ctx, 3);
}

async function slide4(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Gap", "Hardware depth is accurate, but not lightweight.", "AirDepth targets the low-cost RGB-only path.");
  ctx.addText(slide, { text: "Traditional depth", x: 160, y: 168, w: 300, h: 30, fontSize: 24, bold: true, color: C.navy, align: "center" });
  ctx.addText(slide, { text: "RGB-only target", x: 820, y: 168, w: 300, h: 30, fontSize: 24, bold: true, color: C.orange, align: "center" });
  ["LiDAR", "Radar", "Stereo", "RGB-D"].forEach((t, i) => stepCard(slide, ctx, i + 1, t, ["High accuracy", "Range sensing", "Triangulation", "Dedicated depth"][i], 120, 230 + i * 78, 390, i === 0 ? C.navy : i === 3 ? C.orange : C.steel));
  ["Low payload", "Low cost", "Single camera", "Needs learning"].forEach((t, i) => stepCard(slide, ctx, i + 1, t, ["Fits lighter platforms", "No special depth sensor", "Simpler field setup", "Metric scale is inferred"][i], 790, 230 + i * 78, 390, i === 3 ? C.orange : C.navy));
  arrow(slide, ctx, 572, 335, 132, 70, C.orange);
  footer(slide, ctx, 4);
}

async function slide5(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Method", "AirDepth separates perception from metric conversion.", "The depth model supplies relative structure; the regressor and calibration recover usable scale.");
  const xs = [110, 315, 520, 725, 930];
  ["Detect", "Depth", "Features", "Regress", "Calibrate"].forEach((label, i) => {
    pencil(slide, ctx, xs[i], 190 + (i % 2) * 36, [C.navy, C.steel, C.mist, C.orange, C.cream][i], label, ["find target box", "relative map", "box + local depth", "predict meters", "adapt to real"][i], ["□", "≋", "Φ", "ŷ", "↺"][i]);
    if (i < xs.length - 1) arrow(slide, ctx, xs[i] + 128, 302, 54, 30, i === 2 ? C.orange : C.navy);
  });
  footer(slide, ctx, 5);
}

async function slide6(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Data Strategy", "Synthetic data teaches scale; real data tests transfer.", "The two datasets play different roles in the evidence chain.");
  ctx.addShape(slide, { x: 104, y: 190, w: 470, h: 330, fill: C.white, line: ctx.line() });
  ctx.addShape(slide, { x: 706, y: 190, w: 470, h: 330, fill: C.white, line: ctx.line() });
  metric(slide, ctx, "15,064", "synthetic Unreal images", 158, 232, 170, C.bg, C.navy);
  metric(slide, ctx, "20-150m", "known ground-truth range", 356, 232, 170, C.bg, C.steel);
  metric(slide, ctx, "489", "real Nenrus images", 760, 232, 170, C.bg, C.orange);
  metric(slide, ctx, "transfer", "calibration evaluation", 958, 232, 170, C.bg, C.navy);
  ctx.addText(slide, { text: "Training + controlled benchmark", x: 170, y: 410, w: 330, h: 28, fontSize: 22, bold: true, color: C.navy, align: "center" });
  ctx.addText(slide, { text: "Real-domain bias + calibration", x: 775, y: 410, w: 330, h: 28, fontSize: 22, bold: true, color: C.orange, align: "center" });
  footer(slide, ctx, 6);
}

async function slide7(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Synthetic Dataset", "Controlled images made metric supervision possible.", "Each frame carries known distance and environment labels.");
  const imgs = ["drone1.png", "drone2.png", "drone3.png", "drone4.png", "drone5.png", "drone6.png"];
  for (let i = 0; i < imgs.length; i += 1) {
    await image(slide, ctx, `experiments/background_drone_exp/data/images/synthetic_buildings/${imgs[i]}`, 80 + (i % 3) * 238, 180 + Math.floor(i / 3) * 150, 210, 118);
  }
  const facts = [
    ["01", "Distance", "20, 30, ... 150m"],
    ["02", "Weather", "clear sky / light rain"],
    ["03", "Time", "10AM / 8PM"],
    ["04", "Ground truth", "controlled Unreal setup"],
  ];
  facts.forEach((f, i) => stepCard(slide, ctx, f[0], f[1], f[2], 820, 178 + i * 82, 330, i === 1 ? C.orange : i === 2 ? C.steel : C.navy));
  footer(slide, ctx, 7);
}

async function slide8(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Feature Design", "The bounding box became a distance sensor.", "Geometry was not just an intermediate detector output.");
  await image(slide, ctx, "experiments/background_drone_exp/data/images/synthetic_buildings/drone3.png", 82, 188, 470, 300);
  ctx.addShape(slide, { x: 295, y: 315, w: 84, h: 56, fill: "#00000000", line: ctx.line(C.orange, 4) });
  const features = [
    ["Width / height", "apparent scale"],
    ["Area ratio", "target footprint"],
    ["Aspect ratio", "viewing shape"],
    ["Center position", "image location"],
  ];
  features.forEach((f, i) => stepCard(slide, ctx, i + 1, f[0], f[1], 728, 180 + i * 82, 360, i === 0 ? C.navy : i === 3 ? C.orange : C.steel));
  footer(slide, ctx, 8);
}

async function slide9(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Depth Features", "Local depth worked better than global scene context.", "Relative depth was useful as a cue, not as the final metric answer.");
  const steps = [
    ["Run depth model", "dense relative map"],
    ["Crop around drone", "target-centered signal"],
    ["Summarize robustly", "midpoint / median / context"],
    ["Jitter the box", "stable features under detector noise"],
  ];
  steps.forEach((s, i) => {
    const y = 180 + i * 88;
    ctx.addShape(slide, { x: 280, y: y + 22, w: 480, h: 6, fill: i === 3 ? C.orange : C.navy, line: ctx.line() });
    stepCard(slide, ctx, i + 1, s[0], s[1], 410, y, 500, i === 3 ? C.orange : i === 2 ? C.steel : C.navy);
  });
  ctx.addText(slide, { text: "Relative depth alone is not meters", x: 110, y: 580, w: 1060, h: 40, fontSize: 30, bold: true, color: C.orange, align: "center" });
  footer(slide, ctx, 9);
}

async function slide10(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Model Evolution", "Ablations turned a weak depth cue into a 6m system.", "Each step added one kind of missing scale information.");
  const vals = [
    ["Scale only", 47.70, C.mist],
    ["Depth linear", 32.29, C.steel],
    ["Linear + box", 13.20, C.navy2],
    ["Random Forest", 7.19, C.navy],
    ["RF + jitter", 6.39, C.orange],
    ["RF-XGB ensemble", 6.01, C.orange],
  ];
  vals.forEach((v, i) => {
    const x = 130 + i * 170;
    const h = 330 * (v[1] / 47.7);
    ctx.addShape(slide, { x, y: 520 - h, w: 92, h, fill: v[2], line: ctx.line() });
    ctx.addText(slide, { text: `${v[1].toFixed(2)}m`, x: x - 8, y: 486 - h, w: 108, h: 24, fontSize: 15, bold: true, color: v[2], align: "center" });
    ctx.addText(slide, { text: v[0], x: x - 30, y: 540, w: 150, h: 38, fontSize: 12, bold: true, color: C.ink, align: "center" });
  });
  ctx.addText(slide, { text: "MAE lower is better", x: 910, y: 164, w: 230, h: 26, fontSize: 16, bold: true, color: C.muted, align: "right" });
  footer(slide, ctx, 10);
}

async function slide11(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Synthetic Result", "The final ensemble was strongest and stable on holdout.", "Held-out synthetic test set: 2,260 rows.");
  const items = [
    { label: "Scale only", value: 47.70, color: C.mist },
    { label: "Depth linear", value: 32.29, color: C.steel },
    { label: "Linear + box", value: 13.20, color: C.navy2 },
    { label: "Random Forest", value: 7.19, color: C.navy },
    { label: "RF + jitter", value: 6.39, color: C.orange },
    { label: "Final ensemble", value: 6.01, color: C.orange },
  ];
  barChart(slide, ctx, items, 100, 190, 760, 310, 50);
  metric(slide, ctx, "6.01m", "test MAE", 940, 208, 190, C.white, C.orange);
  metric(slide, ctx, "0.949", "R2", 940, 330, 190, C.white, C.navy);
  metric(slide, ctx, "81%", "within 10m", 940, 452, 190, C.white, C.steel);
  footer(slide, ctx, 11);
}

async function slide12(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Distance Breakdown", "Farther drones remain the hardest regime.", "The final model degrades gracefully, but target size still matters.");
  const bands = [
    ["Near", 3.37, "≤60m", C.navy],
    ["Mid", 6.96, "60-100m", C.steel],
    ["Far", 9.18, ">100m", C.orange],
  ];
  bands.forEach((b, i) => {
    const x = 200 + i * 300;
    const h = b[1] * 34;
    ctx.addShape(slide, { geometry: "ellipse", x: x - 42, y: 210, w: 84, h: 84, fill: b[3], line: ctx.line() });
    ctx.addText(slide, { text: b[0], x: x - 80, y: 312, w: 160, h: 28, fontSize: 24, bold: true, color: C.navy, align: "center" });
    ctx.addText(slide, { text: b[2], x: x - 80, y: 344, w: 160, h: 22, fontSize: 14, color: C.muted, align: "center" });
    ctx.addShape(slide, { geometry: "roundRect", x: x - 45, y: 555 - h, w: 90, h, fill: b[3], line: ctx.line() });
    ctx.addText(slide, { text: `${b[1]}m`, x: x - 60, y: 520 - h, w: 120, h: 24, fontSize: 18, bold: true, color: b[3], align: "center" });
  });
  ctx.addText(slide, { text: "Less pixels -> less stable boxes and depth cues", x: 320, y: 610, w: 650, h: 30, fontSize: 22, bold: true, color: C.orange, align: "center" });
  footer(slide, ctx, 12);
}

async function slide13(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Sim-to-Real", "The raw synthetic model overpredicted real drones.", "This exposed systematic domain shift, not just random error.");
  ctx.addShape(slide, { x: 110, y: 202, w: 410, h: 270, fill: C.white, line: ctx.line() });
  ctx.addShape(slide, { x: 760, y: 202, w: 410, h: 270, fill: C.white, line: ctx.line() });
  metric(slide, ctx, "23.80m", "raw real MAE", 210, 260, 210, C.bg, C.orange);
  metric(slide, ctx, "99.8%", "overprediction rate", 860, 260, 210, C.bg, C.red);
  arrow(slide, ctx, 565, 310, 130, 70, C.orange);
  ctx.addText(slide, { text: "different camera, drone scale, annotation behavior, and image statistics", x: 220, y: 520, w: 840, h: 34, fontSize: 23, bold: true, color: C.navy, align: "center" });
  footer(slide, ctx, 13);
}

async function slide14(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Calibration", "A lightweight real-domain mapping fixed most of the bias.", "20% real calibration data; remaining rows used for evaluation.");
  const groups = [
    ["Overall", 23.80, 3.28],
    ["Kongsberg", 33.86, 2.78],
    ["Vestfold", 16.65, 3.54],
  ];
  groups.forEach((g, i) => {
    const x = 180 + i * 310;
    ctx.addText(slide, { text: g[0], x: x - 70, y: 545, w: 180, h: 28, fontSize: 18, bold: true, color: C.ink, align: "center" });
    const beforeH = g[1] * 8.5;
    const afterH = g[2] * 8.5;
    ctx.addShape(slide, { x: x - 30, y: 505 - beforeH, w: 62, h: beforeH, fill: C.red, line: ctx.line() });
    ctx.addShape(slide, { x: x + 48, y: 505 - afterH, w: 62, h: afterH, fill: C.green, line: ctx.line() });
    ctx.addText(slide, { text: `${g[1].toFixed(2)}m`, x: x - 52, y: 477 - beforeH, w: 105, h: 22, fontSize: 15, bold: true, color: C.red, align: "center" });
    ctx.addText(slide, { text: `${g[2].toFixed(2)}m`, x: x + 27, y: 477 - afterH, w: 105, h: 22, fontSize: 15, bold: true, color: C.green, align: "center" });
  });
  pill(slide, ctx, "Before calibration", 900, 202, 170, C.red);
  pill(slide, ctx, "After calibration", 900, 248, 170, C.green);
  metric(slide, ctx, "96.4%", "within 10m after calibration", 870, 350, 230, C.white, C.green);
  footer(slide, ctx, 14);
}

async function slide15(slide, ctx) {
  bg(slide, ctx); title(slide, ctx, "Contributions", "The project proved the path, and named the limits.", "A judge should hear both the result and the deployment boundary.");
  const cols = [
    ["01", "Method", "Hybrid monocular metric-distance framework."],
    ["02", "Experiments", "Ablations from scale-only to final ensemble."],
    ["03", "Transfer", "Real-domain calibration reduced systematic bias."],
  ];
  cols.forEach((c, i) => {
    const x = 120 + i * 350;
    num(slide, ctx, c[0], x + 108, 190, i === 2 ? C.orange : C.navy, 74);
    ctx.addShape(slide, { geometry: "roundRect", x, y: 300, w: 290, h: 150, fill: C.white, line: ctx.line() });
    ctx.addText(slide, { text: c[1], x: x + 30, y: 330, w: 230, h: 28, fontSize: 24, bold: true, color: C.navy, align: "center" });
    ctx.addText(slide, { text: c[2], x: x + 40, y: 374, w: 210, h: 48, fontSize: 15, color: C.ink, align: "center" });
  });
  ctx.addText(slide, { text: "Limits: detector quality, real calibration need, runtime still to benchmark, far range remains hardest.", x: 170, y: 545, w: 940, h: 40, fontSize: 20, bold: true, color: C.orange, align: "center" });
  footer(slide, ctx, 15);
}

async function slide16(slide, ctx) {
  bg(slide, ctx);
  ctx.addText(slide, { text: "Questions?", x: 80, y: 95, w: 520, h: 70, fontSize: 60, bold: true, color: C.navy, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: "Monocular depth alone is not enough. AirDepth combines relative depth, bounding-box geometry, supervised regression, and calibration to recover useful metric scale.", x: 84, y: 185, w: 560, h: 120, fontSize: 24, color: C.ink });
  ctx.addShape(slide, { x: 745, y: 90, w: 430, h: 430, fill: C.white, line: ctx.line() });
  await image(slide, ctx, "experiments/background_drone_exp/data/images/synthetic_buildings/drone6.png", 765, 110, 390, 230);
  metric(slide, ctx, "RGB only", "single camera input", 790, 380, 160, C.bg, C.navy);
  metric(slide, ctx, "meters", "calibrated output", 980, 380, 160, C.bg, C.orange);
  footer(slide, ctx, 16);
}

const builders = [slide1, slide2, slide3, slide4, slide5, slide6, slide7, slide8, slide9, slide10, slide11, slide12, slide13, slide14, slide15, slide16];

export async function buildSlide(presentation, ctx, index) {
  const slide = presentation.slides.add();
  await builders[index - 1](slide, ctx);
  return slide;
}
