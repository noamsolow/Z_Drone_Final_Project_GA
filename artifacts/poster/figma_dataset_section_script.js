// Figma use_figma script: populate the "Our Dataset" section.
// Assumes the poster template file already contains:
// - Frame named "AirDepth Poster Template - 80x110 ratio"
// - Rectangle named "Section 3 - Our Dataset"
// - Two user-added image nodes selected, or at least two image-filled nodes on the page.

await figma.loadFontAsync({ family: "Inter", style: "Regular" });
await figma.loadFontAsync({ family: "Inter", style: "Medium" });
await figma.loadFontAsync({ family: "Inter", style: "Bold" });

const page = figma.currentPage;
const poster = page.query('FRAME[name="AirDepth Poster Template - 80x110 ratio"]').first();
if (!poster) throw new Error("Poster frame not found");

const createdNodeIds = [];
const mutatedNodeIds = [];

const white = { r: 1, g: 1, b: 1 };
const teal = { r: 0.078, g: 0.392, b: 0.486 };
const tealDark = { r: 0.045, g: 0.255, b: 0.32 };
const black = { r: 0.02, g: 0.027, b: 0.033 };

function solid(color, opacity = 1) {
  return [{ type: "SOLID", color, opacity }];
}

function hasImageFill(node) {
  return "fills" in node && Array.isArray(node.fills) && node.fills.some((f) => f.type === "IMAGE");
}

function isDescendantOf(node, ancestor) {
  let p = node.parent;
  while (p) {
    if (p.id === ancestor.id) return true;
    p = p.parent;
  }
  return false;
}

function addText(name, chars, x, y, w, h, size, style = "Regular", color = white, align = "LEFT") {
  const t = figma.createText();
  t.name = name;
  t.fontName = { family: "Inter", style };
  t.characters = chars;
  t.fontSize = size;
  t.fills = solid(color);
  t.textAlignHorizontal = align;
  t.textAlignVertical = "TOP";
  t.x = x;
  t.y = y;
  t.resize(w, h);
  poster.appendChild(t);
  createdNodeIds.push(t.id);
  return t;
}

function addRect(name, x, y, w, h, radius, fill, stroke = black, strokeWeight = 0.8) {
  const r = figma.createRectangle();
  r.name = name;
  r.x = x;
  r.y = y;
  r.resize(w, h);
  r.cornerRadius = radius;
  r.fills = solid(fill);
  r.strokes = strokeWeight > 0 ? solid(stroke) : [];
  r.strokeWeight = strokeWeight;
  poster.appendChild(r);
  createdNodeIds.push(r.id);
  return r;
}

function removePreviousDatasetContent() {
  const old = poster
    .query('[name^="Dataset "]')
    .toArray()
    .filter((n) => !["Section 3 - Our Dataset"].includes(n.name));
  for (const node of old) node.remove();
}

removePreviousDatasetContent();

for (const node of poster.query('TEXT[name="Label - Our Dataset"], TEXT[name="Label - Dataset Images"]').toArray()) {
  node.visible = false;
  mutatedNodeIds.push(node.id);
}

// Prefer selected user images. Fallback: use image nodes outside the poster.
let imageNodes = page.selection.filter((node) => hasImageFill(node));
if (imageNodes.length < 2) {
  imageNodes = page.findAll((node) => hasImageFill(node) && !isDescendantOf(node, poster));
}
if (imageNodes.length < 2) {
  throw new Error("Could not find two image nodes. Select the two dataset images in Figma and rerun.");
}

// Use the two widest/largest images, preserving the user's order where possible.
imageNodes = imageNodes
  .slice()
  .sort((a, b) => b.width * b.height - a.width * a.height)
  .slice(0, 2);

const sx = 0;
const sy = 615;
const sw = 320;

addText("Dataset title", "Our Dataset", sx + 18, sy + 18, sw - 36, 24, 17, "Bold");
addText(
  "Dataset intro",
  "Generated ~15,064 synthetic drone images in Unreal Engine with automated bbox alignment, folder structuring, ray-based validity checks, and varied camera viewpoints.",
  sx + 18,
  sy + 47,
  sw - 36,
  58,
  10.2,
  "Regular"
);

const pills = [
  ["bbox automation", sx + 18, sy + 110, 88],
  ["ray checks", sx + 112, sy + 110, 65],
  ["view angles", sx + 184, sy + 110, 80],
];
for (const [label, x, y, w] of pills) {
  addRect("Dataset pill - " + label, x, y, w, 20, 9, tealDark, white, 0.5);
  addText("Dataset pill label - " + label, label, x + 6, y + 5, w - 12, 12, 7.2, "Medium", white, "CENTER");
}

addText("Dataset image label - near", "20 m | clear sky | 10AM", sx + 18, sy + 142, 132, 15, 8.2, "Bold", white, "CENTER");
addText("Dataset image label - far", "50 m | light rain | 8PM", sx + 166, sy + 142, 136, 15, 8.2, "Bold", white, "CENTER");

function placeImage(node, name, x, y, w, h) {
  node.name = name;
  node.x = x;
  node.y = y;
  node.resize(w, h);
  node.cornerRadius = 8;
  node.strokes = solid(white, 0.8);
  node.strokeWeight = 0.7;
  poster.appendChild(node);
  mutatedNodeIds.push(node.id);
}

placeImage(imageNodes[0], "Dataset image - 20m clear sky 10AM", sx + 18, sy + 160, 132, 76);
placeImage(imageNodes[1], "Dataset image - 50m light rain 8PM", sx + 166, sy + 160, 136, 76);

const tx = sx + 18;
const ty = sy + 252;
const tw = sw - 36;
addRect("Dataset stats table background", tx, ty, tw, 126, 10, tealDark, white, 0.7);
addText("Dataset table header", "Dataset split summary", tx + 10, ty + 8, tw - 20, 14, 9.5, "Bold");

const rows = [
  ["Images", "15,064", "100%"],
  ["Weather", "clear 7,547", "50.10%"],
  ["", "rain 7,517", "49.90%"],
  ["Time", "10AM 9,998", "66.37%"],
  ["", "8PM 5,066", "33.63%"],
  ["Ranges", "Near 6,301", "41.83%"],
  ["", "Mid 5,027", "33.37%"],
  ["", "Far 3,736", "24.80%"],
];

let y = ty + 30;
for (const [group, value, pct] of rows) {
  addText("Dataset table group - " + group + value, group, tx + 10, y, 54, 10, 7.3, group ? "Medium" : "Regular");
  addText("Dataset table value - " + value, value, tx + 70, y, 116, 10, 7.3, "Regular");
  addText("Dataset table pct - " + pct, pct, tx + 192, y, 42, 10, 7.3, "Medium", white, "RIGHT");
  y += 11.5;
}

const screenshot = await poster.screenshot({ scale: 0.75 });
return {
  createdNodeIds,
  mutatedNodeIds,
  imageNodeIds: imageNodes.map((n) => n.id),
  datasetSection: "populated",
  screenshot,
};
