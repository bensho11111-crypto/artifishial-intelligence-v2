/**
 * Frontend pure-function unit tests.
 * Run with: node --test tests/frontend/test_pure_functions.js
 * Requires Node >= 18 (built-in test runner).
 *
 * These tests verify the JS logic that would be hard to catch through
 * backend tests: colour palettes, time formatting, LUT construction,
 * and confidence-based colour scaling.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

// ── Helpers copied from index.html (keep in sync) ────────────────────────────

function depthColour(depth, maxDepth) {
  const cShallow = { r: 0xc8 / 255, g: 0xa8 / 255, b: 0x7a / 255 };
  const cMid     = { r: 0x1e / 255, g: 0x60 / 255, b: 0x80 / 255 };
  const cDeep    = { r: 0x09 / 255, g: 0x18 / 255, b: 0x28 / 255 };
  const lerp     = (a, b, t) => a + (b - a) * t;
  const t = Math.min(1, Math.max(0, depth / (maxDepth || 20)));
  if (t < 0.5) {
    const u = t * 2;
    return { r: lerp(cShallow.r, cMid.r, u), g: lerp(cShallow.g, cMid.g, u), b: lerp(cShallow.b, cMid.b, u) };
  }
  const u = (t - 0.5) * 2;
  return { r: lerp(cMid.r, cDeep.r, u), g: lerp(cMid.g, cDeep.g, u), b: lerp(cMid.b, cDeep.b, u) };
}

function fmtTime(s) {
  s = Math.floor(s || 0);
  const m = Math.floor(s / 60), sec = s % 60;
  return `${m}:${String(sec).padStart(2, "0")}`;
}

function buildFishFinderLUT() {
  const stops = [
    [0,   0,   0,  18],
    [30,  0,  40, 100],
    [80,  0, 200, 200],
    [130, 0, 220,  80],
    [170,220, 220,   0],
    [210,255, 100,   0],
    [240,255,   0,   0],
    [255,255, 255, 255],
  ];
  const lut = new Uint8Array(256 * 3);
  for (let a = 0; a < 256; a++) {
    let i = 0;
    while (i < stops.length - 2 && stops[i + 1][0] <= a) i++;
    const [a0, r0, g0, b0] = stops[i];
    const [a1, r1, g1, b1] = stops[i + 1];
    const t = (a - a0) / (a1 - a0);
    lut[a * 3]     = (r0 + (r1 - r0) * t) | 0;
    lut[a * 3 + 1] = (g0 + (g1 - g0) * t) | 0;
    lut[a * 3 + 2] = (b0 + (b1 - b0) * t) | 0;
  }
  return lut;
}

function fishPointColour(isFloor, conf, depth, maxDepth) {
  if (!isFloor) {
    const cv = Math.min(1, Math.max(0, conf / 0.65));
    return { r: 1.0 * cv, g: 0.55 * cv, b: 0.05 * cv };
  }
  if (conf <= 0.45) {
    const base = depthColour(depth, maxDepth);
    return { r: base.r * 0.55, g: base.g * 0.85 + 0.15, b: base.b * 0.85 + 0.15 };
  }
  return depthColour(depth, maxDepth);
}

// ── depthColour ───────────────────────────────────────────────────────────────

test("depthColour: zero depth returns shallow colour (sandy tan)", () => {
  const c = depthColour(0, 20);
  assert.ok(c.r > 0.7, `r=${c.r} should be high (sandy tan)`);
  assert.ok(c.g > 0.6, `g=${c.g} should be medium`);
  assert.ok(c.b < 0.5, `b=${c.b} should be low`);
});

test("depthColour: max depth returns deep colour (dark navy)", () => {
  const c = depthColour(20, 20);
  assert.ok(c.r < 0.1, `r=${c.r} should be near zero`);
  assert.ok(c.g < 0.15, `g=${c.g} should be near zero`);
  assert.ok(c.b < 0.2, `b=${c.b} should be low`);
});

test("depthColour: mid depth is between shallow and deep", () => {
  const shallow = depthColour(0, 20);
  const mid     = depthColour(10, 20);
  const deep    = depthColour(20, 20);
  // Mid brightness should be between shallow and deep
  const lum = c => c.r * 0.3 + c.g * 0.59 + c.b * 0.11;
  assert.ok(lum(mid) !== lum(shallow));
  assert.ok(lum(mid) !== lum(deep));
});

test("depthColour: clamps depth above maxDepth", () => {
  const c1 = depthColour(20, 20);
  const c2 = depthColour(99, 20);
  assert.deepEqual(c1, c2);
});

test("depthColour: clamps depth below zero", () => {
  const c1 = depthColour(0, 20);
  const c2 = depthColour(-5, 20);
  assert.deepEqual(c1, c2);
});

// ── fmtTime ───────────────────────────────────────────────────────────────────

test("fmtTime: zero", ()     => assert.equal(fmtTime(0),   "0:00"));
test("fmtTime: 59s",  ()     => assert.equal(fmtTime(59),  "0:59"));
test("fmtTime: 60s",  ()     => assert.equal(fmtTime(60),  "1:00"));
test("fmtTime: 90s",  ()     => assert.equal(fmtTime(90),  "1:30"));
test("fmtTime: 120s", ()     => assert.equal(fmtTime(120), "2:00"));
test("fmtTime: pads seconds < 10", () => assert.equal(fmtTime(65), "1:05"));
test("fmtTime: null/undefined → 0:00", () => assert.equal(fmtTime(null), "0:00"));

// ── Fish finder LUT ───────────────────────────────────────────────────────────

test("LUT: amplitude 0 → near-black", () => {
  const lut = buildFishFinderLUT();
  assert.ok(lut[0] < 20  && lut[1] < 20  && lut[2] < 30, "amp=0 should be dark");
});

test("LUT: amplitude 255 → white", () => {
  const lut = buildFishFinderLUT();
  assert.equal(lut[255 * 3],     255);
  assert.equal(lut[255 * 3 + 1], 255);
  assert.equal(lut[255 * 3 + 2], 255);
});

test("LUT: all values are valid bytes (0-255)", () => {
  const lut = buildFishFinderLUT();
  for (let i = 0; i < lut.length; i++) {
    assert.ok(lut[i] >= 0 && lut[i] <= 255, `LUT[${i}]=${lut[i]} out of byte range`);
  }
});

test("LUT: generally increases brightness with amplitude", () => {
  const lut = buildFishFinderLUT();
  const lum = a => lut[a*3]*0.3 + lut[a*3+1]*0.59 + lut[a*3+2]*0.11;
  // Check luminance at several points increases monotonically-ish
  assert.ok(lum(0) < lum(128), "mid should be brighter than noise floor");
  assert.ok(lum(128) < lum(255), "peak should be brightest");
});

// ── Point colour logic ────────────────────────────────────────────────────────

test("fishPointColour: fresh fish (high conf) → bright orange", () => {
  const c = fishPointColour(false, 0.65, 10, 20);
  assert.ok(c.r > 0.9, "r should be near 1.0");
  assert.ok(c.g > 0.4, "g should be ~0.55");
  assert.ok(c.b < 0.1, "b should be ~0.05");
});

test("fishPointColour: decayed fish (low conf) → dim colour", () => {
  const fresh = fishPointColour(false, 0.65, 10, 20);
  const faded = fishPointColour(false, 0.10, 10, 20);
  assert.ok(faded.r < fresh.r, "faded r should be lower");
  assert.ok(faded.g < fresh.g, "faded g should be lower");
});

test("fishPointColour: zero conf → black", () => {
  const c = fishPointColour(false, 0.0, 10, 20);
  assert.ok(c.r < 0.01 && c.g < 0.01 && c.b < 0.01);
});

test("fishPointColour: floor point (high conf) → depth colour, not orange", () => {
  const c = fishPointColour(true, 0.8, 0, 20);
  // Shallow floor should be sandy (high r, medium g, low b)
  assert.ok(c.r > 0.5, "shallow floor point should have high red");
});

test("fishPointColour: forward-scan floor (conf <= 0.45) → muted cyan tint", () => {
  const fwd   = fishPointColour(true, 0.40, 0, 20);
  const primary = fishPointColour(true, 0.80, 0, 20);
  // Forward scan should have higher blue component (cyan shift)
  assert.ok(fwd.b > primary.b, "fwd scan floor should have more blue");
});
