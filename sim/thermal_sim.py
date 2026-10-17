"""thermal_sim.py - preview what the pocket-cam will print, without wasting paper.

Runs a photo through a Python port of pocket-cam/include/imgproc.h (same variants as
sweep_table.h), then simulates the 58mm thermal print itself:

  * camera:  center square crop, grayscale, OV2640-ish sensor noise (and the 4-frame average)
  * head:    384 square heater elements at 0.125 mm pitch (203 dpi)
  * heat:    ESC 7 heat time scales the energy per dot; each element keeps some heat into
             the next row (vertical smear / darker dense areas); the head cools during the
             pause between 24-row GS v 0 bands
  * paper:   heat spreads sideways in the paper, the color former has a threshold, so lone
             dots come out round and smaller while dense dots merge (dot gain); black is
             not pure black and the paper is not pure white

Usage:
  python sim/thermal_sim.py photo.jpg                # grid of all variants + sweep strip
  python sim/thermal_sim.py photo.jpg --variant 5    # one variant, full size, high-res
  python sim/thermal_sim.py --list                   # list the variants
  python sim/thermal_sim.py                          # built-in test scene

Outputs go to sim/out/. Physical constants are rough guesses: tune --dot-gain, --history,
--ink etc. until a simulated strip matches a real one, then trust it for the rest.
"""
import argparse
import os
import re
import sys
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from numba import njit
except ImportError:  # works without numba, just slower
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE = os.path.join(HERE, "..", "pocket-cam", "include")

PRINT_W = PRINT_H = 384
SRC_SIDE = 600
STRIP_H = 160
STRIP_Y0 = (PRINT_H - STRIP_H) // 2
BAND_ROWS = 24
DOT_MM = 0.125
PAPER_MM = 57.0  # 2 1/4" roll


# ---------------------------------------------------------------- variants (sweep_table.h)
@dataclass
class Variant:
    name: str = "BASE"
    dither: str = "FS"
    resize: str = "box"
    blur: int = 0        # 0 off, 1 gaussian 3x3, 2 median 3x3
    usm: float = 0.0
    clip: float = 0.02   # negative = skip auto-levels
    eq: bool = False
    clahe: bool = False
    gamma: float = 1.0
    avg4: bool = False
    heat: int = 250      # ESC 7 heat time, 0..255 (one byte, so 255 is the hard maximum)
    coarse: int = 1
    density: int = -1    # DC2 # print density 0..31 (50% + 5% each), -1 = leave the printer's default


BASE = Variant()

# Names used in the UI and on the printed label
DITHERS = {
    "THRESH": "threshold", "BAYER4": "Bayer 4x4", "BAYER8": "Bayer 8x8", "CLUSTER8": "clustered dot",
    "BLUE": "blue noise", "FS": "Floyd-Steinberg", "FS_SERP": "Floyd-Steinberg serp.",
    "ATKINSON": "Atkinson", "SIERRA_LITE": "Sierra Lite", "BURKES": "Burkes", "STUCKI": "Stucki",
    "JJN": "Jarvis-Judice-Ninke",
}
RESIZES = ["nearest", "box", "bilinear"]
BLURS = ["off", "gaussian 3x3", "median 3x3"]


def density_pct(d):
    return 100 if d < 0 else 50 + 5 * d


def sweep_variants(extra=False):
    b = BASE
    v = [
        replace(b, name="THRESHOLD", dither="THRESH"),
        replace(b, name="BAYER 8x8", dither="BAYER8"),
        replace(b, name="CLUSTERED DOT", dither="CLUSTER8"),
        replace(b, name="BLUE NOISE", dither="BLUE"),
        replace(b, name="FLOYD-STEINBERG", dither="FS"),
        replace(b, name="FS SERPENTINE", dither="FS_SERP"),
        replace(b, name="ATKINSON", dither="ATKINSON"),
        replace(b, name="SIERRA LITE", dither="SIERRA_LITE"),
        replace(b, name="UNSHARP 0.5", usm=0.5),
        replace(b, name="UNSHARP 1.0", usm=1.0),
        replace(b, name="GAMMA 0.8 (LIGHTER)", gamma=0.8),
        replace(b, name="CLAHE LOCAL CONTRAST", clahe=True, clip=-1.0),
        replace(b, name="4-FRAME AVERAGE", avg4=True),
        replace(b, name="HEAT 180", heat=180),
        replace(b, name="HEAT 255", heat=255),
    ]
    if extra:
        v += [
            replace(b, name="BAYER 4x4", dither="BAYER4"),
            replace(b, name="BURKES", dither="BURKES"),
            replace(b, name="STUCKI", dither="STUCKI"),
            replace(b, name="JARVIS-JUDICE-NINKE", dither="JJN"),
            replace(b, name="RESIZE NEAREST", resize="nearest"),
            replace(b, name="RESIZE BILINEAR", resize="bilinear"),
            replace(b, name="MEDIAN 3x3 DENOISE", blur=2),
            replace(b, name="BLUR 3x3", blur=1),
            replace(b, name="LEVELS CLIP 0%", clip=0.0),
            replace(b, name="LEVELS CLIP 5%", clip=0.05),
            replace(b, name="HIST EQUALIZE", eq=True, clip=-1.0),
            replace(b, name="COARSE DOTS 2x2", coarse=2),
        ]
    return v


# ---------------------------------------------------------------- imgproc.h port
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]])
CLUSTER8 = np.array([
    [24, 10, 12, 26, 35, 47, 49, 37], [8, 0, 2, 14, 45, 59, 61, 51],
    [22, 6, 4, 16, 43, 57, 63, 53], [30, 20, 18, 28, 33, 41, 55, 39],
    [34, 46, 48, 36, 25, 11, 13, 27], [44, 58, 60, 50, 9, 1, 3, 15],
    [42, 56, 62, 52, 23, 7, 5, 17], [32, 40, 54, 38, 31, 21, 19, 29]])

# (dx, dy, w) taps and divisor, same as imgproc.h
TAPS = {
    "FS": ([(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16),
    "ATKINSON": ([(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)], 8),
    "SIERRA_LITE": ([(1, 0, 2), (-1, 1, 1), (0, 1, 1)], 4),
    "BURKES": ([(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)], 32),
    "STUCKI": ([(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2),
                (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)], 42),
    "JJN": ([(1, 0, 7), (2, 0, 5), (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
             (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)], 48),
}


def load_blue_noise():
    path = os.path.join(INCLUDE, "blue_noise.h")
    try:
        with open(path) as f:
            body = f.read().split("{", 1)[1]
        vals = [int(x) for x in re.findall(r"\d+", body)][:4096]
        if len(vals) == 4096:
            return np.array(vals, dtype=np.int32).reshape(64, 64)
    except (OSError, IndexError):
        pass
    print("warning: could not read blue_noise.h, using random thresholds", file=sys.stderr)
    return np.random.default_rng(0).integers(0, 256, (64, 64))


@njit(cache=True)
def _resize(src, W, H, mode):
    S = src.shape[0]
    dst = np.zeros((H, W), np.uint8)
    for y in range(H):
        for x in range(W):
            if mode == 0:  # nearest
                v = src[((2 * y + 1) * S) // (2 * H), ((2 * x + 1) * S) // (2 * W)]
            elif mode == 1:  # box
                sx0 = (x * S) // W
                sx1 = max(sx0 + 1, ((x + 1) * S) // W)
                sy0 = (y * S) // H
                sy1 = max(sy0 + 1, ((y + 1) * S) // H)
                s = 0
                for sy in range(sy0, sy1):
                    for sx in range(sx0, sx1):
                        s += src[sy, sx]
                n = (sx1 - sx0) * (sy1 - sy0)
                v = (s + n // 2) // n
            else:  # bilinear
                fx = (x + 0.5) * S / W - 0.5
                fy = (y + 0.5) * S / H - 0.5
                x0 = int(np.floor(fx))
                y0 = int(np.floor(fy))
                ax = fx - x0
                ay = fy - y0
                x1 = min(max(x0 + 1, 0), S - 1)
                y1 = min(max(y0 + 1, 0), S - 1)
                x0 = min(max(x0, 0), S - 1)
                y0 = min(max(y0, 0), S - 1)
                top = src[y0, x0] * (1 - ax) + src[y0, x1] * ax
                bot = src[y1, x0] * (1 - ax) + src[y1, x1] * ax
                v = int(top * (1 - ay) + bot * ay + 0.5)
            dst[y, x] = min(max(v, 0), 255)
    return dst


@njit(cache=True)
def _error_diffuse(g, taps, div, serpentine):
    H, W = g.shape
    PAD = 2
    rows = np.zeros((3, W + 2 * PAD), np.int32)
    bits = np.zeros((H, W), np.uint8)
    for y in range(H):
        cur = y % 3
        rev = serpentine and (y & 1) == 1
        for i in range(W):
            x = W - 1 - i if rev else i
            v = int(g[y, x]) + rows[cur, x + PAD]
            q = 0 if v < 128 else 255
            err = v - q
            if q == 0:
                bits[y, x] = 1
            for t in range(taps.shape[0]):
                dx = -taps[t, 0] if rev else taps[t, 0]
                # C integer division truncates toward zero
                rows[(y + taps[t, 1]) % 3, x + dx + PAD] += int(err * taps[t, 2] / div)
        rows[cur, :] = 0
    return bits


def blur3(img):
    p = np.pad(img.astype(np.int32), 1, mode="edge")
    t = (p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:] + 2) >> 2
    return ((t[:-2] + 2 * t[1:-1] + t[2:] + 2) >> 2)[:, :].astype(np.uint8)


def median3(img):
    p = np.pad(img, 1, mode="edge")
    H, W = img.shape
    stack = np.stack([p[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)])
    return np.median(stack, axis=0).astype(np.uint8)


def unsharp(img, amount):
    b = blur3(img).astype(np.float32)
    v = img + amount * (img.astype(np.float32) - b)
    return np.clip((v + 0.5).astype(np.int32), 0, 255).astype(np.uint8)


def auto_levels(img, clip_frac):
    n = img.size
    hist = np.bincount(img.ravel(), minlength=256)
    clip = int(n * clip_frac)
    lo, acc = 0, 0
    while lo < 255 and acc + hist[lo] <= clip:
        acc += hist[lo]
        lo += 1
    hi, acc = 255, 0
    while hi > 0 and acc + hist[hi] <= clip:
        acc += hist[hi]
        hi -= 1
    if hi - lo < 16:
        lo = min(lo, 239)
        hi = lo + 16
    i = np.arange(256)
    lut = np.clip(np.trunc((i - lo) * 255 / (hi - lo)), 0, 255).astype(np.uint8)
    return lut[img], lo, hi


def equalize(img):
    n = img.size
    hist = np.bincount(img.ravel(), minlength=256)
    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(hist)[0][0]]
    if n == cdf_min:
        return img
    lut = np.clip(((cdf - cdf_min) * 255) // (n - cdf_min), 0, 255).astype(np.uint8)
    return lut[img]


def clahe(img, clip_factor=3.0):
    T = 4
    H, W = img.shape
    tw, th = W // T, H // T
    area = tw * th
    luts = np.zeros((T, T, 256), np.float32)
    for ty in range(T):
        for tx in range(T):
            hist = np.bincount(img[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw].ravel(), minlength=256)
            limit = max(1, int(clip_factor * area / 256.0))
            excess = int(np.maximum(hist - limit, 0).sum())
            hist = np.minimum(hist, limit)
            hist = hist + excess // 256 + (np.arange(256) < excess % 256)
            luts[ty, tx] = np.minimum(255, (np.cumsum(hist) * 255) // area)
    fy = (np.arange(H) + 0.5) / th - 0.5
    fx = (np.arange(W) + 0.5) / tw - 0.5
    y0, x0 = np.floor(fy).astype(int), np.floor(fx).astype(int)
    wy, wx = (fy - y0)[:, None], (fx - x0)[None, :]
    ya, yb = np.clip(y0, 0, T - 1)[:, None], np.clip(y0 + 1, 0, T - 1)[:, None]
    xa, xb = np.clip(x0, 0, T - 1)[None, :], np.clip(x0 + 1, 0, T - 1)[None, :]
    top = luts[ya, xa, img] * (1 - wx) + luts[ya, xb, img] * wx
    bot = luts[yb, xa, img] * (1 - wx) + luts[yb, xb, img] * wx
    return np.clip((top * (1 - wy) + bot * wy + 0.5).astype(np.int32), 0, 255).astype(np.uint8)


def apply_gamma(img, gamma):
    lut = np.clip((255.0 * (np.arange(256) / 255.0) ** gamma + 0.5).astype(np.int32), 0, 255).astype(np.uint8)
    return lut[img]


def dither(name, g, blue):
    H, W = g.shape
    yy, xx = np.mgrid[0:H, 0:W]
    if name == "THRESH":
        thr = 128
    elif name == "BAYER4":
        thr = BAYER4[yy & 3, xx & 3] * 16 + 8
    elif name == "BAYER8":
        thr = BAYER8[yy & 7, xx & 7] * 4 + 2
    elif name == "CLUSTER8":
        thr = CLUSTER8[yy & 7, xx & 7] * 4 + 2
    elif name == "BLUE":
        thr = blue[yy & 63, xx & 63]
    else:
        key = "FS" if name == "FS_SERP" else name
        taps, div = TAPS[key]
        return _error_diffuse(g, np.array(taps, np.int32), div, name == "FS_SERP")
    return (g < thr).astype(np.uint8)


def run_pipeline(v, src, blue):
    """src: SRC_SIDE x SRC_SIDE uint8. Returns (gray, bits) at print size, bits 1 = black."""
    c = max(1, v.coarse)
    W, H = PRINT_W // c, PRINT_H // c
    g = _resize(src, W, H, {"nearest": 0, "box": 1, "bilinear": 2}[v.resize])
    if v.blur == 1:
        g = blur3(g)
    elif v.blur == 2:
        g = median3(g)
    if v.usm > 0:
        g = unsharp(g, v.usm)
    if v.eq:
        g = equalize(g)
    elif v.clahe:
        g = clahe(g)
    if v.clip >= 0:
        g, _, _ = auto_levels(g, v.clip)
    if abs(v.gamma - 1.0) > 0.001:
        g = apply_gamma(g, v.gamma)
    bits = dither(v.dither, g, blue)
    if c > 1:
        bits = np.kron(bits, np.ones((c, c), np.uint8))
        g = np.kron(g, np.ones((c, c), np.uint8))
    return g, bits


# ---------------------------------------------------------------- camera
def load_source(path, noise, seed):
    """Returns (single_frame, four_frame_average) as SRC_SIDE squares, like captureSource()."""
    if path:
        img = Image.open(path).convert("L")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        img = img.resize((SRC_SIDE, SRC_SIDE), Image.LANCZOS)
        clean = np.asarray(img, np.float32)
    else:
        clean = test_scene()
    rng = np.random.default_rng(seed)

    def frame():
        # shot noise (brighter = noisier) + read noise
        sigma = noise * (0.5 + np.sqrt(np.clip(clean, 0, 255) / 255.0))
        return clean + rng.normal(0, 1, clean.shape) * sigma

    one = frame()
    avg = (one + frame() + frame() + frame()) / 4
    to8 = lambda a: np.clip(a + 0.5, 0, 255).astype(np.uint8)
    return to8(one), to8(avg)


def test_scene():
    """Synthetic scene: tone ramp, soft spheres, fine lines and text."""
    S = SRC_SIDE
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    img = 60 + 150 * (x / S) * 0.6 + 60 * (y / S)
    for cx, cy, r, base in [(170, 330, 120, 200), (430, 400, 100, 120), (450, 170, 70, 230)]:
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        shade = base * np.clip(1 - ((x - cx + r * 0.4) ** 2 + (y - cy + r * 0.4) ** 2) / (2.2 * r * r), 0.15, 1)
        img = np.where(d < r, shade, img)
    img[0:60, :] = np.repeat(np.linspace(0, 255, S)[None, :], 60, 0)  # ramp along the top
    for i in range(10):
        img[520:590, 30 + i * 22:30 + i * 22 + 2 + i // 3] = 15
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    d.text((280, 530), "pocket-cam", fill=20, font=font(40))
    return np.asarray(pil, np.float32)


# ---------------------------------------------------------------- thermal print physics
@dataclass
class PrintModel:
    ss: int = 6              # render sub-pixels per dot
    # Energy per dot vs ESC 7 heat time, calibrated to real prints: heat 250 is the sweet spot
    # (well-formed dots that only just merge); lower heat gives smaller, fainter dots.
    energy_250: float = 0.91       # energy at heat 250
    energy_per_heat: float = 0.0021
    dot_gain: float = 0.42   # sideways heat spread in the paper, in dot pitches (sigma)
    threshold: float = 0.42  # energy where the color former is half developed
    softness: float = 0.09   # width of the development curve (smaller = crisper dots)
    history: float = 0.30    # fraction of an element's heat carried into the next row
    history_decay: float = 0.45
    tail: float = 0.25       # residual heat that develops the row AFTER a burned dot
    band_cool: bool = True   # head cools in the pause between 24-row bands
    sag: float = 0.0         # energy lost on a fully black row (weak battery / thin wires)
    ink: tuple = (52, 50, 60)       # developed color, phenol-free paper is a cool dark gray
    paper: tuple = (244, 242, 236)  # paper base color
    grain: float = 3.0       # paper texture strength


@lru_cache(maxsize=8)
def _paper_texture(shape, ss, seed):
    """Fine grain and large-scale mottling, cached because live view reuses them every frame."""
    rng = np.random.default_rng(seed)
    grain = gaussian_filter(rng.normal(0, 1, shape).astype(np.float32), 0.8)
    mottle = gaussian_filter(rng.normal(0, 1, shape).astype(np.float32), 6 * ss) * 6
    return grain, mottle


def dot_energy(heat, density, m):
    """Relative energy per dot for an ESC 7 heat time and DC2 # density (-1 = printer default = 100%)."""
    e = m.energy_250 + m.energy_per_heat * (heat - 250)
    return max(e, 0.0) * density_pct(density) / 100.0


def simulate_print(bits, heat, m, seed=0, density=-1):
    """bits: H x W, 1 = burn. Returns an RGB uint8 image at m.ss pixels per dot."""
    H, W = bits.shape
    b = bits.astype(np.float32)
    e = dot_energy(heat, density, m)

    # energy delivered per element per row, with heat carried from the previous rows
    dep = np.zeros_like(b)
    temp = np.zeros(W, np.float32)
    for y in range(H):
        if m.band_cool and y % BAND_ROWS == 0:
            temp *= 0.3
        row_e = e * (1.0 - m.sag * b[y].mean())
        dep[y] = b[y] * row_e * (1.0 + m.history * temp) + m.tail * temp * e
        temp = temp * m.history_decay + b[y]

    # square heater footprint, then lateral heat spread in the paper
    s = m.ss
    field = np.kron(dep, np.ones((s, s), np.float32))
    field = gaussian_filter(field, sigma=m.dot_gain * s, mode="constant")

    # color former: thresholded, saturating response
    dens = 1.0 / (1.0 + np.exp(-(field - m.threshold) / m.softness))
    dens -= 1.0 / (1.0 + np.exp(m.threshold / m.softness))  # zero energy -> zero density
    dens = np.clip(dens, 0, 1)

    grain, mottle = _paper_texture(dens.shape, s, seed)
    grain, mottle = grain * m.grain, mottle * m.grain
    ink, paper = np.array(m.ink, np.float32), np.array(m.paper, np.float32)
    rgb = paper + (ink - paper) * dens[..., None]
    rgb += (grain + mottle * (1 - dens))[..., None]
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- layout helpers
_fonts = {}


def font(size):
    if size not in _fonts:
        try:
            _fonts[size] = ImageFont.truetype("consola.ttf", size)
        except OSError:
            try:
                _fonts[size] = ImageFont.truetype("DejaVuSansMono.ttf", size)
            except OSError:
                _fonts[size] = ImageFont.load_default()
    return _fonts[size]


LABEL_CHARS = 42  # font B characters per line on the 58 mm printer


def contrast_name(v):
    return "hist. equalize" if v.eq else ("CLAHE" if v.clahe else "none")


def label_lines(v, idx=None, total=None):
    """Every setting that shaped the print, word-wrapped to the printer's font B line width."""
    head = f"{idx + 1:02d}/{total:02d} {v.name}" if idx is not None else v.name
    levels = "off" if v.clip < 0 else f"{v.clip * 100:g}% clip"
    density = "default" if v.density < 0 else f"{density_pct(v.density)}%"
    parts = [f"dither {DITHERS.get(v.dither, v.dither)}", f"dot {v.coarse}x{v.coarse}",
             f"heat {v.heat}", f"density {density}", f"resize {v.resize}", f"blur {BLURS[v.blur]}",
             f"unsharp {v.usm:.1f}", f"levels {levels}", f"contrast {contrast_name(v)}",
             f"gamma {v.gamma:.2f}", f"4-frame avg {'on' if v.avg4 else 'off'}"]
    lines, cur = [head[:LABEL_CHARS]], ""
    for p in parts:
        if cur and len(cur) + 3 + len(p) > LABEL_CHARS:
            lines.append(cur)
            cur = p
        else:
            cur = f"{cur} | {p}" if cur else p
    return lines + [cur]


def on_paper(print_rgb, m, text_lines=(), seed=0):
    """Center a print on a 57mm-wide strip of paper, with the printer's text label above."""
    s = m.ss
    paper_w = int(round(PAPER_MM / DOT_MM * s))
    lh = int(3.0 / DOT_MM * s / 1.25) if text_lines else 0  # ~font B line height
    top = len(text_lines) * lh + 2 * s * 4
    canvas = Image.new("RGB", (paper_w, top + print_rgb.shape[0] + 8 * s), m.paper)
    x0 = (paper_w - print_rgb.shape[1]) // 2
    if text_lines:
        txt = np.zeros((top // s, PRINT_W), np.uint8)
        t = Image.fromarray(txt)
        d = ImageDraw.Draw(t)
        for i, line in enumerate(text_lines):
            d.text((0, 2 + i * lh // s), line, fill=1, font=font(max(8, lh // s - 3)))
        txt_rgb = simulate_print(np.asarray(t), BASE.heat, m, seed + 1)
        canvas.paste(Image.fromarray(txt_rgb), (x0, 0))
    canvas.paste(Image.fromarray(print_rgb), (x0, top))
    return canvas


def ideal(bits, s):
    img = (1 - np.kron(bits, np.ones((s, s), np.uint8))) * 255
    return Image.fromarray(img.astype(np.uint8)).convert("RGB")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="photo to print (default: built-in test scene)")
    ap.add_argument("--variant", "-v", type=int, action="append",
                    help="variant number(s) to render full size (1-based, see --list)")
    ap.add_argument("--extra", action="store_true", help="include the SWEEP_EXTRA variants")
    ap.add_argument("--list", action="store_true", help="list variants and exit")
    ap.add_argument("--out", default=os.path.join(HERE, "out"), help="output folder")
    ap.add_argument("--noise", type=float, default=4.0, help="camera noise (0 = clean photo)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ss", type=int, default=4, help="render pixels per printer dot")
    ap.add_argument("--dot-gain", type=float, default=PrintModel.dot_gain)
    ap.add_argument("--threshold", type=float, default=PrintModel.threshold)
    ap.add_argument("--history", type=float, default=PrintModel.history)
    ap.add_argument("--tail", type=float, default=PrintModel.tail)
    ap.add_argument("--sag", type=float, default=PrintModel.sag)
    ap.add_argument("--no-band-cool", action="store_true")
    ap.add_argument("--ink", default="52,50,60", help="developed color R,G,B")
    ap.add_argument("--heat", type=int, help="override every variant's heat time (0..255)")
    ap.add_argument("--density", type=int, help="override every variant's print density (0..31)")
    ap.add_argument("--gamma", type=float, help="override every variant's gamma (try 0.6-0.8)")
    args = ap.parse_args()

    variants = sweep_variants(args.extra)
    if args.list:
        for i, v in enumerate(variants):
            print(f"{i + 1:2d}  {v.name}")
        return
    overrides = {k: getattr(args, k) for k in ("heat", "gamma", "density") if getattr(args, k) is not None}
    variants = [replace(v, **overrides) for v in variants]

    m = PrintModel(ss=args.ss, dot_gain=args.dot_gain, threshold=args.threshold, history=args.history,
                   tail=args.tail, sag=args.sag, band_cool=not args.no_band_cool,
                   ink=tuple(int(c) for c in args.ink.split(",")))
    os.makedirs(args.out, exist_ok=True)
    blue = load_blue_noise()
    raw1, raw_avg = load_source(args.image, args.noise, args.seed)

    if args.variant:
        for n in args.variant:
            if not 1 <= n <= len(variants):
                sys.exit(f"variant must be 1..{len(variants)}")
            v = variants[n - 1]
            g, bits = run_pipeline(v, raw_avg if v.avg4 else raw1, blue)
            sim = simulate_print(bits, v.heat, m, args.seed, v.density)
            page = on_paper(sim, m, label_lines(v, n - 1, len(variants)), args.seed)
            side = Image.new("RGB", (page.width * 2, page.height), (255, 255, 255))
            side.paste(page, (0, 0))
            id_img = ideal(bits, m.ss)
            side.paste(id_img, (page.width + (page.width - id_img.width) // 2, page.height - id_img.height - 8 * m.ss))
            path = os.path.join(args.out, f"variant_{n:02d}.png")
            side.save(path)
            print(f"wrote {path}  (left: simulated print, right: the 1-bit data sent to the printer)")
        return

    # grid of full-size simulated prints + one long strip like the hardware sweep
    tiles, strips = [], []
    for i, v in enumerate(variants):
        g, bits = run_pipeline(v, raw_avg if v.avg4 else raw1, blue)
        sim = simulate_print(bits, v.heat, m, args.seed + i, v.density)
        print(f"[{i + 1:02d}/{len(variants)}] {v.name:22s} ink coverage {bits.mean() * 100:4.1f}%")
        tile = Image.fromarray(sim)
        d = ImageDraw.Draw(tile)
        d.rectangle((0, 0, tile.width, 22 + 2 * m.ss), fill=(255, 255, 255))
        d.text((6, 4), f"{i + 1}. {v.name}  heat {v.heat}", fill=(200, 0, 0), font=font(18))
        tiles.append(tile)
        strip_bits = bits[STRIP_Y0:STRIP_Y0 + STRIP_H]
        strips.append(on_paper(simulate_print(strip_bits, v.heat, m, args.seed + i, v.density), m,
                               label_lines(v, i, len(variants)), args.seed + i))

    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    tw, th, gap = tiles[0].width, tiles[0].height, 12
    grid = Image.new("RGB", (cols * (tw + gap) + gap, rows * (th + gap) + gap), (90, 90, 90))
    for i, t in enumerate(tiles):
        grid.paste(t, (gap + (i % cols) * (tw + gap), gap + (i // cols) * (th + gap)))
    grid_path = os.path.join(args.out, "grid.png")
    grid.save(grid_path)

    strip = Image.new("RGB", (strips[0].width, sum(s.height for s in strips)), m.paper)
    y = 0
    for s in strips:
        strip.paste(s, (0, y))
        y += s.height
    strip_path = os.path.join(args.out, "sweep_strip.png")
    strip.save(strip_path)
    print(f"wrote {grid_path}\nwrote {strip_path}\nUse --variant N for a full-size close-up of one.")


if __name__ == "__main__":
    main()
