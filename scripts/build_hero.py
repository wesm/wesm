# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow==12.3.0", "numpy==2.5.3"]
# ///
"""Render assets/kenn-hero.svg: a meditating pixel-art Wes orbited by Kenn projects.

The figure comes from assets/wes-lotus.png, a 16-bit style illustration on a flat
backdrop. It is resampled onto its own pixel grid, cut out, and embedded as one PNG.

Usage: uv run scripts/build_hero.py
"""

import base64
import io
from collections import deque
from collections.abc import Iterator
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "assets" / "wes-lotus.png"
OUTPUT = ROOT / "assets" / "kenn-hero.svg"

# Source pixels per art pixel, measured from the illustration.
SOURCE_PITCH = 6.2
# RGB distance from the backdrop color within which a pixel counts as backdrop.
BACKDROP_TOLERANCE = 40
# Bluish outline pixels this close to the backdrop are antialiasing blend, not figure.
FRINGE_TOLERANCE = 70
# Smaller backdrop-colored regions are kept; they are details such as eye pixels.
MIN_BACKDROP_REGION = 8
# Art-grid box around the eyes, used for the blink animation.
EYE_ROWS, EYE_COLS = range(32, 35), range(88, 111)
LASH = "#7c330d"
SPRITE_HEIGHT = 330
UPSCALE = 4

# Colors from the kenn.io site theme.
BG, BG_SOFT = "#0a0e14", "#0c1117"
FG, FG_MUTED, FG_DIM = "#e3e8ee", "#9aafc0", "#5e7385"
LINE = "rgba(227,232,238,0.13)"
SAGE, SAGE_DIM, TAN = "#6dac9b", "#4f8475", "#d9a880"
MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,'Cascadia Mono',Consolas,monospace"

# Ordered as on kenn.io.
PROJECTS = [
    ("AgentsView", "what your agents did, what it cost"),
    ("roborev", "local, continuous code review"),
    ("Ghosthub", "all your multiplexers, one terminal"),
    ("msgvault", "keep and search your messages"),
    ("Kenn Forge", "agent workspace with forge sync"),
    ("kata", "the issue tracker inside the loop"),
    ("Docbank", "agent-native document vault"),
    ("kwt", "Git worktrees for people and agents"),
    ("BenchDB", "performance results, in context"),
]

VIEW_W, VIEW_H = 1200, 600
CENTER = (600, 364)
ROW_GAP = 84
NODE_X = (380, 820)
RING_R = 200


def hexrgb(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)])


def sample_grid(path: Path, pitch: float) -> np.ndarray:
    """Resample an illustration onto its art-pixel grid, taking the median of each cell."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    ny, nx = int(im.shape[0] / pitch), int(im.shape[1] / pitch)
    out = np.zeros((ny, nx, 3), int)
    for j, i in np.ndindex(ny, nx):
        y0, x0 = int(j * pitch + 1), int(i * pitch + 1)
        y1, x1 = int((j + 1) * pitch), int((i + 1) * pitch)
        out[j, i] = np.median(im[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    return out


def neighbors(y: int, x: int, shape: tuple[int, ...]) -> Iterator[tuple[int, int]]:
    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
        if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
            yield ny, nx


def large_regions(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Keep the 4-connected regions of mask that have at least min_size pixels."""
    seen = np.zeros_like(mask)
    out = np.zeros_like(mask)
    for start in zip(*np.nonzero(mask), strict=True):
        if seen[start]:
            continue
        seen[start] = True
        region, queue = [], deque([start])
        while queue:
            y, x = queue.popleft()
            region.append((y, x))
            for nb in neighbors(y, x, mask.shape):
                if mask[nb] and not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
        if len(region) >= min_size:
            out[tuple(zip(*region, strict=True))] = True
    return out


def figure_mask(grid: np.ndarray) -> np.ndarray:
    """Mark everything except the flat backdrop, including gaps enclosed by the arms."""
    border = np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]])
    distance = np.linalg.norm(grid - np.median(border, axis=0), axis=-1)
    backdrop = large_regions(distance < BACKDROP_TOLERANCE, MIN_BACKDROP_REGION)
    bluish = (grid[..., 2] > grid[..., 0]) & (distance < FRINGE_TOLERANCE)
    for _ in range(2):
        padded = np.pad(backdrop, 1)
        h, w = backdrop.shape
        touching = np.zeros_like(backdrop)
        for dy, dx in np.ndindex(3, 3):
            touching |= padded[dy : dy + h, dx : dx + w]
        backdrop |= touching & bluish
    return ~backdrop


def closed_eyes(grid: np.ndarray, figure: np.ndarray) -> np.ndarray:
    """Pixels that paint the eyes shut: skin from below each eye, with a lash line."""
    lids = np.zeros((*grid.shape[:2], 4), np.uint8)
    lash_row = EYE_ROWS[-1]
    for y in EYE_ROWS:
        for x in EYE_COLS:
            r, _, b = grid[y, x]
            if figure[y, x] and (b >= r - 20 or r < 0x40):
                color = hexrgb(LASH) if y == lash_row else grid[lash_row + 2, x]
                lids[y, x] = (*color, 255)
    return lids


def build_sprite() -> tuple[np.ndarray, np.ndarray]:
    """Return the cut-out figure and its closed-eye overlay, cropped to the figure."""
    grid = sample_grid(SOURCE, SOURCE_PITCH)
    figure = figure_mask(grid)
    sprite = np.zeros((*grid.shape[:2], 4), np.uint8)
    sprite[figure] = np.c_[grid[figure], np.full(figure.sum(), 255)]
    lids = closed_eyes(grid, figure)
    ys, xs = np.nonzero(figure)
    crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return sprite[crop], lids[crop]


def sprite_image(pixels: np.ndarray, css_class: str = "") -> str:
    """Embed RGBA pixels as one PNG, pre-scaled with nearest neighbor so edges stay hard.

    A single image avoids the hairline seams that browsers draw between adjacent
    vector rectangles when the SVG is scaled.
    """
    img = Image.fromarray(pixels)
    img = img.resize((img.width * UPSCALE, img.height * UPSCALE), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    cls = f' class="{css_class}"' if css_class else ""
    return (
        f'<image{cls} width="{pixels.shape[1]}" height="{pixels.shape[0]}" '
        f'style="image-rendering:pixelated" href="data:image/png;base64,{data}"/>'
    )


def connector_end(node: tuple[int, int]) -> tuple[float, float]:
    cx, cy = CENTER
    dx, dy = node[0] - cx, node[1] - cy
    dist = (dx * dx + dy * dy) ** 0.5
    return cx + dx / dist * RING_R, cy + dy / dist * RING_R


def row_y(side: int, row: int) -> int:
    """Rows alternate left and right; each column is centered on the sprite."""
    count = len(PROJECTS[side::2])
    return round(CENTER[1] + (row - (count - 1) / 2) * ROW_GAP)


def project_nodes() -> str:
    parts = []
    for i, (name, tagline) in enumerate(PROJECTS):
        side = i % 2
        row = row_y(side, i // 2)
        nx = NODE_X[side]
        rx, ry = connector_end((nx, row))
        delay = f"animation-delay:{i * 0.55:.2f}s"
        line = f"M{rx:.1f} {ry:.1f}L{nx} {row}"
        anchor, tx = ("end", nx - 24) if side == 0 else ("start", nx + 24)
        parts.append(
            f'<path class="wire" d="{line}"/>'
            f'<path class="pellet" style="{delay}" pathLength="100" d="{line}"/>'
            f'<rect class="node" style="{delay}" x="{nx - 5}" y="{row - 5}" '
            'width="10" height="10"/>'
            f'<text x="{tx}" y="{row - 3}" text-anchor="{anchor}" class="name">'
            f"{escape(name)}</text>"
            f'<text x="{tx}" y="{row + 20}" text-anchor="{anchor}" class="tag">'
            f"{escape(tagline)}</text>"
        )
    return "\n".join(parts)


def sparkles() -> str:
    spots = [(470, 170, FG, 0.0), (742, 214, TAN, 1.3), (430, 470, SAGE, 2.1),
             (790, 452, FG, 0.7), (612, 158, SAGE, 2.8), (520, 560, TAN, 1.8),
             (700, 572, FG, 3.4)]  # fmt: skip
    out = []
    for x, y, color, delay in spots:
        d = f"M{x - 1} {y - 5}h2v4h4v2h-4v4h-2v-4h-4v-2h4z"
        out.append(f'<path class="spark" style="animation-delay:{delay}s" fill="{color}" d="{d}"/>')
    return "\n".join(out)


STYLE = f"""
text {{ font-family: {MONO}; }}
.word {{ font-size: 36px; font-weight: 600; fill: {FG}; letter-spacing: -0.5px; }}
.lede {{ font-size: 17px; fill: {FG_MUTED}; }}
.prompt {{ fill: {SAGE}; }}
.who {{ font-size: 16px; fill: {FG}; }}
.whom {{ font-size: 15px; fill: {FG_DIM}; }}
.url {{ fill: {SAGE}; }}
.name {{ font-size: 19px; font-weight: 600; fill: {FG}; }}
.tag {{ font-size: 14.5px; fill: {FG_MUTED}; }}
.rule {{ stroke: {LINE}; stroke-width: 1; }}
.wire {{ stroke: {FG_DIM}; stroke-opacity: 0.6; stroke-width: 1.5; stroke-dasharray: 3 5;
  fill: none; }}
.pellet {{ stroke: {SAGE}; stroke-width: 3; fill: none; stroke-dasharray: 8 200;
  stroke-dashoffset: 8;
  animation: travel 5.5s linear infinite; }}
.node {{ fill: {BG}; stroke: {SAGE_DIM}; stroke-width: 1.5;
  animation: ping 5.5s linear infinite; }}
.ring {{ fill: none; transform-box: fill-box; transform-origin: center; }}
.ring-a {{ stroke: {SAGE_DIM}; stroke-width: 1.5; stroke-dasharray: 2 12;
  animation: spin 90s linear infinite; }}
.ring-b {{ stroke: {TAN}; stroke-opacity: 0.45; stroke-width: 1.5; stroke-dasharray: 1 18;
  animation: spin 60s linear infinite reverse; }}
.aura {{ animation: breathe 6s ease-in-out infinite; }}
.float {{ animation: float 6s ease-in-out infinite; }}
.shadow {{ transform-box: fill-box; transform-origin: center;
  animation: shadow 6s ease-in-out infinite; }}
.lid {{ opacity: 0; animation: blink 7s steps(1) infinite; }}
.spark {{ opacity: 0; animation: twinkle 4.2s ease-in-out infinite; }}
.cursor {{ fill: {SAGE}; animation: cursor 1.1s steps(1) infinite; }}
@keyframes travel {{
  0% {{ stroke-dashoffset: 8; }}
  46%, 100% {{ stroke-dashoffset: -100; }}
}}
@keyframes ping {{
  0%, 42%, 70%, 100% {{ fill: {BG}; stroke: {SAGE_DIM}; }}
  48% {{ fill: {SAGE}; stroke: {SAGE}; }}
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes breathe {{ 0%, 100% {{ opacity: 0.55; }} 50% {{ opacity: 1; }} }}
@keyframes float {{
  0%, 100% {{ transform: translateY(0); }}
  50% {{ transform: translateY(-10px); }}
}}
@keyframes shadow {{
  0%, 100% {{ transform: scale(1); opacity: 0.6; }}
  50% {{ transform: scale(0.82); opacity: 0.35; }}
}}
@keyframes blink {{ 0%, 96% {{ opacity: 0; }} 97%, 99% {{ opacity: 1; }} }}
@keyframes twinkle {{ 0%, 100% {{ opacity: 0; }} 50% {{ opacity: 0.85; }} }}
@keyframes cursor {{ 0%, 49% {{ opacity: 1; }} 50%, 100% {{ opacity: 0; }} }}
@media (prefers-reduced-motion: reduce) {{
  .pellet, .node, .ring, .aura, .float, .shadow, .lid, .spark, .cursor {{ animation: none; }}
  .spark {{ opacity: 0.6; }}
}}
"""


def background() -> str:
    return f"""<defs>
  <radialGradient id="glow-sage" cx="20%" cy="-10%" r="60%">
    <stop offset="0" stop-color="{SAGE}" stop-opacity="0.14"/>
    <stop offset="1" stop-color="{SAGE}" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="glow-tan" cx="105%" cy="110%" r="60%">
    <stop offset="0" stop-color="{TAN}" stop-opacity="0.09"/>
    <stop offset="1" stop-color="{TAN}" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="aura">
    <stop offset="0" stop-color="{SAGE}" stop-opacity="0.3"/>
    <stop offset="0.6" stop-color="{SAGE}" stop-opacity="0.06"/>
    <stop offset="1" stop-color="{SAGE}" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="shadow">
    <stop offset="0" stop-color="#000" stop-opacity="0.85"/>
    <stop offset="1" stop-color="#000" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="sheet" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="{BG_SOFT}"/>
    <stop offset="1" stop-color="{BG}"/>
  </linearGradient>
  <pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">
    <rect x="11.5" y="11.5" width="1" height="1" fill="{FG}" fill-opacity="0.12"/>
  </pattern>
  <clipPath id="card"><rect width="{VIEW_W}" height="{VIEW_H}" rx="14"/></clipPath>
</defs>
<g clip-path="url(#card)">
  <rect width="{VIEW_W}" height="{VIEW_H}" fill="url(#sheet)"/>
  <rect width="{VIEW_W}" height="{VIEW_H}" fill="url(#glow-sage)"/>
  <rect width="{VIEW_W}" height="{VIEW_H}" fill="url(#glow-tan)"/>
  <rect y="124" width="{VIEW_W}" height="{VIEW_H - 124}" fill="url(#dots)"/>
</g>
<rect x="0.5" y="0.5" width="{VIEW_W - 1}" height="{VIEW_H - 1}" rx="14"
  fill="none" stroke="{LINE}"/>"""


def header() -> str:
    right = VIEW_W - 48
    return f"""<text x="48" y="66" class="word">kenn software</text>
<text x="48" y="100" class="lede"><tspan class="prompt">&gt; </tspan
  >agentic systems for what comes next.<tspan class="cursor"> &#x2588;</tspan></text>
<text x="{right}" y="62" text-anchor="end" class="who">wes mckinney</text>
<text x="{right}" y="88" text-anchor="end" class="whom"
  >founder · <tspan class="url">kenn.io</tspan></text>
<line x1="0" y1="124" x2="{VIEW_W}" y2="124" class="rule"/>"""


def center_piece(sprite: np.ndarray, lids: np.ndarray) -> str:
    cx, cy = CENTER
    scale = SPRITE_HEIGHT / sprite.shape[0]
    sx, sy = cx - sprite.shape[1] * scale / 2, cy - SPRITE_HEIGHT / 2 - 6
    shadow_y = sy + SPRITE_HEIGHT + 22
    return f"""<circle class="aura" cx="{cx}" cy="{cy}" r="{RING_R + 30}" fill="url(#aura)"/>
<circle class="ring ring-a" cx="{cx}" cy="{cy}" r="{RING_R}"/>
<circle class="ring ring-b" cx="{cx}" cy="{cy}" r="{RING_R - 26}"/>
<ellipse class="shadow" cx="{cx}" cy="{shadow_y:.1f}" rx="150" ry="14" fill="url(#shadow)"/>
<g class="float">
  <g transform="translate({sx:.1f} {sy:.1f}) scale({scale:.4f})">
    {sprite_image(sprite)}
    {sprite_image(lids, "lid")}
  </g>
</g>"""


def render() -> str:
    sprite, lids = build_sprite()
    title = "Wes McKinney at Kenn Software"
    desc = (
        "A 16-bit Wes McKinney floats in lotus pose, orbited by Kenn Software projects: "
        + ", ".join(f"{n} ({t})" for n, t in PROJECTS)
        + "."
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW_W} {VIEW_H}"
  width="{VIEW_W}" height="{VIEW_H}" role="img" aria-labelledby="t d">
<title id="t">{escape(title)}</title>
<desc id="d">{escape(desc)}</desc>
<style>{STYLE}</style>
{background()}
{header()}
{project_nodes()}
{sparkles()}
{center_piece(sprite, lids)}
</svg>
"""


def main() -> None:
    OUTPUT.write_text(render())
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
