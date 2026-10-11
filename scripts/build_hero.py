# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow==12.3.0", "numpy==2.5.3", "scipy==1.18.1"]
# ///
"""Render assets/kenn-hero.svg: a meditating pixel-art sprite orbited by Kenn projects.

The head comes from assets/avatar.png (the 16-bit GitHub avatar), resampled onto its
native pixel grid at half resolution. The body is modeled as layered smooth shapes,
shaded as rounded volumes, and reduced to the same grid.

Usage: uv run scripts/build_hero.py
"""

import base64
import io
from collections import deque
from collections.abc import Callable
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
AVATAR = ROOT / "assets" / "avatar.png"
OUTPUT = ROOT / "assets" / "kenn-hero.svg"

AVATAR_PITCH = 4.24 * 2
SPRITE_W, SPRITE_H = 112, 108
CX = SPRITE_W // 2
HEAD_DX, HEAD_DY = CX - 27, 2
PX = 3
UPSCALE = 6
LASH = "#9a4d1e"
# Avatar-grid box around the eyes, used for the blink animation.
EYE_ROWS, EYE_COLS = range(21, 23), range(20, 35)

# Body parts are drawn at SUPER times the sprite grid, shaded as domes lit from the upper
# left, then reduced to the grid. Ramps run from outline to highlight.
SUPER = 4
LIGHT = np.array([-0.5, -0.8, 0.75]) / np.linalg.norm([-0.5, -0.8, 0.75])
AMBIENT = 0.25
SKIN = ("#7e3c14", "#bd6a2e", "#da8b4c", "#f0a866", "#fbc282", "#fdd6a2")
TEE = ("#262833", "#3f4250", "#515465", "#626575", "#747786", "#8a8c99")
PANTS = ("#0c1019", "#1b2333", "#263045", "#323e57", "#41506d", "#556685")
PANTS_FRONT = ("#0f1420", "#222c40", "#2e3a53", "#3b4967", "#4c5d7f", "#61749a")

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
    ("Fotobank", "photo system of record you control"),
    ("BenchDB", "performance results, in context"),
]

VIEW_W, VIEW_H = 1200, 660
CENTER = (600, 364)
ROW_Y = [196, 280, 364, 448, 532]
NODE_X = (380, 820)
RING_R = 200


def hexrgb(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)])


def sample_avatar(path: Path) -> np.ndarray:
    """Resample the avatar onto its pixel grid, taking the median of each cell."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    n = int(im.shape[0] / AVATAR_PITCH)
    out = np.zeros((n, n, 3), int)
    for j in range(n):
        for i in range(n):
            y0, x0 = int(j * AVATAR_PITCH + 1), int(i * AVATAR_PITCH + 1)
            y1, x1 = int((j + 1) * AVATAR_PITCH), int((i + 1) * AVATAR_PITCH)
            out[j, i] = np.median(im[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    return out


def is_background(c: np.ndarray) -> bool:
    r, g, b = c
    return b - r > 35 and g - r > 18


def is_skin(c: np.ndarray) -> bool:
    r, _, b = c
    return r > 150 and r - b > 60


def head_mask(av: np.ndarray, last_row: int) -> np.ndarray:
    """Flood-fill the blue backdrop from the border; keep everything else."""
    n = av.shape[0]
    bg = np.zeros((n, n), bool)
    edge = [(y, x) for y in range(n) for x in range(n) if y in (0, n - 1) or x in (0, n - 1)]
    queue = deque(edge)
    while queue:
        y, x = queue.popleft()
        if not (0 <= y < n and 0 <= x < n) or bg[y, x] or not is_background(av[y, x]):
            continue
        bg[y, x] = True
        queue.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
    keep = ~bg
    for _ in range(2):
        fringe = keep & ~(
            shift(keep, 1, 0) & shift(keep, -1, 0) & shift(keep, 0, 1) & shift(keep, 0, -1)
        )
        bluish = av[..., 2] >= av[..., 0] - 10
        keep &= ~(fringe & bluish)
    keep[last_row + 1 :] = False
    return keep


def shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return m translated so out[y, x] == m[y - dy, x - dx] (zero-filled)."""
    out = np.zeros_like(m)
    h, w = m.shape
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = m[
        max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


type Shape = Callable[[ImageDraw.ImageDraw], None]


def scaled(points) -> list[tuple[float, float]]:
    return [(x * SUPER, y * SUPER) for x, y in points]


def mirrored(points) -> list[tuple[float, float]]:
    """Close a left-half outline with its reflection across the sprite center."""
    return list(points) + [(2 * CX - x, y) for x, y in reversed(points)]


def poly(points) -> Shape:
    return lambda d: d.polygon(scaled(points), fill=1)


def ellipse(cx: float, cy: float, rx: float, ry: float) -> Shape:
    return lambda d: d.ellipse(scaled([(cx - rx, cy - ry), (cx + rx, cy + ry)]), fill=1)


def limb(p0, p1, w0: float, w1: float | None = None) -> Shape:
    """Tapered capsule from p0 (width w0) to p1 (width w1)."""
    w1 = w0 if w1 is None else w1
    (x0, y0), (x1, y1) = p0, p1
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    nx, ny = (y0 - y1) / length, (x1 - x0) / length
    side = [(x0 + nx * w0 / 2, y0 + ny * w0 / 2), (x1 + nx * w1 / 2, y1 + ny * w1 / 2),
            (x1 - nx * w1 / 2, y1 - ny * w1 / 2), (x0 - nx * w0 / 2, y0 - ny * w0 / 2)]  # fmt: skip
    ends = [ellipse(x, y, w / 2, w / 2) for (x, y), w in ((p0, w0), (p1, w1))]
    return union(poly(side), *ends)


def union(*shapes: Shape) -> Shape:
    def draw(d: ImageDraw.ImageDraw) -> None:
        for shape in shapes:
            shape(d)

    return draw


def dome_shade(mask: np.ndarray, radius: float) -> np.ndarray:
    """Lambert shading of a rounded volume whose edges curve away over `radius` pixels."""
    r = radius * SUPER
    d = np.minimum(ndimage.distance_transform_edt(mask), r)
    height = ndimage.gaussian_filter(np.sqrt(r * r - (r - d) ** 2), 1.5)
    gy, gx = np.gradient(height)
    normal = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return AMBIENT + (1 - AMBIENT) * np.clip(normal @ LIGHT, 0, 1)


class Body:
    """Layered body parts; each part added sits in front of the earlier ones."""

    def __init__(self) -> None:
        size = (SPRITE_H * SUPER, SPRITE_W * SUPER)
        self.label = np.zeros(size, int)
        self.shade = np.zeros(size)
        self.parts: list[tuple[tuple[str, ...], float]] = [((), 0.0)]

    def add(self, shape: Shape, ramp: tuple[str, ...], radius: float, texture=0.0) -> None:
        img = Image.new("1", (SPRITE_W * SUPER, SPRITE_H * SUPER), 0)
        shape(ImageDraw.Draw(img))
        mask = np.asarray(img).astype(bool)
        offset = int(1.5 * SUPER)
        cast = shift(mask, offset, offset // 2) & ~mask & (self.label > 0)
        self.shade[cast] *= 0.62
        self.parts.append((ramp, texture))
        self.label[mask] = len(self.parts) - 1
        self.shade[mask] = dome_shade(mask, radius)[mask]

    def downsample(self) -> tuple[np.ndarray, np.ndarray]:
        """Give each sprite pixel its dominant part and that part's mean shade."""
        labels = np.zeros((SPRITE_H, SPRITE_W), int)
        shades = np.zeros((SPRITE_H, SPRITE_W))
        for y, x in np.ndindex(SPRITE_H, SPRITE_W):
            cell = (slice(y * SUPER, (y + 1) * SUPER), slice(x * SUPER, (x + 1) * SUPER))
            block = self.label[cell]
            values, counts = np.unique(block, return_counts=True)
            order = np.argsort(-counts)
            top = values[order[0]]
            if top == 0:
                if len(order) == 1 or counts[order[0]] > 0.55 * SUPER * SUPER:
                    continue
                top = values[order[1]]
            labels[y, x] = top
            shades[y, x] = self.shade[cell][block == top].mean()
        return labels, shades

    def rasterize(self, rng: np.random.Generator) -> np.ndarray:
        labels, shades = self.downsample()
        out = np.zeros((SPRITE_H, SPRITE_W, 4), int)
        for y, x in zip(*np.nonzero(labels), strict=True):
            ramp, texture = self.parts[labels[y, x]]
            level = ramp_level(labels, shades, (y, x), rng, texture)
            out[y, x] = (*hexrgb(ramp[level]), 255)
        return out


def ramp_level(labels, shades, pos, rng, texture: float) -> int:
    """Pick a ramp step: outline on the silhouette, creases where a nearer part overlaps."""
    y, x = pos
    level = int(np.clip(shades[y, x] * 5.2 - 0.4, 1, 5))
    up, down, left, right = (
        labels[y + dy, x + dx] if 0 <= y + dy < SPRITE_H and 0 <= x + dx < SPRITE_W else 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    neighbors = (up, down, left, right)
    if 0 in neighbors:
        return 0 if down == 0 or right == 0 else max(level - 2, 1)
    if any(n > labels[y, x] for n in neighbors):
        return max(level - 2, 0)
    if texture and rng.random() < texture:
        return int(np.clip(level + rng.choice((-1, 1)), 1, 5))
    return level


def meditating_body() -> Body:
    """Cross-legged seat (shins crossed, feet tucked under the knees), hands on the knees."""
    body = Body()
    body.add(ellipse(CX, 86, 22, 7), PANTS, 6)  # lap
    for side in (-1, 1):
        body.add(limb((CX + side * 10, 86), (CX + side * 41, 90), 13, 12), PANTS, 6)  # thigh
    body.add(limb((CX - 40, 91), (CX + 27, 99), 10, 8), PANTS, 4.5)  # rear shin
    body.add(limb((CX + 40, 92), (CX - 27, 100), 10, 8), PANTS_FRONT, 4.5)  # front shin
    torso = [
        (CX - 9, 44),
        (CX - 21, 47),
        (CX - 25, 53),
        (CX - 23, 64),
        (CX - 20, 74),
        (CX - 22, 81),
    ]
    body.add(poly(mirrored(torso)), TEE, 13, texture=0.08)
    hem = [(CX - 20, 77), (CX - 23, 83), (CX - 10, 85)]
    body.add(poly(mirrored(hem)), TEE, 3, texture=0.08)
    body.add(poly([(CX - 7, 44), (CX + 7, 44), (CX, 52)]), SKIN, 3)  # V-neck
    for side in (-1, 1):
        body.add(limb((CX + side * 25, 57), (CX + side * 31, 72), 7.5, 7), SKIN, 3.5)
        sleeve = [(20, 46), (25, 50), (29, 55), (30, 62), (23, 63), (21, 52)]
        body.add(poly([(CX + side * x, y) for x, y in sleeve]), TEE, 5, texture=0.08)
        body.add(limb((CX + side * 31, 73), (CX + side * 40, 86), 7, 5.5), SKIN, 3)
        hand = union(ellipse(CX + side * 43, 88, 4.6, 2.6), ellipse(CX + side * 40, 86.4, 1.6, 1.4))
        body.add(hand, SKIN, 2)
    return body


def build_sprite() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return the RGBA sprite and the sprite coordinates of the eye pixels."""
    canvas = meditating_body().rasterize(np.random.default_rng(3))
    av = sample_avatar(AVATAR)
    keep = head_mask(av, 45)
    eyes = []
    for y, x in zip(*np.nonzero(keep), strict=True):
        if y >= 40 and not is_skin(av[y, x]):
            continue
        canvas[y + HEAD_DY, x + HEAD_DX] = (*av[y, x], 255)
        if y in EYE_ROWS and x in EYE_COLS and av[y, x, 0] < 0x80:
            eyes.append((int(y + HEAD_DY), int(x + HEAD_DX)))
    return canvas.astype(np.uint8), eyes


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


def eyelids(canvas: np.ndarray, eyes: list[tuple[int, int]]) -> str:
    """Closed-eye pixels (skin with a dark lash line), shown briefly by the blink animation."""
    lash_row = max(y for y, _ in eyes)
    skin = canvas[lash_row + 2, eyes[0][1], :3]
    lids = np.zeros_like(canvas)
    for y, x in eyes:
        lids[y, x] = (*(hexrgb(LASH) if y == lash_row else skin), 255)
    return sprite_image(lids, "lid")


def connector_end(node: tuple[int, int]) -> tuple[float, float]:
    cx, cy = CENTER
    dx, dy = node[0] - cx, node[1] - cy
    dist = (dx * dx + dy * dy) ** 0.5
    return cx + dx / dist * RING_R, cy + dy / dist * RING_R


def project_nodes() -> str:
    parts = []
    for i, (name, tagline) in enumerate(PROJECTS):
        side = i % 2
        row = ROW_Y[i // 2]
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
.foot {{ font-size: 14px; fill: {FG_DIM}; }}
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
  <rect y="124" width="{VIEW_W}" height="{VIEW_H - 124 - 64}" fill="url(#dots)"/>
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


def footer() -> str:
    y = VIEW_H - 64
    return f"""<line x1="0" y1="{y}" x2="{VIEW_W}" y2="{y}" class="rule"/>
<text x="48" y="{y + 38}" class="foot"><tspan class="prompt">$ </tspan
  >open source · local-first · built for people and their agents</text>
<text x="{VIEW_W - 48}" y="{y + 38}" text-anchor="end" class="foot"
  >github.com/<tspan class="url">kenn-io</tspan></text>"""


def center_piece(canvas: np.ndarray, eyes: list[tuple[int, int]]) -> str:
    cx, cy = CENTER
    sx, sy = cx - SPRITE_W * PX // 2, cy - SPRITE_H * PX // 2 - 6
    shadow_y = sy + SPRITE_H * PX + 26
    return f"""<circle class="aura" cx="{cx}" cy="{cy}" r="{RING_R + 30}" fill="url(#aura)"/>
<circle class="ring ring-a" cx="{cx}" cy="{cy}" r="{RING_R}"/>
<circle class="ring ring-b" cx="{cx}" cy="{cy}" r="{RING_R - 26}"/>
<ellipse class="shadow" cx="{cx}" cy="{shadow_y}" rx="150" ry="14" fill="url(#shadow)"/>
<g class="float">
  <g transform="translate({sx} {sy}) scale({PX})">
    {sprite_image(canvas)}
    {eyelids(canvas, eyes)}
  </g>
</g>"""


def render() -> str:
    canvas, eyes = build_sprite()
    title = "Wes McKinney at Kenn Software"
    desc = (
        "A 16-bit Wes McKinney floats cross-legged in meditation, orbited by Kenn Software "
        "projects: " + ", ".join(f"{n} ({t})" for n, t in PROJECTS) + "."
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
{center_piece(canvas, eyes)}
{footer()}
</svg>
"""


def main() -> None:
    OUTPUT.write_text(render())
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
