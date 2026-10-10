# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow==12.3.0", "numpy==2.5.3"]
# ///
"""Render assets/kenn-hero.svg: a pixel-art lotus-pose sprite orbited by Kenn projects.

The head comes from assets/avatar.png (the 16-bit GitHub avatar), resampled onto its
native pixel grid at half resolution. The body is drawn procedurally on the same grid.

Usage: uv run scripts/build_hero.py
"""

from collections import deque
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
AVATAR = ROOT / "assets" / "avatar.png"
OUTPUT = ROOT / "assets" / "kenn-hero.svg"

AVATAR_PITCH = 4.24 * 2
SPRITE_W, SPRITE_H = 112, 100
CX = SPRITE_W // 2
HEAD_DX, HEAD_DY = CX - 27, 2
PX = 3
# Avatar-grid box around the eyes, used for the blink animation.
EYE_ROWS, EYE_COLS = range(21, 23), range(20, 35)

SHIRT = ("#2f3240", "#545767", "#646777", "#7c7e8c")
SKIN = ("#8a4518", "#de9152", "#f4ad6c", "#fcc685")
PANTS_BACK = ("#10151f", "#1f2940", "#2a3754", "#3a4b6e")
PANTS = ("#141a28", "#2a3753", "#364767", "#4b608a")
PANTS_FRONT = ("#161d2c", "#31405f", "#405476", "#5a72a0")

FOOT = [
    ".......o.o.o.",
    "...oooolololo",
    "..olllllbbbbo",
    ".obbbbbbbbbbo",
    "osbbbbbbbbbso",
    ".ossbbbbbbso.",
    "..ooooooooo..",
]

# Colors from the kenn.io site theme.
BG, BG_SOFT = "#0a0e14", "#0c1117"
FG, FG_MUTED, FG_DIM = "#e3e8ee", "#9aafc0", "#5e7385"
LINE = "rgba(227,232,238,0.13)"
SAGE, SAGE_DIM, TAN = "#6dac9b", "#4f8475", "#d9a880"
MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,'Cascadia Mono',Consolas,monospace"

# Ordered as on kenn.io.
PROJECTS = [
    ("AgentsView", "what your agents did, and cost"),
    ("roborev", "continuous review for agent code"),
    ("Ghosthub", "one native Mac terminal for tmux"),
    ("msgvault", "a lifetime of email, searchable"),
    ("Kenn Forge", "triage, delegate, and review"),
    ("kata", "local issue ledger for agents"),
    ("docbank", "your document system of record"),
    ("kwt", "a worktree + tmux per branch"),
    ("Fotobank", "your photo library, agent-ready"),
    ("BenchDB", "catch perf regressions early"),
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


def draw_mask(draw_fn) -> np.ndarray:
    img = Image.new("1", (SPRITE_W, SPRITE_H), 0)
    draw_fn(ImageDraw.Draw(img))
    return np.asarray(img).astype(bool)


def poly_mask(points) -> np.ndarray:
    return draw_mask(lambda d: d.polygon(points, fill=1))


def ellipse_mask(box) -> np.ndarray:
    return draw_mask(lambda d: d.ellipse(box, fill=1))


def capsule(p0, p1, width: int) -> np.ndarray:
    r = width / 2

    def draw(d):
        d.line([p0, p1], fill=1, width=width)
        for x, y in (p0, p1):
            d.ellipse((x - r, y - r, x + r - 1, y + r - 1), fill=1)

    return draw_mask(draw)


def shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return m translated so out[y, x] == m[y - dy, x - dx] (zero-filled)."""
    out = np.zeros_like(m)
    h, w = m.shape
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = m[
        max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


def paint(canvas, mask, ramp, rng=None, texture=0.0) -> None:
    """Fill mask lit from above with a 4-step ramp: outline, shadow, base, highlight."""
    outline, shadow, base, light = (hexrgb(c) for c in ramp)
    inner = mask & shift(mask, 1, 0) & shift(mask, -1, 0) & shift(mask, 0, 1) & shift(mask, 0, -1)
    lit = inner & (~shift(mask, 1, 0) | ~shift(mask, 1, 1))
    dark = inner & (~shift(mask, -2, 0) | ~shift(mask, -1, -1))
    for y, x in zip(*np.nonzero(mask), strict=True):
        if not inner[y, x]:
            c = outline
        elif lit[y, x]:
            c = light
        elif dark[y, x]:
            c = shadow
        elif rng is not None and rng.random() < texture:
            c = shadow if rng.random() < 0.6 else light
        else:
            c = base
        canvas[y, x] = (*c, 255)


def paint_foot(canvas, direction: int, x: int, y: int) -> None:
    """Stamp a sole-up foot; toes point outward (direction=+1 means right)."""
    ramp = dict(zip("osbl", (hexrgb(c) for c in SKIN), strict=True))
    w = len(FOOT[0])
    for dy, row in enumerate(FOOT):
        for dx, ch in enumerate(row if direction > 0 else row[::-1]):
            if ch != ".":
                px = x + dx - (w - 3 if direction > 0 else 2)
                canvas[y + dy, px] = (*ramp[ch], 255)


def leg_mask(side: int, drop: int) -> np.ndarray:
    """Folded leg from the knee on `side` to the ankle past center."""
    k = CX + side * 46
    a = CX - side * 12
    knee = ellipse_mask((k - 8, 83 + drop, k + 8, 97 + drop))
    shin = poly_mask([(k, 83 + drop), (a, 81 + drop), (a, 88 + drop), (k, 97 + drop)])
    ankle = ellipse_mask((a - 5, 81 + drop, a + 5, 89 + drop))
    return knee | shin | ankle


def paint_body(canvas, rng) -> None:
    for s in (-1, 1):
        paint(canvas, capsule((CX + s * 24, 54), (CX + s * 30, 69), 8), SKIN)
    torso = poly_mask(
        [(CX - 9, 42), (CX + 9, 42), (CX + 21, 46), (CX + 27, 52), (CX + 28, 58), (CX + 23, 60),
         (CX + 21, 84), (CX - 21, 84), (CX - 23, 60), (CX - 28, 58), (CX - 27, 52), (CX - 21, 46)]
    )  # fmt: skip
    paint(canvas, torso, SHIRT, rng, 0.08)
    paint(canvas, poly_mask([(CX - 7, 42), (CX + 7, 42), (CX, 51)]), SKIN)

    paint(canvas, ellipse_mask((CX - 26, 78, CX + 26, 95)), PANTS_BACK, rng, 0.03)
    paint(canvas, leg_mask(-1, 0), PANTS, rng, 0.03)
    paint_foot(canvas, 1, CX + 22, 77)
    paint(canvas, leg_mask(1, 2), PANTS_FRONT, rng, 0.03)
    paint_foot(canvas, -1, CX - 22, 79)

    for s in (-1, 1):
        paint(canvas, capsule((CX + s * 30, 68), (CX + s * 42, 81), 7), SKIN)
        x0, x1 = sorted((CX + s * 40, CX + s * 49))
        paint(canvas, ellipse_mask((x0, 79, x1, 86)), SKIN)
        canvas[82, CX + s * 46] = (*hexrgb(SKIN[0]), 255)
        canvas[82, CX + s * 47] = (*hexrgb(SKIN[1]), 255)


def build_sprite() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return the RGBA sprite and the sprite coordinates of the eye pixels."""
    canvas = np.zeros((SPRITE_H, SPRITE_W, 4), int)
    paint_body(canvas, np.random.default_rng(7))
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


def sprite_paths(canvas: np.ndarray) -> str:
    """Encode opaque pixels as one path per color, merging horizontal runs."""
    runs: dict[str, list[str]] = {}
    for y in range(canvas.shape[0]):
        x = 0
        while x < canvas.shape[1]:
            if canvas[y, x, 3] == 0:
                x += 1
                continue
            color = rgbhex(canvas[y, x])
            start = x
            while (
                x < canvas.shape[1]
                and canvas[y, x, 3]
                and (canvas[y, x, :3] == canvas[y, start, :3]).all()
            ):
                x += 1
            runs.setdefault(color, []).append(f"M{start} {y}h{x - start}v1h-{x - start}z")
    return "".join(f'<path fill="{c}" d="{"".join(d)}"/>' for c, d in sorted(runs.items()))


def eyelids(canvas: np.ndarray, eyes: list[tuple[int, int]]) -> str:
    """Closed-eye pixels (skin with a dark lash line), shown briefly by the blink animation."""
    lash_row = max(y for y, _ in eyes)
    lash = "".join(f"M{x} {y}h1v1h-1z" for y, x in eyes if y == lash_row)
    skin = rgbhex(canvas[lash_row + 2, eyes[0][1]])
    lid = "".join(f"M{x} {y}h1v1h-1z" for y, x in eyes if y != lash_row)
    return f'<g class="lid"><path fill="{skin}" d="{lid}"/><path fill="#9a4d1e" d="{lash}"/></g>'


def rgbhex(rgb: np.ndarray) -> str:
    r, g, b = (int(v) for v in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


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
.tag {{ font-size: 15px; fill: {FG_MUTED}; }}
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
  <g transform="translate({sx} {sy}) scale({PX})" shape-rendering="crispEdges">
    {sprite_paths(canvas)}
    {eyelids(canvas, eyes)}
  </g>
</g>"""


def render() -> str:
    canvas, eyes = build_sprite()
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
{center_piece(canvas, eyes)}
{footer()}
</svg>
"""


def main() -> None:
    OUTPUT.write_text(render())
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
