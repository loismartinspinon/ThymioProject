"""
Generates the printable documents for the Thymio experimental arena
(EPFL - Basics of Mobile Robotics).

Outputs (written next to this script):
    arena_technical_drawing.pdf   A3 landscape, 3 sheets: dimensioned top view (1:5),
                                  coordinate tables, navigation analysis (1:6)
    arena_full_scale_1to1.pdf     single 1040 x 883.75 mm page (arena + 20 mm margin), 1:1
    arena_tiled_A3_1to1.pdf       same arena split on 9 A3 sheets (3 x 3), 1:1
    obstacles_sheet_A3.pdf        the two 6 x 6 cm obstacles, 1:1
    aruco_markers_A3.pdf          the four 5 x 5 cm ArUco markers (IDs 0-3), 1:1
    arena_complete.pdf            all of the above in one PDF

Requires: reportlab, opencv-python (>= 4.7, for cv2.aruco)
    pip install reportlab opencv-python
Run:
    python arena/generate_arena.py

Coordinate system: origin (0, 0) at the bottom-left corner of the arena,
x to the right, y upwards, all arena values in centimetres.
"""

import os
import math

import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.colors import Color, black, white
from reportlab.lib.utils import simpleSplit


# =============================================================================
# Arena specification (cm)
# =============================================================================

ARENA_W = 100.0
# 3 rows of A3 tiles (28.125 cm each). The third row was added on top of the original
# 100 x 56.25 cm (16:9) arena to get free space for random objects: the origin, the
# obstacles and the bottom markers did not move, only the top markers moved up.
PREVIOUS_ARENA_H = 56.25
ARENA_H = 84.375
TILES = (3, 3)                             # columns x rows of A3 landscape tiles (1:1 tiled print)

MARKER_SIZE = 5.0                          # ArUco side, black border included
ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Marker centres. The OUTER corner of each marker coincides with the arena corner.
MARKERS = [
    # id, corner name, centre (x, y)
    (0, "bottom-left (origin)", (2.5, 2.5)),
    (1, "bottom-right", (97.5, 2.5)),
    (2, "top-right", (97.5, ARENA_H - MARKER_SIZE / 2)),
    (3, "top-left", (2.5, ARENA_H - MARKER_SIZE / 2)),
]

OBSTACLE_SIZE = 6.0
OBSTACLE_RADIUS = 1.0                      # corner radius of the rounded square
OBSTACLES = [
    ("O1", (25.0, 17.0)),
    ("O2", (55.0, 39.0)),
]
# Both obstacles are shifted 10 cm to the left of the previous (35, 17) / (65, 39) layout,
# so the right part of the arena stays free for local-navigation tests.

# Values used only for the navigation analysis sheet
THYMIO_W = 11.0                            # Thymio II footprint is about 11.0 x 11.2 cm
INFLATION = 6.0                            # half-width (5.5) + 0.5 cm safety margin
START = (11.0, 11.0)                       # suggested start (not printed on the arena)
GOAL = (88.0, 48.0)                        # suggested goal  (not printed on the arena)
LOCAL_ZONE = (63.0, 22.0, 85.0, 42.0)      # x0, y0, x1, y1: free area for a hand-placed obstacle

RECT_PX_PER_CM = 19.2                      # 1920 px / 100 cm (rectified 1920 x 1620 map)
RECT_W = round(ARENA_W * RECT_PX_PER_CM)   # 1920
RECT_H = round(ARENA_H * RECT_PX_PER_CM)   # 1620
VISION_MAP = (256, 216)                    # MAP_WIDTH x MAP_HEIGHT in Vision.py (2.56 px/cm)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paper. The technical drawing is laid out in A4-landscape units and scaled up
# by K so that it fills an A3 sheet. The top view is at 1:5 and the navigation
# sheet at 1:6 on the paper (the taller arena no longer fits at 1:4).
PAPER = A3
PAPER_NAME = "A3"
K = PAPER[0] / A4[0]                       # A3 / A4 = sqrt(2)
TOP_VIEW_SCALE = 5                         # 1:5 on A3
NAV_SCALE = 6                              # 1:6 on A3


def draw_mm_per_cm(scale):
    """Arena cm -> mm in A4 layout units, for a 1:scale drawing on the paper."""
    return 10.0 / scale / K


def scale_text(scale):
    return f"1:{scale} on {PAPER_NAME}"

GREY_LIGHT = Color(0.86, 0.86, 0.86)
GREY_MID = Color(0.55, 0.55, 0.55)
GREY_DARK = Color(0.30, 0.30, 0.30)
BLUE = Color(0.12, 0.37, 0.75)
RED = Color(0.78, 0.18, 0.15)
GREEN = Color(0.10, 0.55, 0.30)


# =============================================================================
# Helpers
# =============================================================================

def marker_bits(marker_id):
    """6 x 6 matrix (border included), True = black cell, row 0 = top of the marker."""
    img = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, 6, borderBits=1)
    return [[img[r, c] < 128 for c in range(6)] for r in range(6)]


class Frame:
    """Maps arena coordinates (cm) to page coordinates (points)."""

    def __init__(self, ox_mm, oy_mm, mm_per_cm):
        self.ox = ox_mm * mm
        self.oy = oy_mm * mm
        self.k = mm_per_cm * mm

    def x(self, xc):
        return self.ox + xc * self.k

    def y(self, yc):
        return self.oy + yc * self.k

    def len(self, lc):
        return lc * self.k


def draw_marker(c, f, marker_id, centre):
    """Draws an ArUco marker as one vector path (no seams between cells)."""
    bits = marker_bits(marker_id)
    side = f.len(MARKER_SIZE)
    cell = side / 6.0
    x0 = f.x(centre[0]) - side / 2
    ytop = f.y(centre[1]) + side / 2
    p = c.beginPath()
    for r in range(6):
        for col in range(6):
            if bits[r][col]:
                p.rect(x0 + col * cell, ytop - (r + 1) * cell, cell, cell)
    c.setFillColor(black)
    c.drawPath(p, fill=1, stroke=0)


def draw_obstacle(c, f, centre, color=black):
    s = f.len(OBSTACLE_SIZE)
    c.setFillColor(color)
    c.roundRect(f.x(centre[0]) - s / 2, f.y(centre[1]) - s / 2, s, s,
                f.len(OBSTACLE_RADIUS), stroke=0, fill=1)


def draw_arena_content(c, f):
    """White arena + 4 markers + 2 obstacles. Used by every 1:1 output."""
    c.setFillColor(white)
    c.rect(f.x(0), f.y(0), f.len(ARENA_W), f.len(ARENA_H), stroke=0, fill=1)
    for mid, _, centre in MARKERS:
        draw_marker(c, f, mid, centre)
    for _, centre in OBSTACLES:
        draw_obstacle(c, f, centre)


def arrow_head(c, x, y, angle, length=2.2 * mm, half=0.7 * mm):
    """Filled arrow head whose tip is at (x, y), pointing along angle (rad)."""
    ca, sa = math.cos(angle), math.sin(angle)
    bx, by = x - length * ca, y - length * sa
    p = c.beginPath()
    p.moveTo(x, y)
    p.lineTo(bx - half * sa, by + half * ca)
    p.lineTo(bx + half * sa, by - half * ca)
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def dim_h(c, x1, x2, y, text, ext1=None, ext2=None, font=7):
    """Horizontal dimension between page x1 and x2 at page height y.
    ext1/ext2: page y where the extension lines start (None = no extension line)."""
    c.setStrokeColor(black)
    c.setFillColor(black)
    c.setLineWidth(0.3)
    for x, e in ((x1, ext1), (x2, ext2)):
        if e is not None:
            d = 1.5 * mm if y < e else -1.5 * mm
            c.line(x, e, x, y - d)
    c.line(x1, y, x2, y)
    arrow_head(c, x1, y, math.pi)
    arrow_head(c, x2, y, 0.0)
    c.setFont("Helvetica", font)
    c.drawCentredString((x1 + x2) / 2, y + 0.9 * mm, text)


def dim_v(c, y1, y2, x, text, ext1=None, ext2=None, font=7):
    """Vertical dimension between page y1 and y2 at page x (text rotated)."""
    c.setStrokeColor(black)
    c.setFillColor(black)
    c.setLineWidth(0.3)
    for y, e in ((y1, ext1), (y2, ext2)):
        if e is not None:
            d = 1.5 * mm if x < e else -1.5 * mm
            c.line(e, y, x - d, y)
    c.line(x, y1, x, y2)
    arrow_head(c, x, y1, -math.pi / 2)
    arrow_head(c, x, y2, math.pi / 2)
    c.saveState()
    c.translate(x - 0.9 * mm, (y1 + y2) / 2)
    c.rotate(90)
    c.setFont("Helvetica", font)
    c.drawCentredString(0, 0, text)
    c.restoreState()


def centre_line(c, x1, y1, x2, y2):
    c.saveState()
    c.setStrokeColor(GREY_MID)
    c.setLineWidth(0.25)
    c.setDash([4, 1.5, 1, 1.5])
    c.line(x1, y1, x2, y2)
    c.restoreState()


def wrap_text(c, text, x, y, width, font="Helvetica", size=8, leading=None):
    """Draws wrapped text, returns the y below the last line."""
    leading = leading or size * 1.3
    c.setFont(font, size)
    for line in simpleSplit(text, font, size, width):
        c.drawString(x, y, line)
        y -= leading
    return y


def fmt(v):
    """35.0 -> '35', 56.25 -> '56.25'."""
    return f"{v:g}"


def draw_previous_edge(c, f, x1=ARENA_W - MARKER_SIZE, label_below=True):
    """Dashed line at the top edge of the original 56.25 cm arena + label of the added row."""
    c.saveState()
    c.setStrokeColor(GREY_MID)
    c.setLineWidth(0.4)
    c.setDash([3, 2])
    c.line(f.x(MARKER_SIZE), f.y(PREVIOUS_ARENA_H), f.x(x1), f.y(PREVIOUS_ARENA_H))
    c.restoreState()
    c.setFillColor(GREY_DARK)
    c.setFont("Helvetica-Oblique", 6.3)
    xm = f.x((MARKER_SIZE + x1) / 2)
    c.drawCentredString(xm, f.y(PREVIOUS_ARENA_H) + 1.5 * mm,
                        f"added row (y {fmt(PREVIOUS_ARENA_H)} - {fmt(ARENA_H)}): free space for random objects")
    if label_below:
        c.drawCentredString(xm, f.y(PREVIOUS_ARENA_H) - 3.5 * mm,
                            f"previous top edge, y = {fmt(PREVIOUS_ARENA_H)} (not printed)")


def segment_clearance(a, b, centre, samples=4000):
    """Smallest distance (cm) between segment a-b and the rounded obstacle at centre (0 = crosses it)."""
    inner = OBSTACLE_SIZE / 2 - OBSTACLE_RADIUS
    best = float("inf")
    for i in range(samples + 1):
        t = i / samples
        px, py = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        dx = max(abs(px - centre[0]) - inner, 0.0)
        dy = max(abs(py - centre[1]) - inner, 0.0)
        best = min(best, max(math.hypot(dx, dy) - OBSTACLE_RADIUS, 0.0))
    return best


def obstacle_box(centre):
    h = OBSTACLE_SIZE / 2
    return centre[0] - h, centre[0] + h, centre[1] - h, centre[1] + h


def title_block(c, W, H, sheet, n_sheets, title, scale_text):
    """Outer frame + bottom title strip of every technical drawing sheet."""
    x0, y0, x1, y1 = 8 * mm, 8 * mm, W - 8 * mm, 24 * mm
    c.setStrokeColor(black)
    c.setLineWidth(0.6)
    c.rect(x0, y0, x1 - x0, H - 16 * mm)
    c.rect(x0, y0, x1 - x0, y1 - y0)
    cols = [x0, x0 + 105 * mm, x0 + 135 * mm, x0 + 168 * mm, x0 + 186 * mm, x1]
    c.setLineWidth(0.3)
    for x in cols[1:-1]:
        c.line(x, y0, x, y1)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x0 + 3 * mm, y0 + 9.5 * mm, title)
    c.setFont("Helvetica", 7)
    c.drawString(x0 + 3 * mm, y0 + 4 * mm, "EPFL - Basics of Mobile Robotics - Thymio project")
    cells = [
        ("SCALE", scale_text),
        ("UNITS", "cm"),
        ("SHEET", f"{sheet} / {n_sheets}"),
    ]
    for (label, value), xa in zip(cells, cols[1:4]):
        c.setFont("Helvetica", 5.5)
        c.setFillColor(GREY_DARK)
        c.drawString(xa + 2 * mm, y0 + 11 * mm, label)
        c.setFont("Helvetica-Bold", 8.5)
        c.setFillColor(black)
        c.drawString(xa + 2 * mm, y0 + 4.5 * mm, value)
    c.setFont("Helvetica", 5.5)
    c.setFillColor(GREY_DARK)
    c.drawString(cols[4] + 2 * mm, y0 + 11 * mm, "ARENA / CAMERA")
    c.setFont("Helvetica", 6.8)
    c.setFillColor(black)
    c.drawString(cols[4] + 2 * mm, y0 + 6.5 * mm, f"{fmt(ARENA_W)} x {fmt(ARENA_H)} cm, white matte, 4 ArUco + 2 obstacles")
    c.drawString(cols[4] + 2 * mm, y0 + 2.8 * mm, "Aukey 1080p webcam, 1920 x 1080 px, vertical, above")


def table(c, x, y, col_w, rows, size=7.5, row_h=5.2 * mm, header=True):
    """Simple ruled table. rows[0] is the header. Returns the y below the table."""
    total_w = sum(col_w)
    c.setLineWidth(0.3)
    c.setStrokeColor(GREY_MID)
    for i, row in enumerate(rows):
        if i == 0 and header:
            c.setFillColor(Color(0.92, 0.92, 0.92))
            c.rect(x, y - row_h, total_w, row_h, stroke=0, fill=1)
        c.setFillColor(black)
        c.setFont("Helvetica-Bold" if (i == 0 and header) else "Helvetica", size)
        xc = x
        for w, cell in zip(col_w, row):
            c.drawString(xc + 1.5 * mm, y - row_h + 1.6 * mm, str(cell))
            xc += w
        c.line(x, y - row_h, x + total_w, y - row_h)
        y -= row_h
    c.line(x, y + row_h * len(rows), x + total_w, y + row_h * len(rows))
    return y


# =============================================================================
# 1. Technical drawing (A4 landscape, 3 sheets)
# =============================================================================

def sheet_top_view(c):
    W, H = landscape(A4)            # layout units, scaled by K onto the A3 page
    c.scale(K, K)
    f = Frame(52, 54, draw_mm_per_cm(TOP_VIEW_SCALE))   # 1:5 on A3 -> 1 cm of arena = 2 mm on paper
    title_block(c, W, H, 1, 3, "Thymio experimental arena - dimensioned top view", scale_text(TOP_VIEW_SCALE))

    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(black)
    c.drawString(14 * mm, H - 16 * mm, "TOP VIEW - final obstacle configuration")
    c.setFont("Helvetica", 7.5)
    c.drawString(14 * mm, H - 21 * mm,
                 "Origin (0, 0) at bottom-left arena corner, x to the right, y upwards. "
                 "All dimensions in cm. Exactly two obstacles.")

    # grid (10 cm)
    c.setStrokeColor(GREY_LIGHT)
    c.setLineWidth(0.25)
    for gx in range(10, 100, 10):
        c.line(f.x(gx), f.y(0), f.x(gx), f.y(ARENA_H))
    for gy in range(10, math.ceil(ARENA_H), 10):
        c.line(f.x(0), f.y(gy), f.x(ARENA_W), f.y(gy))

    # previous top edge: the row above it was added as free space
    draw_previous_edge(c, f)

    # arena content
    for mid, _, centre in MARKERS:
        draw_marker(c, f, mid, centre)
    for _, centre in OBSTACLES:
        draw_obstacle(c, f, centre)

    c.setStrokeColor(black)
    c.setLineWidth(0.9)
    c.rect(f.x(0), f.y(0), f.len(ARENA_W), f.len(ARENA_H), stroke=1, fill=0)

    # obstacle centre crosses (white on black) and labels
    for name, (cx, cy) in OBSTACLES:
        c.setStrokeColor(white)
        c.setLineWidth(0.35)
        d = f.len(1.6)
        c.line(f.x(cx) - d, f.y(cy), f.x(cx) + d, f.y(cy))
        c.line(f.x(cx), f.y(cy) - d, f.x(cx), f.y(cy) + d)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 7.5)
    (_, (l1x, l1y)), (_, (l2x, l2y)) = OBSTACLES
    c.drawRightString(f.x(l1x - 4), f.y(l1y + 4.5), f"O1  ({fmt(l1x)}, {fmt(l1y)})")
    c.drawRightString(f.x(l2x - 4), f.y(l2y - 5.0), f"O2  ({fmt(l2x)}, {fmt(l2y)})")

    # marker labels
    c.setFont("Helvetica-Bold", 6.5)
    c.drawString(f.x(6), f.y(1.2), "ID 0")
    c.drawRightString(f.x(94), f.y(1.2), "ID 1")
    c.drawRightString(f.x(94), f.y(ARENA_H - 2.25), "ID 2")
    c.drawString(f.x(6), f.y(ARENA_H - 2.25), "ID 3")

    # centre lines through the obstacles, extended to the dimension lines
    (n1, (x1c, y1c)), (n2, (x2c, y2c)) = OBSTACLES
    centre_line(c, f.x(x1c), f.y(y1c), f.x(x1c), f.y(0) - 9.5 * mm)
    centre_line(c, f.x(x2c), f.y(y2c), f.x(x2c), f.y(0) - 15.5 * mm)
    centre_line(c, f.x(x1c), f.y(y1c), f.x(0) - 9.5 * mm, f.y(y1c))
    centre_line(c, f.x(x2c), f.y(y2c), f.x(0) - 15.5 * mm, f.y(y2c))

    # bottom: baseline dimensions from the origin
    yb = f.y(0)
    dim_h(c, f.x(0), f.x(x1c), yb - 8 * mm, fmt(x1c), ext1=yb)
    dim_h(c, f.x(0), f.x(x2c), yb - 14 * mm, fmt(x2c), ext1=yb)
    dim_h(c, f.x(0), f.x(ARENA_W), yb - 20 * mm, fmt(ARENA_W), ext1=yb, ext2=yb)

    # left: baseline dimensions from the origin
    xl = f.x(0)
    dim_v(c, f.y(0), f.y(y1c), xl - 8 * mm, fmt(y1c), ext1=xl)
    dim_v(c, f.y(0), f.y(y2c), xl - 14 * mm, fmt(y2c), ext1=xl)
    dim_v(c, f.y(0), f.y(ARENA_H), xl - 20 * mm, fmt(ARENA_H), ext1=xl, ext2=xl)

    # obstacle O2 size + corner radius
    bx0, bx1, by0, by1 = obstacle_box((x2c, y2c))
    dim_h(c, f.x(bx0), f.x(bx1), f.y(by1) + 4 * mm, fmt(OBSTACLE_SIZE), ext1=f.y(by1), ext2=f.y(by1))
    dim_v(c, f.y(by0), f.y(by1), f.x(bx1) + 5 * mm, fmt(OBSTACLE_SIZE), ext1=f.x(bx1), ext2=f.x(bx1))
    c.setStrokeColor(black)
    c.setLineWidth(0.3)
    kx, ky = f.x(bx1 - 0.3), f.y(by0 + 0.3)
    c.line(kx, ky, kx + 7 * mm, ky - 5 * mm)
    c.setFont("Helvetica", 6.5)
    c.setFillColor(black)
    c.drawString(kx + 7.5 * mm, ky - 6 * mm, "R1 (typ.)")

    # marker size (ID 3 width, ID 2 height)
    yt = f.y(ARENA_H)
    dim_h(c, f.x(0), f.x(MARKER_SIZE), yt + 5 * mm, fmt(MARKER_SIZE), ext1=yt, ext2=yt)
    xr = f.x(ARENA_W)
    dim_v(c, f.y(ARENA_H - MARKER_SIZE), f.y(ARENA_H), xr + 5 * mm, fmt(MARKER_SIZE), ext1=xr, ext2=xr)

    # origin symbol
    c.setStrokeColor(black)
    c.setLineWidth(0.4)
    c.circle(f.x(0), f.y(0), 1.2 * mm, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 6.5)
    c.drawRightString(f.x(0) - 1.6 * mm, f.y(0) - 4.2 * mm, "(0, 0)")

    # legend + axis triad on the right
    lx, ly = xr + 12 * mm, f.y(ARENA_H) - 2 * mm
    c.setFont("Helvetica-Bold", 7)
    c.drawString(lx, ly, "LEGEND")
    ly -= 6 * mm
    s = 3.5 * mm
    c.setFillColor(black)
    c.roundRect(lx, ly - 1 * mm, s, s, 0.6 * mm, stroke=0, fill=1)
    c.setFont("Helvetica", 6.3)
    c.drawString(lx + s + 2 * mm, ly, "Obstacle 6 x 6, R1")
    ly -= 6 * mm
    bits = marker_bits(0)
    cell = s / 6
    for r in range(6):
        for col in range(6):
            if bits[r][col]:
                c.rect(lx + col * cell, ly - 1 * mm + s - (r + 1) * cell, cell, cell, stroke=0, fill=1)
    c.drawString(lx + s + 2 * mm, ly, "ArUco 5 x 5")
    ly -= 4 * mm
    c.drawString(lx + s + 2 * mm, ly, ARUCO_DICT_NAME)
    ly -= 6 * mm
    c.setStrokeColor(GREY_LIGHT)
    c.setLineWidth(0.6)
    c.line(lx, ly + 0.8 * mm, lx + s, ly + 0.8 * mm)
    c.drawString(lx + s + 2 * mm, ly, "Grid 10 cm")
    ly -= 6 * mm
    centre_line(c, lx, ly + 0.8 * mm, lx + s, ly + 0.8 * mm)
    c.setFillColor(black)
    c.drawString(lx + s + 2 * mm, ly, "Centre line")

    # axis triad
    ax, ay = lx + 2 * mm, ly - 22 * mm
    c.setStrokeColor(black)
    c.setFillColor(black)
    c.setLineWidth(0.5)
    c.line(ax, ay, ax + 12 * mm, ay)
    c.line(ax, ay, ax, ay + 12 * mm)
    arrow_head(c, ax + 12 * mm, ay, 0)
    arrow_head(c, ax, ay + 12 * mm, math.pi / 2)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(ax + 13 * mm, ay - 1 * mm, "x")
    c.drawString(ax - 1 * mm, ay + 13.5 * mm, "y")


def sheet_coordinates(c):
    W, H = landscape(A4)
    c.scale(K, K)
    title_block(c, W, H, 2, 3, "Thymio experimental arena - coordinates", "-")
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(14 * mm, H - 16 * mm, "COORDINATES")

    x = 14 * mm
    y = H - 24 * mm
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, "Obstacles (black matte, 6 x 6 cm rounded square, corner radius 1 cm)")
    y -= 2 * mm
    rows = [["Obstacle", "Centre x", "Centre y", "x range", "y range", "Size", f"Centre in {RECT_W}x{RECT_H} map (px)"]]
    for name, (cx, cy) in OBSTACLES:
        x0, x1, y0, y1 = obstacle_box((cx, cy))
        px = cx * RECT_PX_PER_CM
        py = (ARENA_H - cy) * RECT_PX_PER_CM
        rows.append([name, f"{cx:.2f}", f"{cy:.2f}", f"{fmt(x0)} - {fmt(x1)}", f"{fmt(y0)} - {fmt(y1)}",
                     "6 x 6", f"({px:.1f}, {py:.1f})"])
    y = table(c, x, y, [20 * mm, 20 * mm, 20 * mm, 27 * mm, 27 * mm, 18 * mm, 50 * mm], rows)

    y -= 8 * mm
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, f"ArUco markers ({ARUCO_DICT_NAME}, 5 x 5 cm incl. black border, cell 0.833 cm, printed upright)")
    y -= 2 * mm
    rows = [["ID", "Corner", "Centre x", "Centre y", "x range", "y range", "Arena corner = outer marker corner"]]
    for mid, name, (cx, cy) in MARKERS:
        h = MARKER_SIZE / 2
        corner = (0 if cx < ARENA_W / 2 else ARENA_W, 0 if cy < ARENA_H / 2 else ARENA_H)
        rows.append([str(mid), name, f"{cx:.2f}", f"{cy:.3f}", f"{fmt(cx - h)} - {fmt(cx + h)}",
                     f"{fmt(cy - h)} - {fmt(cy + h)}", f"({fmt(corner[0])}, {fmt(corner[1])})"])
    y = table(c, x, y, [12 * mm, 38 * mm, 20 * mm, 20 * mm, 25 * mm, 27 * mm, 52 * mm], rows)

    y -= 8 * mm
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, "Free space / clearances (edge to edge, cm)")
    y -= 2 * mm
    o1, o2 = obstacle_box(OBSTACLES[0][1]), obstacle_box(OBSTACLES[1][1])
    r = OBSTACLE_RADIUS
    # closest points: the facing rounded corners (arc centres 1 cm inside the box corners)
    gap_diag = math.hypot((o2[0] + r) - (o1[1] - r), (o2[2] + r) - (o1[3] - r)) - 2 * r
    rows = [
        ["Passage", "Width", "Thymio (11 cm) fits with", "Comment"],
        ["O1 - O2 (closest rounded corners)", f"{gap_diag:.1f}", f"{gap_diag - THYMIO_W:.1f}", "main corridor, wide"],
        ["O1 - O2 horizontal / vertical offset", f"{o2[0] - o1[1]:g} / {o2[2] - o1[3]:g}", "-", "no overlap in x or y"],
        ["O2 - right edge", f"{ARENA_W - o2[1]:g}", f"{ARENA_W - o2[1] - THYMIO_W:g}", "very wide: local-navigation area"],
        ["O1 - left edge", f"{o1[0]:g}", f"{o1[0] - THYMIO_W:g}", "comfortable (2 robot widths)"],
        ["O1 - top edge  /  O2 - bottom edge", f"{ARENA_H - o1[3]:g} / {o2[2]:g}", f"{o2[2] - THYMIO_W:g}", "wide"],
        ["O1 - bottom edge", f"{o1[2]:g}", f"{o1[2] - THYMIO_W:g}", "narrow, usually closed after inflation"],
        ["O2 - top edge", f"{ARENA_H - o2[3]:g}", f"{ARENA_H - o2[3] - THYMIO_W:g}", "very wide since the added row"],
    ]
    y = table(c, x, y, [62 * mm, 30 * mm, 40 * mm, 62 * mm], rows)

    # right column: conversions and notes
    rx = 216 * mm
    ry = H - 24 * mm
    c.setFont("Helvetica-Bold", 9)
    c.drawString(rx, ry, "Conversions")
    ry -= 5 * mm
    notes = [
        "Rectified map with the four outer marker corners mapped to the image corners. "
        f"The arena is {fmt(ARENA_W)} x {fmt(ARENA_H)} cm (ratio {ARENA_W / ARENA_H:.3f}), no longer 16:9: "
        "the map must keep this ratio so that px/cm is the same on both axes.",
        f"px = x * W / {fmt(ARENA_W)}",
        f"py = ({fmt(ARENA_H)} - y) * H / {fmt(ARENA_H)}",
        "(image y axis points down)",
        f"For W x H = {RECT_W} x {RECT_H}: {RECT_PX_PER_CM:g} px/cm. Obstacle = 115 px, marker = 96 px, "
        "marker cell = 16 px.",
        f"Vision.py map: {VISION_MAP[0]} x {VISION_MAP[1]} px, {VISION_MAP[0] / ARENA_W:g} px/cm.",
        f"Arena centre: ({fmt(ARENA_W / 2)}, {fmt(ARENA_H / 2)}).",
        f"Top row (y {fmt(PREVIOUS_ARENA_H)} - {fmt(ARENA_H)}) added above the original "
        f"{fmt(ARENA_W)} x {fmt(PREVIOUS_ARENA_H)} cm arena: obstacles and bottom markers unchanged, "
        "top markers moved up.",
        "O1 and O2 are shifted 10 cm to the left of the previous layout (35, 17) / (65, 39): "
        f"the band between O2 and the right edge is {ARENA_W - o2[1]:g} cm wide.",
    ]
    for n in notes:
        ry = wrap_text(c, n, rx, ry, W - rx - 12 * mm, size=7.2) - 1.5 * mm

    # assembly / printing instructions
    y -= 9 * mm
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, "Printing and assembly")
    y -= 5 * mm
    full_w, full_h = ARENA_W * 10 + 2 * MARGIN_MM, ARENA_H * 10 + 2 * MARGIN_MM
    steps = [
        f"1. Plotter: print arena_full_scale_1to1.pdf ({fmt(full_w)} x {fmt(full_h)} mm) at 100 % on white matte paper. "
        "Do not trim the 2 cm white margin (ArUco quiet zone). Check the 100 mm scale bar.",
        f"2. No plotter: print arena_tiled_A3_1to1.pdf ({TILES[0] * TILES[1]} x A3, 100 %), trim only joining edges, "
        "butt-join and tape on the back. No tile seam crosses an obstacle or a marker.",
        f"3. Alternative: on any white matte {fmt(ARENA_W)} x {fmt(ARENA_H)} cm surface, place the 4 cut markers of aruco_markers_A3.pdf "
        "(outer corner on the arena corner, 'top' arrow towards +y) and the 2 obstacles of obstacles_sheet_A3.pdf "
        "at the coordinates above.",
        "4. Mount the camera so the four markers are fully visible with some white around them; no other dark objects "
        f"inside the arena. The arena is now {fmt(ARENA_H)} cm deep: the camera may have to be mounted higher.",
    ]
    for s_ in steps:
        y = wrap_text(c, s_, x, y, 194 * mm, size=7.4, leading=9.2) - 1.2 * mm


def sheet_navigation(c):
    W, H = landscape(A4)
    c.scale(K, K)
    title_block(c, W, H, 3, 3, "Thymio experimental arena - navigation analysis", scale_text(NAV_SCALE))
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(14 * mm, H - 13 * mm, "LAYOUT RATIONALE - navigation analysis")

    f = Frame(24, 84, draw_mm_per_cm(NAV_SCALE))

    # arena
    c.setStrokeColor(black)
    c.setLineWidth(0.8)
    c.rect(f.x(0), f.y(0), f.len(ARENA_W), f.len(ARENA_H), stroke=1, fill=0)
    for mid, _, centre in MARKERS:
        draw_marker(c, f, mid, centre)

    # inflated obstacles (configuration space for the robot centre)
    for _, (cx, cy) in OBSTACLES:
        s = OBSTACLE_SIZE + 2 * INFLATION
        c.setStrokeColor(GREY_MID)
        c.setFillColor(Color(0.94, 0.94, 0.94))
        c.setLineWidth(0.4)
        c.setDash([2, 1.5])
        c.roundRect(f.x(cx) - f.len(s / 2), f.y(cy) - f.len(s / 2), f.len(s), f.len(s),
                    f.len(OBSTACLE_RADIUS + INFLATION), stroke=1, fill=1)
        c.setDash([])
    for name, centre in OBSTACLES:
        draw_obstacle(c, f, centre)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 6.5)
        c.drawCentredString(f.x(centre[0]), f.y(centre[1] + 1.2), name)

    # edge band the robot centre cannot enter (half-width)
    c.setStrokeColor(GREY_LIGHT)
    c.setLineWidth(0.4)
    c.setDash([1, 1.5])
    hw = THYMIO_W / 2
    c.rect(f.x(hw), f.y(hw), f.len(ARENA_W - 2 * hw), f.len(ARENA_H - 2 * hw), stroke=1, fill=0)
    c.setDash([])

    draw_previous_edge(c, f, x1=GOAL[0] - 10, label_below=False)   # stops before the G label

    def polyline(points, color, width, dash=None):
        c.saveState()
        c.setStrokeColor(color)
        c.setLineWidth(width)
        if dash:
            c.setDash(dash)
        p = c.beginPath()
        p.moveTo(f.x(points[0][0]), f.y(points[0][1]))
        for px, py in points[1:]:
            p.lineTo(f.x(px), f.y(py))
        c.drawPath(p, stroke=1, fill=0)
        c.restoreState()

    zx0, zy0, zx1, zy1 = LOCAL_ZONE
    c.saveState()
    c.setStrokeColor(BLUE)
    c.setFillColor(Color(BLUE.red, BLUE.green, BLUE.blue, alpha=0.07))
    c.setLineWidth(0.5)
    c.setDash([4, 2])
    c.roundRect(f.x(zx0), f.y(zy0), f.len(zx1 - zx0), f.len(zy1 - zy0), 2 * mm, stroke=1, fill=1)
    c.restoreState()
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", 6.5)
    c.drawCentredString(f.x((zx0 + zx1) / 2), f.y(zy0) + 6 * mm, "local-navigation test zone")
    c.setFont("Helvetica", 6.0)
    c.drawCentredString(f.x((zx0 + zx1) / 2), f.y(zy0) + 3 * mm, "(hand-placed obstacle, not printed)")

    o1, o2 = OBSTACLES[0][1], OBSTACLES[1][1]
    e = OBSTACLE_SIZE / 2 + INFLATION
    path_mid = [START, (o1[0] - e, o1[1] + e), (o1[0] + e, o1[1] + e),
                (o2[0] - e, o2[1] - e), (o2[0] + e, o2[1] - e), GOAL]
    path_bottom = [START, (o1[0] - e, 8.0), (o1[0] + e, 8.0), (o2[0] + e + 1, o2[1] - e - 1), GOAL]
    path_top = [START, (o1[0] - e, o1[1] + e), (o2[0] - e, 49.5), (o2[0] + e, 49.5), GOAL]

    polyline(path_bottom, GREY_MID, 0.8, [3, 2])
    polyline(path_top, GREY_MID, 0.8, [3, 2])
    polyline(path_mid, BLUE, 1.4)
    polyline([START, GOAL], RED, 0.9, [2, 1.5])

    # collisions of the straight line with the robot footprint
    for (cx, cy) in (o1, o2):
        t = (cx - START[0]) / (GOAL[0] - START[0])
        yy = START[1] + t * (GOAL[1] - START[1])
        X, Y = f.x(cx), f.y(yy)
        d = 1.4 * mm
        c.setStrokeColor(RED)
        c.setLineWidth(1.0)
        c.line(X - d, Y - d, X + d, Y + d)
        c.line(X - d, Y + d, X + d, Y - d)

    # start / goal with Thymio footprint
    for (px, py), label, col in ((START, "S", GREEN), (GOAL, "G", RED)):
        c.setStrokeColor(col)
        c.setFillColor(Color(col.red, col.green, col.blue, alpha=0.12))
        c.setLineWidth(0.6)
        c.rect(f.x(px - hw), f.y(py - hw), f.len(THYMIO_W), f.len(THYMIO_W), stroke=1, fill=1)
        c.setFillColor(col)
        c.circle(f.x(px), f.y(py), 1.0 * mm, stroke=0, fill=1)
        c.setFont("Helvetica-Bold", 8)
        ly_txt = f.y(py - hw) - 3.6 * mm if label == "S" else f.y(py + hw) + 1.4 * mm
        c.drawCentredString(f.x(px), ly_txt, f"{label} ({fmt(px)}, {fmt(py)})")

    # gap annotations
    c.setFillColor(GREY_DARK)
    c.setFont("Helvetica", 6.3)
    b1, b2 = obstacle_box(o1), obstacle_box(o2)
    r = OBSTACLE_RADIUS
    gap_diag = math.hypot((b2[0] + r) - (b1[1] - r), (b2[2] + r) - (b1[3] - r)) - 2 * r
    mx, my = (o1[0] + o2[0]) / 2, (o1[1] + o2[1]) / 2
    c.drawString(f.x(mx - 3.5), f.y(my - 7.5), f"{gap_diag:.1f} cm")
    c.drawString(f.x(o1[0] + 4), f.y(4.5), f"{b1[2]:g} cm")
    c.drawCentredString(f.x(o2[0]), f.y((PREVIOUS_ARENA_H + ARENA_H) / 2 + 4),
                        f"{ARENA_H - b2[3]:g} cm from O2 to top edge")
    c.drawCentredString(f.x((b2[1] + ARENA_W) / 2), f.y(LOCAL_ZONE[1]) - 5 * mm,
                        f"{ARENA_W - b2[1]:g} cm free right of O2")

    # legend
    lx, ly = f.x(ARENA_W) + 6 * mm, f.y(ARENA_H) - 2 * mm
    items = [
        (RED, [2, 1.5], 0.9, "Straight line S-G: blocked twice (x)"),
        (BLUE, None, 1.4, "Main route: between O1 and O2"),
        (GREY_MID, [3, 2], 0.8, "Alternative routes (narrow edge passage)"),
        (GREY_MID, [2, 1.5], 0.4, f"Obstacle inflated by {fmt(INFLATION)} cm"),
        (GREY_LIGHT, [1, 1.5], 0.4, "Limit for robot centre (5.5 cm)"),
        (BLUE, [4, 2], 0.8, "Free zone for local navigation"),
    ]
    c.setFont("Helvetica-Bold", 7)
    c.setFillColor(black)
    c.drawString(lx, ly, "LEGEND")
    for col, dash, wdt, txt in items:
        ly -= 5.5 * mm
        c.saveState()
        c.setStrokeColor(col)
        c.setLineWidth(wdt)
        if dash:
            c.setDash(dash)
        c.line(lx, ly + 0.9 * mm, lx + 7 * mm, ly + 0.9 * mm)
        c.restoreState()
        c.setFillColor(black)
        c.setFont("Helvetica", 6.3)
        c.drawString(lx + 9 * mm, ly, txt)
    ly -= 6 * mm
    c.setFont("Helvetica-Oblique", 6.0)
    ly = wrap_text(c, "S and G are suggested test poses only; they are not printed on the arena.",
                   lx, ly, W - lx - 12 * mm, font="Helvetica-Oblique", size=6.0)

    # explanation text in two columns
    def clearance_text(name, centre):
        d = segment_clearance(START, GOAL, centre)
        return f"crosses {name}" if d == 0 else f"passes {d:.1f} cm from {name}"

    paragraphs = [
        ("Diagonal, left-shifted layout.", f"O1 ({fmt(o1[0])}, {fmt(o1[1])}) and O2 ({fmt(o2[0])}, {fmt(o2[1])}) "
         f"stay diagonally offset (x and y ranges {b2[0] - b1[1]:g} cm / {b2[2] - b1[3]:g} cm apart, so they never "
         "form a wall), but the pair is shifted 10 cm to the left of the previous layout (35, 17) / (65, 39)."),
        ("Forced deviation.", f"The straight line S-G {clearance_text('O1', o1)} and {clearance_text('O2', o2)}, "
         "less than the Thymio half-width (5.5 cm): the robot would hit both. The planner must bend the path "
         "twice (above O1, then below O2)."),
        ("Room for local navigation.", f"No printed obstacle lies right of x = {b2[1]:g} cm: the band up to the "
         f"right edge is {ARENA_W - b2[1]:g} cm wide over the full arena height. An obstacle placed by hand on the "
         "route in this zone (unknown to the global map) can be bypassed on either side by the local "
         "avoidance before the robot reaches G."),
        ("Safe free space.", f"The corridor between the obstacles is {gap_diag:.1f} cm wide (about "
         f"{gap_diag / THYMIO_W:.1f} Thymio widths); O1 is {b1[0]:g} cm from the left edge (2 robot widths). "
         "Both obstacles are convex and separated: no pocket or dead end where the robot could get trapped."),
        ("Several routes.", "Besides the central route, the robot can go around O1 from below or O2 from above. "
         f"Below O1 the passage is only {b1[2]:g} cm, about 3 cm wider than the robot, so with an inflation of "
         f"about 6 cm the planner usually rejects it. Above O2 the added row leaves {ARENA_H - b2[3]:g} cm: "
         "a wide route and room for random objects."),
        ("Vision.", "Black matte obstacles on white matte paper give maximum contrast for thresholding. "
         "The outer marker corners coincide with the arena corners, so the homography maps the arena "
         f"exactly onto a {fmt(ARENA_W)} x {fmt(ARENA_H)} image ({VISION_MAP[0]} x {VISION_MAP[1]} px in "
         "Vision.py). It is no longer 16:9: the camera must see all four markers."),
    ]
    gap = 7 * mm
    col_w = (W - 28 * mm - 2 * gap) / 3
    xs = [14 * mm + k * (col_w + gap) for k in range(3)]
    ys = [f.y(0) - 7 * mm] * 3
    for i, (head, body) in enumerate(paragraphs):
        k = (0, 0, 1, 1, 2, 2)[i]
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 7.6)
        c.drawString(xs[k], ys[k], head)
        ys[k] = wrap_text(c, body, xs[k], ys[k] - 3.4 * mm, col_w, size=7.3, leading=9.0) - 2.2 * mm


def open_canvas(c, path, pagesize, title):
    """Returns (canvas, own). own=True: a new file is created at path and must be saved.
    own=False: pages are appended to the shared canvas c (combined document)."""
    if c is None:
        c = canvas.Canvas(path, pagesize=pagesize)
        c.setTitle(title)
        return c, True
    c.setPageSize(pagesize)
    return c, False


def close_canvas(c, own, path):
    if own:
        c.save()
    return path


def build_technical_drawing(c=None):
    path = os.path.join(OUT_DIR, "arena_technical_drawing.pdf")
    c, own = open_canvas(c, path, landscape(PAPER), "Thymio arena - technical drawing")
    sheet_top_view(c)
    c.showPage()
    sheet_coordinates(c)
    c.showPage()
    sheet_navigation(c)
    c.showPage()
    return close_canvas(c, own, path)


# =============================================================================
# 2. Full-scale arena (one large page) and tiled A4 version
# =============================================================================

MARGIN_MM = 20.0


def crop_marks(c, x0, y0, x1, y1, gap=3 * mm, length=8 * mm):
    """L-shaped trim marks outside the rectangle (x0, y0)-(x1, y1)."""
    c.setStrokeColor(GREY_MID)
    c.setLineWidth(0.3)
    for x, sx in ((x0, -1), (x1, 1)):
        for y, sy in ((y0, -1), (y1, 1)):
            c.line(x + sx * gap, y, x + sx * (gap + length), y)
            c.line(x, y + sy * gap, x, y + sy * (gap + length))


def scale_bar(c, x, y, length_mm=100):
    c.setStrokeColor(GREY_DARK)
    c.setFillColor(GREY_DARK)
    c.setLineWidth(0.4)
    c.line(x, y, x + length_mm * mm, y)
    for i in range(0, length_mm + 1, 10):
        h = 2.5 * mm if i % 50 == 0 else 1.5 * mm
        c.line(x + i * mm, y, x + i * mm, y + h)
    c.setFont("Helvetica", 6)
    c.drawString(x + length_mm * mm + 2 * mm, y, f"{length_mm} mm scale check - must measure exactly {length_mm / 10:g} cm")


def build_full_scale(c=None):
    path = os.path.join(OUT_DIR, "arena_full_scale_1to1.pdf")
    W = (ARENA_W * 10 + 2 * MARGIN_MM) * mm
    H = (ARENA_H * 10 + 2 * MARGIN_MM) * mm
    aw, ah = ARENA_W * 10, ARENA_H * 10
    c, own = open_canvas(c, path, (W, H), f"Thymio arena 1:1 ({fmt(aw)} x {fmt(ah)} mm)")
    f = Frame(MARGIN_MM, MARGIN_MM, 10.0)
    draw_arena_content(c, f)
    crop_marks(c, f.x(0), f.y(0), f.x(ARENA_W), f.y(ARENA_H))
    c.setFont("Helvetica", 6.5)
    c.setFillColor(GREY_DARK)
    c.drawString(f.x(5), H - 8 * mm,
                 f"Thymio arena - 1:1 - arena {fmt(aw)} x {fmt(ah)} mm, sheet {fmt(aw + 2 * MARGIN_MM)} x "
                 f"{fmt(ah + 2 * MARGIN_MM)} mm - print at 100 % "
                 "(actual size, no fit-to-page) on white matte paper. Keep the white margin: it is the "
                 "quiet zone the ArUco detector needs.")
    scale_bar(c, f.x(10), 8 * mm)
    c.drawRightString(f.x(ARENA_W - 5), 8 * mm,
                      f"O1 ({fmt(OBSTACLES[0][1][0])}, {fmt(OBSTACLES[0][1][1])}) cm   "
                      f"O2 ({fmt(OBSTACLES[1][1][0])}, {fmt(OBSTACLES[1][1][1])}) cm   ArUco {ARUCO_DICT_NAME} IDs 0-3, origin = bottom-left")
    c.showPage()
    return close_canvas(c, own, path)


def build_tiled(c=None):
    """1:1 arena on 3 x 3 A3 landscape sheets, each tile 333.3 x 281.25 mm."""
    path = os.path.join(OUT_DIR, f"arena_tiled_{PAPER_NAME}_1to1.pdf")
    PW, PH = landscape(PAPER)
    ncol, nrow = TILES
    tw, th = ARENA_W * 10 / ncol, ARENA_H * 10 / nrow   # tile size, mm
    ox = (PW / mm - tw) / 2
    oy = (PH / mm - th) / 2
    c, own = open_canvas(c, path, (PW, PH), f"Thymio arena 1:1 - tiled on {ncol * nrow} {PAPER_NAME} sheets")
    for row in range(nrow - 1, -1, -1):          # print top row first
        for col in range(ncol):
            c.saveState()
            p = c.beginPath()
            p.rect(ox * mm, oy * mm, tw * mm, th * mm)
            c.clipPath(p, stroke=0, fill=0)
            f = Frame(ox - col * tw, oy - row * th, 10.0)
            draw_arena_content(c, f)
            c.restoreState()
            crop_marks(c, ox * mm, oy * mm, (ox + tw) * mm, (oy + th) * mm, gap=1.5 * mm, length=5 * mm)
            # tile label (left margin, rotated) and scale check (right margin, rotated)
            n = (nrow - 1 - row) * ncol + col + 1
            c.saveState()
            c.translate((ox - 6) * mm, oy * mm)
            c.rotate(90)
            c.setFont("Helvetica", 6.5)
            c.setFillColor(GREY_DARK)
            twc, thc = tw / 10, th / 10
            c.drawString(0, 0, f"Tile {n}/{ncol * nrow} - row {nrow - row} (from top), column {col + 1} - "
                               f"x {col * twc:.2f}-{(col + 1) * twc:.2f} cm, "
                               f"y {row * thc:.3f}-{(row + 1) * thc:.3f} cm")
            c.drawString(0, -3.2 * mm, "Print at 100 %. Trim only the edges that join another tile, butt-join, tape on the back.")
            c.restoreState()
            c.saveState()
            c.translate((ox + tw + 9) * mm, (oy + 10) * mm)
            c.rotate(90)
            scale_bar(c, 0, 0, 50)
            c.restoreState()
            c.showPage()
    return close_canvas(c, own, path)


# =============================================================================
# 3. Obstacle sheet and marker sheet (A4 portrait, 1:1)
# =============================================================================

def build_obstacle_sheet(c=None):
    path = os.path.join(OUT_DIR, f"obstacles_sheet_{PAPER_NAME}.pdf")
    PW, PH = PAPER
    c, own = open_canvas(c, path, PAPER, "Thymio arena - obstacles 1:1")
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, PH - 22 * mm, "Obstacles - print at 100 % (actual size)")
    c.setFont("Helvetica", 8.5)
    wrap_text(c, "Exactly two obstacles: black matte rounded squares, 6 x 6 cm, corner radius 1 cm. "
                 "Cut along the dashed line (1 cm white rim). The four ticks on the cut line point to the "
                 "obstacle centre: align them with the centre lines drawn lightly in pencil on the arena.",
              20 * mm, PH - 30 * mm, PW - 40 * mm, size=8.5)

    f = Frame(0, 0, 10.0)
    centres_mm = [(PW / mm / 2 - 25, PH / mm - 140.0), (PW / mm / 2 - 25, PH / mm - 260.0)]
    for (name, (ax, ay)), (cx_mm, cy_mm) in zip(OBSTACLES, centres_mm):
        centre = (cx_mm / 10, cy_mm / 10)        # Frame(0,0,10): 1 cm -> 10 mm
        draw_obstacle(c, f, centre)
        # dashed cut line (1 cm rim)
        rim = (OBSTACLE_SIZE + 2.0) * 10 * mm
        X, Y = cx_mm * mm, cy_mm * mm
        c.saveState()
        c.setStrokeColor(GREY_MID)
        c.setLineWidth(0.4)
        c.setDash([3, 2])
        c.roundRect(X - rim / 2, Y - rim / 2, rim, rim, 20 * mm, stroke=1, fill=0)
        c.restoreState()
        # centre ticks
        c.setStrokeColor(GREY_DARK)
        c.setLineWidth(0.5)
        t = 3 * mm
        c.line(X, Y + rim / 2, X, Y + rim / 2 - t)
        c.line(X, Y - rim / 2, X, Y - rim / 2 + t)
        c.line(X - rim / 2, Y, X - rim / 2 + t, Y)
        c.line(X + rim / 2, Y, X + rim / 2 - t, Y)
        # labels
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(X + rim / 2 + 8 * mm, Y + 6 * mm, f"Obstacle {name}")
        c.setFont("Helvetica", 9)
        c.drawString(X + rim / 2 + 8 * mm, Y + 0.5 * mm, f"centre at ({fmt(ax)}, {fmt(ay)}) cm")
        c.setFont("Helvetica", 7.5)
        c.setFillColor(GREY_DARK)
        c.drawString(X + rim / 2 + 8 * mm, Y - 4.5 * mm, f"{fmt(ax)} cm from the left edge,")
        c.drawString(X + rim / 2 + 8 * mm, Y - 8.5 * mm, f"{fmt(ay)} cm from the bottom edge")

    scale_bar(c, 20 * mm, 30 * mm)
    c.setFont("Helvetica", 7)
    c.setFillColor(GREY_DARK)
    c.drawString(20 * mm, 20 * mm, "Origin (0, 0) = bottom-left arena corner (marker ID 0). Do not add other obstacles.")
    c.showPage()
    return close_canvas(c, own, path)


def build_marker_sheet(c=None):
    path = os.path.join(OUT_DIR, f"aruco_markers_{PAPER_NAME}.pdf")
    PW, PH = PAPER
    c, own = open_canvas(c, path, PAPER, f"Thymio arena - ArUco markers {ARUCO_DICT_NAME} IDs 0-3")
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, PH - 22 * mm, f"ArUco markers {ARUCO_DICT_NAME} - print at 100 % (actual size)")
    c.setFont("Helvetica", 8.5)
    wrap_text(c, "Each marker is 5 x 5 cm including its black border. Cut along the dashed line, keeping the "
                 "white rim. Place each marker so that its outer black corner touches the arena corner, with "
                 "the 'top' arrow pointing to the top edge of the arena (+y). Sheet layout = arena layout.",
              20 * mm, PH - 30 * mm, PW - 40 * mm, size=8.5)

    f = Frame(0, 0, 10.0)
    xl, xr_, yt, yb = PW / mm / 2 - 55, PW / mm / 2 + 55, PH / mm - 140.0, PH / mm - 260.0
    pos_mm = {3: (xl, yt), 2: (xr_, yt), 0: (xl, yb), 1: (xr_, yb)}
    for mid, name, (ax, ay) in MARKERS:
        cx_mm, cy_mm = pos_mm[mid]
        draw_marker(c, f, mid, (cx_mm / 10, cy_mm / 10))
        X, Y = cx_mm * mm, cy_mm * mm
        rim = 66 * mm
        c.saveState()
        c.setStrokeColor(GREY_MID)
        c.setLineWidth(0.4)
        c.setDash([3, 2])
        c.rect(X - rim / 2, Y - rim / 2, rim, rim, stroke=1, fill=0)
        c.restoreState()
        # orientation arrow above the cut line
        c.setFillColor(GREY_DARK)
        p = c.beginPath()
        p.moveTo(X, Y + rim / 2 + 6 * mm)
        p.lineTo(X - 2 * mm, Y + rim / 2 + 2.5 * mm)
        p.lineTo(X + 2 * mm, Y + rim / 2 + 2.5 * mm)
        p.close()
        c.drawPath(p, fill=1, stroke=0)
        c.setFont("Helvetica", 7)
        c.drawString(X + 3 * mm, Y + rim / 2 + 2.8 * mm, "top (+y)")
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(X, Y - rim / 2 - 6 * mm, f"ID {mid} - {name}")
        c.setFont("Helvetica", 8)
        corner = (0 if ax < ARENA_W / 2 else ARENA_W, 0 if ay < ARENA_H / 2 else ARENA_H)
        c.drawCentredString(X, Y - rim / 2 - 10.5 * mm,
                            f"centre ({fmt(ax)}, {fmt(ay)}) cm - outer corner at ({fmt(corner[0])}, {fmt(corner[1])})")

    scale_bar(c, 20 * mm, 30 * mm)
    c.setFont("Helvetica", 7)
    c.setFillColor(GREY_DARK)
    c.drawString(20 * mm, 20 * mm,
                 "Detection: cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))")
    c.showPage()
    return close_canvas(c, own, path)


BUILDERS = (build_technical_drawing, build_full_scale, build_tiled,
            build_obstacle_sheet, build_marker_sheet)


def build_combined():
    """All documents in one PDF (pages keep their own physical size)."""
    path = os.path.join(OUT_DIR, "arena_complete.pdf")
    c = canvas.Canvas(path, pagesize=landscape(PAPER))
    c.setTitle("Thymio arena - complete set (drawing, 1:1 arena, tiles, obstacles, markers)")
    for build in BUILDERS:
        build(c)
    c.save()
    return path


if __name__ == "__main__":
    for build in BUILDERS + (build_combined,):
        print("written:", build())
