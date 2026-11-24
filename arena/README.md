# Thymio experimental arena: final obstacle configuration

Arena 100 × 56.25 cm (16:9, like the 1920 × 1080 Aukey 1080p camera, which is mounted vertically above the arena), white matte. It has 4 ArUco markers and **exactly 2** black matte obstacles.
Origin (0, 0) is the bottom-left corner of the arena. x points right, y points up, and all values are in cm.

## Files

| File | Content | Print |
|---|---|---|
| `arena_complete.pdf` | **Everything in one PDF** (12 pages, in the order below). Each page keeps its real size | Print only the pages you need, at **100 %** |
| `arena_technical_drawing.pdf` | Sheet 1: dimensioned top view. Sheet 2: coordinate tables + assembly. Sheet 3: navigation analysis | A3 landscape (sheets 1 and 3 at 1:4) |
| `arena_full_scale_1to1.pdf` | Full arena with markers and obstacles, plus a 2 cm white margin | 1040 × 602.5 mm, **100 %** |
| `arena_tiled_A3_1to1.pdf` | Same arena split on 6 A3 tiles (3 × 2, 33.33 × 28.125 cm each). No seam crosses an obstacle or a marker | A3 landscape, **100 %** |
| `obstacles_sheet_A3.pdf` | The two 6 × 6 cm obstacles, with cut line and centre ticks | A3, **100 %** |
| `aruco_markers_A3.pdf` | The four 5 × 5 cm markers, IDs 0–3, laid out like the arena | A3, **100 %** |
| `generate_arena.py` | Generates all of the above (`pip install reportlab opencv-python`) | – |

Every 1:1 sheet has a scale bar. Check it with a ruler after printing, and turn off "fit to page".

## Obstacles

Black matte rounded squares, 6 × 6 cm, corner radius 1 cm (convex).

| Obstacle | Centre (x, y) | x range | y range | Centre in a 1920 × 1080 rectified map (px) |
|---|---|---|---|---|
| O1 | **(25, 17)** | 22 – 28 | 14 – 20 | (480.0, 753.6) |
| O2 | **(55, 39)** | 52 – 58 | 36 – 42 | (1056.0, 331.2) |

Both obstacles are shifted 10 cm to the left of the previous layout, (35, 17) / (65, 39). This keeps the right part of the arena free for local-navigation tests. To change the positions, edit `OBSTACLES` in `generate_arena.py` and run it again.

## ArUco markers

Dictionary **`DICT_4X4_50`**. Each marker is 5 × 5 cm including its black border, which gives a cell of 0.833 cm. Markers are printed upright, so the marker's top edge faces +y. The **outer corner of each marker is the arena corner**.

| ID | Corner | Centre (x, y) | x range | y range |
|---|---|---|---|---|
| 0 | bottom-left (origin) | (2.5, 2.5) | 0 – 5 | 0 – 5 |
| 1 | bottom-right | (97.5, 2.5) | 95 – 100 | 0 – 5 |
| 2 | top-right | (97.5, 53.75) | 95 – 100 | 51.25 – 56.25 |
| 3 | top-left | (2.5, 53.75) | 0 – 5 | 51.25 – 56.25 |

To convert to pixels, map the outer marker corners to the image corners of a W × H map:
`px = x · W / 100`, `py = (56.25 − y) · H / 56.25` (the image y axis points down).

## Clearances (edge to edge)

| Passage | Width | Space left for an 11 cm Thymio |
|---|---|---|
| Between O1 and O2 (closest rounded corners) | 29.6 cm | 18.6 cm |
| O2 to right edge | 42 cm | 31 cm (local-navigation area) |
| O1 to left edge | 22 cm | 11 cm |
| O1 to top edge / O2 to bottom edge | 36.25 / 36 cm | 25 cm |
| O1 to bottom edge | 14 cm | 3 cm (narrow) |
| O2 to top edge | 14.25 cm | 3.25 cm (narrow) |

## Layout rationale

- **Diagonal, shifted left.** O1 (25, 17) and O2 (55, 39) are still diagonally offset. Their x and y ranges do not overlap (24 cm apart in x, 16 cm in y), so together they never form a wall across the arena.
- **Forced deviation.** A straight path from S = (11, 11) to G = (88, 48) crosses O1 and passes 2.5 cm from O2. That is less than the Thymio half-width (5.5 cm), so the planner has to bend the path twice: above O1, then below O2.
- **Room for local navigation.** No printed obstacle lies to the right of x = 58 cm, so there is a free band 42 cm wide over the full height of the arena. An obstacle you place by hand on the route there is unknown to the global map. The local avoidance can go around it on either side before the robot reaches G. Sheet 3 of the drawing marks a suggested zone, x 63–85 and y 22–42.
- **Safe free space.** The main corridor between the obstacles is 29.6 cm (about 2.7 robot widths), O1 is 22 cm from the left edge, and the open sides are 36 cm and more. Both obstacles are convex and well separated, so there is no pocket or dead end where the robot could get trapped.
- **Several routes.** The robot can also go around O1 from below or O2 from above. These edge passages are only about 3 cm wider than the robot, so once obstacles are inflated by about 6 cm a planner will usually reject them. They remain physically usable as fall-back routes.
- **Vision.** Black matte obstacles on white matte paper give maximum contrast. The marker outer corners coincide with the arena corners, so the homography maps the arena exactly onto a 16:9 image. In a 1920 × 1080 map that is 19.2 px/cm: an obstacle is 115 px wide and a marker cell is 16 px.

S, G and the local-navigation zone are only suggestions for the analysis. They are not printed on the arena.
