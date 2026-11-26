# Thymio experimental arena: final obstacle configuration

Arena **100 × 84.375 cm**, white matte, printed on **3 × 3 A3 sheets**. The camera (Aukey 1080p, 1920 × 1080) is mounted vertically above the arena. The arena has 4 ArUco markers and **exactly 2** black matte obstacles.
Origin (0, 0) is the bottom-left corner of the arena. x points right, y points up, and all values are in cm.

**Added row.** The arena used to be 100 × 56.25 cm (16:9, 3 × 2 sheets). A third row of sheets was added on top (y 56.25 – 84.375) to get free space for random objects. The origin, the obstacles and the bottom markers did not move. Only the two top markers (ID 2 and ID 3) moved up to the new top corners. The arena is no longer 16:9, so `Vision.py` uses a 256 × 216 px map (2.56 px/cm on both axes) and the camera may have to be mounted higher to see the four markers.

## Files

| File | Content | Print |
|---|---|---|
| `arena_complete.pdf` | **Everything in one PDF** (15 pages, in the order below). Each page keeps its real size | Print only the pages you need, at **100 %** |
| `arena_technical_drawing.pdf` | Sheet 1: dimensioned top view (1:5). Sheet 2: coordinate tables + assembly. Sheet 3: navigation analysis (1:6) | A3 landscape |
| `arena_full_scale_1to1.pdf` | Full arena with markers and obstacles, plus a 2 cm white margin | 1040 × 883.75 mm, **100 %** |
| `arena_tiled_A3_1to1.pdf` | Same arena split on 9 A3 tiles (3 × 3, 33.33 × 28.125 cm each). No seam crosses an obstacle or a marker | A3 landscape, **100 %** |
| `obstacles_sheet_A3.pdf` | The two 6 × 6 cm obstacles, with cut line and centre ticks | A3, **100 %** |
| `aruco_markers_A3.pdf` | The four 5 × 5 cm markers, IDs 0–3, laid out like the arena | A3, **100 %** |
| `generate_arena.py` | Generates all of the above (`pip install reportlab opencv-python`) | – |

Every 1:1 sheet has a scale bar. Check it with a ruler after printing, and turn off "fit to page".

Tiles 1–3 are the added top row (tiles 1 and 3 carry markers ID 3 and ID 2). Tiles 4–9 are the original 6 tiles, unchanged except that the top markers are no longer on tiles 4 and 6.

## Obstacles

Black matte rounded squares, 6 × 6 cm, corner radius 1 cm (convex).

| Obstacle | Centre (x, y) | x range | y range | Centre in a 1920 × 1620 rectified map (px) |
|---|---|---|---|---|
| O1 | **(25, 17)** | 22 – 28 | 14 – 20 | (480.0, 1293.6) |
| O2 | **(55, 39)** | 52 – 58 | 36 – 42 | (1056.0, 871.2) |

Both obstacles are shifted 10 cm to the left of the previous layout, (35, 17) / (65, 39). This keeps the right part of the arena free for local-navigation tests. To change the positions, edit `OBSTACLES` in `generate_arena.py` and run it again.

## ArUco markers

Dictionary **`DICT_4X4_50`**. Each marker is 5 × 5 cm including its black border, which gives a cell of 0.833 cm. Markers are printed upright, so the marker's top edge faces +y. The **outer corner of each marker is the arena corner**.

| ID | Corner | Centre (x, y) | x range | y range |
|---|---|---|---|---|
| 0 | bottom-left (origin) | (2.5, 2.5) | 0 – 5 | 0 – 5 |
| 1 | bottom-right | (97.5, 2.5) | 95 – 100 | 0 – 5 |
| 2 | top-right | (97.5, 81.875) | 95 – 100 | 79.375 – 84.375 |
| 3 | top-left | (2.5, 81.875) | 0 – 5 | 79.375 – 84.375 |

To convert to pixels, map the outer marker corners to the image corners of a W × H map with W / H = 100 / 84.375:
`px = x · W / 100`, `py = (84.375 − y) · H / 84.375` (the image y axis points down).

## Clearances (edge to edge)

| Passage | Width | Space left for an 11 cm Thymio |
|---|---|---|
| Between O1 and O2 (closest rounded corners) | 29.6 cm | 18.6 cm |
| O2 to right edge | 42 cm | 31 cm (local-navigation area) |
| O1 to left edge | 22 cm | 11 cm |
| O1 to top edge / O2 to bottom edge | 64.375 / 36 cm | 25 cm |
| O1 to bottom edge | 14 cm | 3 cm (narrow) |
| O2 to top edge | 42.375 cm | 31.375 cm (wide since the added row) |

## Layout rationale

- **Diagonal, shifted left.** O1 (25, 17) and O2 (55, 39) are still diagonally offset. Their x and y ranges do not overlap (24 cm apart in x, 16 cm in y), so together they never form a wall across the arena.
- **Forced deviation.** A straight path from S = (11, 11) to G = (88, 48) crosses O1 and passes 2.5 cm from O2. That is less than the Thymio half-width (5.5 cm), so the planner has to bend the path twice: above O1, then below O2.
- **Room for local navigation.** No printed obstacle lies to the right of x = 58 cm, so there is a free band 42 cm wide over the full height of the arena. An obstacle you place by hand on the route there is unknown to the global map. The local avoidance can go around it on either side before the robot reaches G. Sheet 3 of the drawing marks a suggested zone, x 63–85 and y 22–42.
- **Free space for random objects.** The added top row (y 56.25 – 84.375) has no printed obstacle. Objects placed there are seen by the camera like any other obstacle.
- **Safe free space.** The main corridor between the obstacles is 29.6 cm (about 2.7 robot widths), O1 is 22 cm from the left edge, and the open sides are 36 cm and more. Both obstacles are convex and well separated, so there is no pocket or dead end where the robot could get trapped.
- **Several routes.** The robot can also go around O1 from below or O2 from above. Below O1 the passage is only about 3 cm wider than the robot, so once obstacles are inflated by about 6 cm a planner will usually reject it. Above O2 the added row leaves 42.375 cm, so that route is now wide.
- **Vision.** Black matte obstacles on white matte paper give maximum contrast. The marker outer corners coincide with the arena corners, so the homography maps the arena exactly onto a 100 × 84.375 image: 256 × 216 px in `Vision.py` (2.56 px/cm), or 1920 × 1620 px at 19.2 px/cm. An obstacle is then 115 px wide and a marker cell is 16 px.

S, G and the local-navigation zone are only suggestions for the analysis. They are not printed on the arena.
