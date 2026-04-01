#!/usr/bin/env python
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import ezdxf
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

WIDTH = 3940.0
HOLE_OFFSET = 89.4
HOLE_SIDE = 1292.2
HOLE_SHRINK = 15.0
HOLE_CROSS_WIDTH = 15.0

SQUARE_ORIGIN_X = 2620.0 + 15
SQUARE_ORIGIN_TOP = WIDTH - 114.4 - 15
SQUARE_SIDE = 220.0
SQUARE_GAP = 280.0
SQUARE_STEP = SQUARE_SIDE + SQUARE_GAP
N_COLS = 16
N_ROWS = 8

BAR_Y_TOP = WIDTH - HOLE_OFFSET - HOLE_SIDE - 250.0
BAR_Y_BOTTOM = WIDTH - HOLE_OFFSET - HOLE_SIDE - 2250.0
BAR_X_LEFT = 300.0
BAR_X_RIGHT = 350.0
BAR_WIDTH = BAR_X_RIGHT - BAR_X_LEFT
BAR_SPACING = 300.0
N_BARS = 6

ADD_RECT_X0 = 1392.2 + 100.0
ADD_RECT_Y_TOP = WIDTH - HOLE_OFFSET
ADD_RECT_Y_BOTTOM = WIDTH - HOLE_OFFSET - 300.0
ADD_RECT_WIDTH = 1000.0
ADD_RECT_DY = 1000.0

GRID_CELL_SIZE_FINE = 15.0
GRID_CELL_SIZE_COARSE = 2000.0
REFINE_BUFFER = 5.0

rect_pts = [(0, 0), (0, WIDTH), (10385, WIDTH), (10385, 0)]
hole_pts = [
    (100 + HOLE_SHRINK, WIDTH - HOLE_OFFSET - HOLE_SHRINK),
    (100 + HOLE_SHRINK, WIDTH - HOLE_OFFSET - HOLE_SIDE + HOLE_SHRINK),
    (100 + HOLE_SIDE - HOLE_SHRINK, WIDTH - HOLE_OFFSET - HOLE_SIDE + HOLE_SHRINK),
    (100 + HOLE_SIDE - HOLE_SHRINK, WIDTH - HOLE_OFFSET - HOLE_SHRINK),
]


def build_adaptive_cells(rect, x0, y0, x1, y1, size, kong_buffer):
    cells = []
    cell = box(x0, y0, x1, y1)
    if not rect.intersects(cell):
        return cells
    if size <= GRID_CELL_SIZE_FINE:
        return [cell]
    near = kong_buffer.intersects(cell)
    if near:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        half = size / 2
        for (a0, b0, a1, b1) in [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)]:
            cells.extend(build_adaptive_cells(rect, a0, b0, a1, b1, half, kong_buffer))
    else:
        cells.append(cell)
    return cells


def _build_cells_for_block(args):
    rect, kong_buffer, x0, y0, x1, y1 = args
    return build_adaptive_cells(rect, x0, y0, x1, y1, GRID_CELL_SIZE_COARSE, kong_buffer)


def _intersect_batch(args):
    cells_batch, merged = args
    out = []
    for c in cells_batch:
        p = c.intersection(merged)
        if not p.is_empty:
            out.append(p)
    return out


def main():
    rect = Polygon(rect_pts)
    hole = Polygon(hole_pts)

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add("background", color=1)
    doc.layers.add("地线孔", color=2)
    doc.layers.add("地线孔十字", color=6)
    doc.layers.add("io", color=3)
    doc.layers.add("过孔1", color=4)
    doc.layers.add("过孔2", color=5)
    doc.layers.add("merged", color=7)

    def write_geom(geom, layer):
        if geom.is_empty:
            return
        if geom.geom_type == "Polygon":
            ext = list(geom.exterior.coords)[:-1]
            if len(ext) >= 3:
                msp.add_lwpolyline([(float(x), float(y)) for x, y in ext], close=True, dxfattribs={"layer": layer})
            for interior in geom.interiors:
                pts = list(interior.coords)[:-1]
                if len(pts) >= 3:
                    msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": layer})
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                write_geom(g, layer)

    write_geom(rect, "background")
    write_geom(hole, "地线孔")

    hl = 100.0 + HOLE_SHRINK
    hr = 100.0 + HOLE_SIDE - HOLE_SHRINK
    ht = WIDTH - HOLE_OFFSET - HOLE_SHRINK
    hb = WIDTH - HOLE_OFFSET - HOLE_SIDE + HOLE_SHRINK
    cx = (hl + hr) / 2.0
    cy = (ht + hb) / 2.0
    hw = HOLE_CROSS_WIDTH / 2.0
    vert_pts = [(cx - hw, hb), (cx - hw, ht), (cx + hw, ht), (cx + hw, hb)]
    horz_pts = [(hl, cy - hw), (hl, cy + hw), (hr, cy + hw), (hr, cy - hw)]
    for pts in (vert_pts, horz_pts):
        msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": "地线孔十字"})

    for row in range(N_ROWS):
        for col in range(N_COLS):
            left = SQUARE_ORIGIN_X + col * SQUARE_STEP
            top = SQUARE_ORIGIN_TOP - row * SQUARE_STEP
            pts = [
                (left, top),
                (left, top - SQUARE_SIDE),
                (left + SQUARE_SIDE, top - SQUARE_SIDE),
                (left + SQUARE_SIDE, top),
            ]
            msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": "io"})

    for i in range(N_BARS):
        x_left = BAR_X_LEFT + i * (BAR_WIDTH + BAR_SPACING)
        x_right = x_left + BAR_WIDTH
        pts = [
            (x_left, BAR_Y_TOP),
            (x_left, BAR_Y_BOTTOM),
            (x_right, BAR_Y_BOTTOM),
            (x_right, BAR_Y_TOP),
        ]
        msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": "过孔1"})

    for dy in (0.0, -ADD_RECT_DY):
        pts = [
            (ADD_RECT_X0, ADD_RECT_Y_TOP + dy),
            (ADD_RECT_X0, ADD_RECT_Y_BOTTOM + dy),
            (ADD_RECT_X0 + ADD_RECT_WIDTH, ADD_RECT_Y_BOTTOM + dy),
            (ADD_RECT_X0 + ADD_RECT_WIDTH, ADD_RECT_Y_TOP + dy),
        ]
        msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": "过孔2"})

    cross = unary_union([
        Polygon(vert_pts),
        Polygon(horz_pts),
    ])
    hole_minus_cross = hole.difference(cross)
    io_polys = []
    for row in range(N_ROWS):
        for col in range(N_COLS):
            left = SQUARE_ORIGIN_X + col * SQUARE_STEP
            top = SQUARE_ORIGIN_TOP - row * SQUARE_STEP
            io_polys.append(Polygon([
                (left, top), (left, top - SQUARE_SIDE),
                (left + SQUARE_SIDE, top - SQUARE_SIDE), (left + SQUARE_SIDE, top),
            ]))
    过孔1_polys = []
    for i in range(N_BARS):
        x_left = BAR_X_LEFT + i * (BAR_WIDTH + BAR_SPACING)
        x_right = x_left + BAR_WIDTH
        过孔1_polys.append(Polygon([
            (x_left, BAR_Y_TOP), (x_left, BAR_Y_BOTTOM),
            (x_right, BAR_Y_BOTTOM), (x_right, BAR_Y_TOP),
        ]))
    过孔2_polys = []
    for dy in (0.0, -ADD_RECT_DY):
        过孔2_polys.append(Polygon([
            (ADD_RECT_X0, ADD_RECT_Y_TOP + dy), (ADD_RECT_X0, ADD_RECT_Y_BOTTOM + dy),
            (ADD_RECT_X0 + ADD_RECT_WIDTH, ADD_RECT_Y_BOTTOM + dy),
            (ADD_RECT_X0 + ADD_RECT_WIDTH, ADD_RECT_Y_TOP + dy),
        ]))
    镂空 = unary_union([hole_minus_cross] + io_polys + 过孔1_polys + 过孔2_polys)
    if hasattr(镂空, "make_valid"):
        镂空 = 镂空.make_valid()

    kong_buffer = 镂空.buffer(REFINE_BUFFER)
    blocks = []
    nx = int(10385.0 // GRID_CELL_SIZE_COARSE) + 1
    ny = int(WIDTH // GRID_CELL_SIZE_COARSE) + 1
    for i in range(nx):
        for j in range(ny):
            x0 = i * GRID_CELL_SIZE_COARSE
            y0 = j * GRID_CELL_SIZE_COARSE
            x1 = min(x0 + GRID_CELL_SIZE_COARSE, 10385.0)
            y1 = min(y0 + GRID_CELL_SIZE_COARSE, WIDTH)
            blocks.append((rect, kong_buffer, x0, y0, x1, y1))

    n_workers = min(multiprocessing.cpu_count(), 8)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_build_cells_for_block, blocks))
    adaptive_cells = []
    for r in results:
        adaptive_cells.extend(r)

    merged = rect.difference(镂空)
    merged = merged.union(cross)
    if hasattr(merged, "make_valid"):
        merged = merged.make_valid()

    chunk_size = max(1, len(adaptive_cells) // (n_workers * 4))
    chunks = [adaptive_cells[i:i + chunk_size] for i in range(0, len(adaptive_cells), chunk_size)]
    batch_args = [(chunk, merged) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        pieces_nested = list(ex.map(_intersect_batch, batch_args))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for pieces in pieces_nested:
        for piece in pieces:
            write_geom(piece, "merged")
    out_path = os.path.join(RESULTS_DIR, "io_AZ.dxf")
    doc.saveas(out_path)
    print(out_path)

    merged_doc = ezdxf.new("R2010")
    merged_msp = merged_doc.modelspace()
    merged_doc.layers.add("merged", color=7)

    def write_merged_geom(geom):
        if geom.is_empty:
            return
        if geom.geom_type == "Polygon":
            ext = list(geom.exterior.coords)[:-1]
            if len(ext) >= 3:
                merged_msp.add_lwpolyline([(float(x), float(y)) for x, y in ext], close=True, dxfattribs={"layer": "merged"})
            for interior in geom.interiors:
                pts = list(interior.coords)[:-1]
                if len(pts) >= 3:
                    merged_msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": "merged"})
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                write_merged_geom(g)

    for pieces in pieces_nested:
        for piece in pieces:
            write_merged_geom(piece)
    merged_path = os.path.join(RESULTS_DIR, "io_AZ_merged.dxf")
    merged_doc.saveas(merged_path)
    print(merged_path)


if __name__ == "__main__":
    main()
